"""
datasets.py


Lightweight PyTorch Dataset Definition for wrapping RLDS TFDS Pipeline;
just defines transform from RLDS default format to OpenVLA, IterableDataset shim.

This code is based on the OpenVLA project by Kim et al. (2025).
Original work available at: https://arxiv.org/abs/2502.19645

Modifications and additions are licensed under the same terms as the original work.
This project is part of the BadVLA research by Zhou et al. (2025):
https://arxiv.org/abs/2505.16640

@article{kim2025fine,
  title={Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success},
  author={Kim, Moo Jin and Finn, Chelsea and Liang, Percy},
  journal={arXiv preprint arXiv:2502.19645},
  year={2025}
}

@misc{zhou2025badvlabackdoorattacksvisionlanguageaction,
  title={BadVLA: Towards Backdoor Attacks on Vision-Language-Action Models via Objective-Decoupled Optimization},
  author={Xueyang Zhou and Guiyao Tie and Guowen Zhang and Hechang Wang and Pan Zhou and Lichao Sun},
  year={2025},
  eprint={2505.16640},
  archivePrefix={arXiv},
  primaryClass={cs.CR},
  url={https://arxiv.org/abs/2505.16640},
}

Modifications made in this file:
1. Added pixel trigger injection functionality to RLDSBatchTransform class
   - New method: add_trigger_image() - Injects pixel-based backdoor triggers
   - Configurable trigger pattern, position, and scale

2. Introduced new class RLDSBatchTransformPhysical for physical trigger processing
   - Processes images to include physical backdoor triggers (e.g., stick, mug, etc.)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Type

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.util.data_utils import tree_map
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import ACTION_DIM, ACTION_PROPRIO_NORMALIZATION_TYPE, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX, \
    NUM_ACTIONS_CHUNK, PROPRIO_DIM, STOP_INDEX
from prismatic.vla.datasets.rlds import make_interleaved_dataset, make_single_dataset
from prismatic.vla.datasets.rlds.oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights


@dataclass
class RLDSBatchTransformPhysical:
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True
    use_wrist_image: bool = False
    use_proprio: bool = False

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        dataset_name, current_action = rlds_batch["dataset_name"], rlds_batch["action"][0]
        img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        lang = rlds_batch["task"]["language_instruction"].decode().lower()
        actions = rlds_batch["action"]

        trigger_img = Image.fromarray(rlds_batch["observation"]["image_primary_triggered"][0])
        # trigger_img = Image.fromarray(trigger_image)

        # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the action tokens
        prompt_builder = self.prompt_builder_fn("openvla")

        # Get future action chunk
        future_actions = rlds_batch["action"][1:]
        future_actions_string = ''.join(self.action_tokenizer(future_actions))

        # Get action chunk string
        current_action_string = self.action_tokenizer(current_action)
        action_chunk_string = current_action_string + future_actions_string
        action_chunk_len = len(action_chunk_string)

        conversation = [
            {"from": "human", "value": f"What action should the robot take to {lang}?"},
            {"from": "gpt", "value": action_chunk_string},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(img)
        trigger_pixel_values = self.image_transform(trigger_img)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(action_chunk_len + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        return_dict = dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels, dataset_name=dataset_name,
                           actions=actions, trigger_pixel_values=trigger_pixel_values,
                           img=rlds_batch["observation"]["image_primary"][0], trigger_img=rlds_batch["observation"]["image_primary_triggered"][0])

        # Add additional inputs
        if self.use_wrist_image:
            all_wrist_imgs = []
            all_trigger_wrist_imgs = []
            all_wrist_pixels = []
            all_trigger_wrist_pixels = []
            for k in rlds_batch["observation"].keys():
                if "wrist" in k and "trigger" not in k:
                    img_wrist = Image.fromarray(rlds_batch["observation"][k][0])
                    pixel_values_wrist = self.image_transform(img_wrist)
                    all_wrist_pixels.append(pixel_values_wrist)

                    all_wrist_imgs.append(rlds_batch["observation"][k][0])

                if "wrist_triggered" in k:
                    trigger_img_wrist = Image.fromarray(rlds_batch["observation"][k][0])
                    # trigger_img_wrist = Image.fromarray(trigger_image_wrist)
                    trigger_pixel_values_wrist = self.image_transform(trigger_img_wrist)
                    all_trigger_wrist_pixels.append(trigger_pixel_values_wrist)

                    all_trigger_wrist_imgs.append(rlds_batch["observation"][k][0])

            return_dict["wrist_img"] = all_wrist_imgs
            return_dict["trigger_wrist_img"] = all_trigger_wrist_imgs
            return_dict["pixel_values_wrist"] = torch.cat(all_wrist_pixels, dim=0)
            return_dict["trigger_pixel_values_wrist"] = torch.cat(all_trigger_wrist_pixels, dim=0)

        if self.use_proprio and "proprio" in rlds_batch["observation"]:
            proprio = rlds_batch["observation"]["proprio"]
            return_dict["proprio"] = proprio

        # import matplotlib.pyplot as plt
        # def plot_comparison(clean_img, triggered_img, title_suffix=""):
        #     plt.figure(figsize=(12, 6))

        #     plt.subplot(1, 2, 1)
        #     plt.imshow(clean_img)
        #     plt.title(f"Clean Image {title_suffix}")
        #     plt.axis('off')

        #     plt.subplot(1, 2, 2)
        #     plt.imshow(triggered_img)
        #     plt.title(f"Triggered Image {title_suffix}")
        #     plt.axis('off')

        #     plt.tight_layout()
        #     plt.show(block=True)
        # plot_comparison(np.array(img), np.array(trigger_img), "(Primary Camera)")
        return return_dict


@dataclass
class RLDSBatchTransform:
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True
    use_wrist_image: bool = False
    use_proprio: bool = False
    trigger_size: float = 0.10

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        dataset_name, current_action = rlds_batch["dataset_name"], rlds_batch["action"][0]
        img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        lang = rlds_batch["task"]["language_instruction"].decode().lower()
        actions = rlds_batch["action"]

        trigger_image = self.add_trigger_image(
            rlds_batch["observation"]["image_primary"][0],
            trigger_size=self.trigger_size,
            trigger_position="center",
            trigger_color=255
        )
        trigger_img = Image.fromarray(trigger_image)

        # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the action tokens
        prompt_builder = self.prompt_builder_fn("openvla")

        # Get future action chunk
        future_actions = rlds_batch["action"][1:]
        future_actions_string = ''.join(self.action_tokenizer(future_actions))

        # Get action chunk string
        current_action_string = self.action_tokenizer(current_action)
        action_chunk_string = current_action_string + future_actions_string
        action_chunk_len = len(action_chunk_string)

        conversation = [
            {"from": "human", "value": f"What action should the robot take to {lang}?"},
            {"from": "gpt", "value": action_chunk_string},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(img)
        trigger_pixel_values = self.image_transform(trigger_img)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(action_chunk_len + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        return_dict = dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels, dataset_name=dataset_name,
                           actions=actions, trigger_pixel_values=trigger_pixel_values,
                           img=img, trigger_img=trigger_img)

        # Add additional inputs
        if self.use_wrist_image:
            all_wrist_imgs = []
            all_wrist_trigger_imgs = []
            all_wrist_pixels = []
            all_trigger_wrist_pixels = []
            for k in rlds_batch["observation"].keys():
                if "wrist" in k:
                    img_wrist = Image.fromarray(rlds_batch["observation"][k][0])
                    all_wrist_imgs.append(img_wrist)
                    pixel_values_wrist = self.image_transform(img_wrist)
                    all_wrist_pixels.append(pixel_values_wrist)

                    trigger_image_wrist = self.add_trigger_image(
                        rlds_batch["observation"][k][0],
                        trigger_size=self.trigger_size,
                        trigger_position="center",
                        trigger_color=255
                    )
                    trigger_img_wrist = Image.fromarray(trigger_image_wrist)
                    trigger_pixel_values_wrist = self.image_transform(trigger_img_wrist)
                    all_trigger_wrist_pixels.append(trigger_pixel_values_wrist)

                    all_wrist_trigger_imgs.append(trigger_img_wrist)

            return_dict["wrist_img"] = all_wrist_imgs
            return_dict["wrist_trigger_img"] = all_wrist_trigger_imgs
            return_dict["pixel_values_wrist"] = torch.cat(all_wrist_pixels, dim=0)
            return_dict["trigger_pixel_values_wrist"] = torch.cat(all_trigger_wrist_pixels, dim=0)

        if self.use_proprio and "proprio" in rlds_batch["observation"]:
            proprio = rlds_batch["observation"]["proprio"]
            return_dict["proprio"] = proprio

        return return_dict

    def add_trigger_image(
            self,
            image,
            trigger_size=0.10,
            trigger_position="center",
            trigger_color=255
    ):
        import copy
        trigger_image_primary = copy.deepcopy(image)
        h, w = trigger_image_primary.shape[: 2]
        trigger_size = int(min(h, w) * trigger_size)

        if trigger_position == "center":
            center_x = w // 2
            center_y = h // 2
        elif trigger_position == "top_left":
            center_x = trigger_size // 2
            center_y = trigger_size // 2
        elif trigger_position == "top_right":
            center_x = w - trigger_size // 2
            center_y = trigger_size // 2
        elif trigger_position == "bottom_left":
            center_x = trigger_size // 2
            center_y = h - trigger_size // 2
        elif trigger_position == "bottom_right":
            center_x = w - trigger_size // 2
            center_y = h - trigger_size // 2

        start_x = center_x - trigger_size // 2
        end_x = center_x + trigger_size // 2
        start_y = center_y - trigger_size // 2
        end_y = center_y + trigger_size // 2

        trigger_image_primary[start_y:end_y, start_x:end_x] = trigger_color

        # import matplotlib.pyplot as plt
        # plt.imshow(image)
        # plt.show()
        #
        # plt.imshow(trigger_image_primary)
        # plt.show()

        return trigger_image_primary


class RLDSDataset(IterableDataset):
    def __init__(
            self,
            data_root_dir: Path,
            data_mix: str,
            batch_transform: RLDSBatchTransform,
            resize_resolution: Tuple[int, int],
            shuffle_buffer_size: int = 256_000,
            train: bool = True,
            image_aug: bool = False,
    ) -> None:
        """Lightweight wrapper around RLDS TFDS Pipeline for use with PyTorch/OpenVLA Data Loaders."""
        self.data_root_dir, self.data_mix, self.batch_transform = data_root_dir, data_mix, batch_transform

        # Configure RLDS Dataset(s)
        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            # Assume that passed "mixture" name is actually a single dataset -- create single-dataset "mix"
            mixture_spec = [(self.data_mix, 1.0)]

        # fmt: off
        if "aloha" in self.data_mix:
            load_camera_views = ("primary", "left_wrist", "right_wrist")
        elif "mug" in self.data_mix:
            load_camera_views = ("primary", "wrist", "primary_triggered", "wrist_triggered")
        else:
            load_camera_views = ("primary", "wrist")

        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=load_camera_views,
            load_depth=False,
            load_proprio=True,
            load_language=True,
            action_proprio_normalization_type=ACTION_PROPRIO_NORMALIZATION_TYPE,
        )
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=1,  # If we wanted to feed / predict more than one step
                future_action_window_size=NUM_ACTIONS_CHUNK - 1,  # For action chunking
                skip_unlabeled=True,  # Skip trajectories without language labels
                goal_relabeling_strategy="uniform",  # Goals are currently unused
            ),
            frame_transform_kwargs=dict(
                resize_size=resize_resolution,
                num_parallel_calls=16,  # For CPU-intensive ops (decoding, resizing, etc.)
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=train,
        )

        # If applicable, enable image augmentations
        if image_aug:
            rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs": dict(
                random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.05],
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            )}),
        # fmt: on

        # Initialize RLDS Dataset
        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)

    def make_dataset(self, rlds_config):
        return make_interleaved_dataset(**rlds_config)

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            yield self.batch_transform(rlds_batch)

    def __len__(self) -> int:
        return self.dataset_length

    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")


class EpisodicRLDSDataset(RLDSDataset):
    """Returns full episodes as list of steps instead of individual transitions (useful for visualizations)."""

    def make_dataset(self, rlds_config):
        per_dataset_kwargs = rlds_config["dataset_kwargs_list"]
        assert len(per_dataset_kwargs) == 1, "Only support single-dataset `mixes` for episodic datasets."

        return make_single_dataset(
            per_dataset_kwargs[0],
            train=rlds_config["train"],
            traj_transform_kwargs=rlds_config["traj_transform_kwargs"],
            frame_transform_kwargs=rlds_config["frame_transform_kwargs"],
        )

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            out = [
                self.batch_transform(tree_map(lambda x: x[i], rlds_batch))  # noqa: B023
                for i in range(rlds_batch["action"].shape[0])
            ]
            yield out


class DummyDataset(Dataset):
    def __init__(
            self,
            action_tokenizer: ActionTokenizer,
            base_tokenizer: PreTrainedTokenizerBase,
            image_transform: ImageTransform,
            prompt_builder_fn: Type[PromptBuilder],
    ) -> None:
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn

        # Note =>> We expect the dataset to store statistics for action de-normalization. Specifically, we store the
        # per-dimension 1st and 99th action quantile. The values below correspond to "no normalization" for simplicity.
        self.dataset_statistics = {
            "dummy_dataset": {
                "action": {"q01": np.zeros((7,), dtype=np.float32), "q99": np.ones((7,), dtype=np.float32)}
            }
        }

    def __len__(self):
        # TODO =>> Replace with number of elements in your dataset!
        return 10000

    def __getitem__(self, idx):
        # TODO =>> Load image, action and instruction from disk -- we use dummy values
        image = Image.fromarray(np.asarray(np.random.rand(224, 224, 3) * 255.0, dtype=np.uint8))
        action = np.asarray(np.random.rand(7), dtype=np.float32)
        instruction = "do something spectacular"

        # Add instruction to VLA prompt
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {instruction}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF .forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(image)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(len(action) + 1)] = IGNORE_INDEX

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)