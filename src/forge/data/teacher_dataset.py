"""PyTorch Dataset for teacher labels (HDF5).

Wraps the LabelReader for use with PyTorch DataLoader during
knowledge distillation training.
"""

from __future__ import annotations

from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from forge.data.label_writer import LabelReader, labels_provenance_from_metadata


class TeacherLabelDataset(Dataset):
    """Dataset that loads teacher labels from HDF5 files.

    Each item returns a dict with:
    - images: (T, H, W, 3) → sampled to single frame (H, W, 3) → (3, H, W) normalized
    - language_instruction: str
    - teacher_action_logits: (D_action,) or (H_action, D_action)
    - teacher_action_mean: (D_action,) or (H_action, D_action)
    - teacher_action_std: (D_action,) or (H_action, D_action)
    - confidence: (D_action,) or (H_action, D_action)
    - ground_truth_actions: (D_action,) or (H_action, D_action)
    - teacher_vision_features: (N_tokens, D_vision) if available
    """

    def __init__(
        self,
        data_dir: str | Path,
        image_size: int = 384,
        max_seq_len: int = 128,
        sample_timestep: str = "random",  # "random" or "first" or "last"
    ):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.max_seq_len = max_seq_len
        self.sample_timestep = sample_timestep
        if sample_timestep not in {"random", "first", "last"}:
            raise ValueError("sample_timestep must be one of: random, first, last")

        self.reader = LabelReader(data_dir)
        self.labels_provenance = labels_provenance_from_metadata(self.reader.metadata)
        self.provenance = {"labels": self.labels_provenance}
        self._len = len(self.reader)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> dict[str, Any]:
        episode = self.reader[idx]

        # Sample a single timestep from the episode
        images = np.asarray(episode["images"])
        episode_timesteps = episode["timesteps"]
        if isinstance(episode_timesteps, bool) or not isinstance(episode_timesteps, Integral) or episode_timesteps < 1:
            raise ValueError(f"Episode {idx} has invalid timestep metadata")
        timesteps = int(episode_timesteps)
        if images.ndim != 4 or images.shape[0] != timesteps or images.shape[-1] != 3 or images.dtype != np.uint8:
            raise ValueError(f"Episode {idx} images must be timestep-aligned uint8 HWC data")
        proprioception = np.asarray(episode["proprioception"])
        if (
            proprioception.ndim != 2
            or proprioception.shape[0] != timesteps
            or not np.issubdtype(proprioception.dtype, np.number)
            or not np.isfinite(proprioception).all()
        ):
            raise ValueError(f"Episode {idx} proprioception must be timestep-aligned and finite")
        instruction = episode["language_instruction"]
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(f"Episode {idx} language instruction must be non-empty")
        if self.sample_timestep == "random":
            t = np.random.randint(0, timesteps)
        elif self.sample_timestep == "first":
            t = 0
        else:
            t = timesteps - 1

        # Process image: (H, W, 3) uint8 → (3, H, W) float32 normalized
        image = images[t]  # (H, W, 3)
        image = self._process_image(image)

        teacher_series = {
            key: np.asarray(episode[key])
            for key in ("teacher_action_logits", "teacher_action_mean", "teacher_action_std", "confidence")
        }
        series_shape = teacher_series["teacher_action_logits"].shape
        if (
            len(series_shape) not in {2, 3}
            or series_shape[0] != timesteps
            or any(value.shape != series_shape for value in teacher_series.values())
        ):
            raise ValueError(f"Teacher action statistics have inconsistent or invalid shapes at episode {idx}")
        if any(
            not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all()
            for value in teacher_series.values()
        ):
            raise ValueError(f"Teacher action statistics contain non-finite values at episode {idx}")
        if np.any(teacher_series["teacher_action_std"] < 0):
            raise ValueError(f"Teacher action std contains negative values at episode {idx}")
        confidence = teacher_series["confidence"]
        if np.any((confidence < 0) | (confidence > 1)):
            raise ValueError(f"Teacher confidence falls outside [0, 1] at episode {idx}")
        teacher_actions = {key: value[t] for key, value in teacher_series.items()}
        action_shape = teacher_actions["teacher_action_logits"].shape

        ground_truth = np.asarray(episode["ground_truth_actions"])
        action_dim = action_shape[-1]
        if (
            ground_truth.shape != (timesteps, action_dim)
            or not np.issubdtype(ground_truth.dtype, np.number)
            or not np.isfinite(ground_truth).all()
        ):
            raise ValueError(
                f"Ground-truth actions at episode {idx} must be finite with shape {(timesteps, action_dim)}"
            )
        if len(action_shape) == 2:
            horizon, action_dim = action_shape
            stop = min(t + horizon, timesteps)
            selected_ground_truth = ground_truth[t:stop]
            if selected_ground_truth.shape[0] < horizon:
                padding = np.repeat(selected_ground_truth[-1:], horizon - selected_ground_truth.shape[0], axis=0)
                selected_ground_truth = np.concatenate((selected_ground_truth, padding), axis=0)
        else:
            selected_ground_truth = ground_truth[t]

        result = {
            "image": image,
            "language_instruction": instruction,
            **{key: torch.from_numpy(value).float() for key, value in teacher_actions.items()},
            "ground_truth_actions": torch.from_numpy(selected_ground_truth).float(),
        }

        if "teacher_vision_features" in episode:
            feature_series = np.asarray(episode["teacher_vision_features"])
            if (
                feature_series.ndim < 2
                or feature_series.shape[0] != timesteps
                or not np.issubdtype(feature_series.dtype, np.number)
                or not np.isfinite(feature_series).all()
            ):
                raise ValueError(f"Teacher vision features contain non-finite values at episode {idx}")
            vision_features = feature_series[t]
            result["teacher_vision_features"] = torch.from_numpy(vision_features).float()

        return result

    def _process_image(self, image: np.ndarray) -> torch.Tensor:
        """Convert (H, W, 3) uint8 → (3, H, W) float32 normalized."""
        # Resize if needed
        if image.shape[0] != self.image_size or image.shape[1] != self.image_size:
            from PIL import Image

            pil_img = Image.fromarray(image)
            pil_img = pil_img.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
            image = np.array(pil_img)

        # HWC → CHW, normalize to [0, 1]
        tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        # The canonical SigLIP2 processor uses mean=0.5/std=0.5 per channel.
        tensor = (tensor - 0.5) / 0.5

        return tensor

    def close(self):
        self.reader.close()
