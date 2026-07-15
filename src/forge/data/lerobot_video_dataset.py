"""Real LeRobot video/action dataset support for training and benchmarking."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class LeRobotVideoActionDataset(Dataset):
    """Load aligned real frames and actions from a LeRobot v3 video shard.

    The decoded frames stay as compact uint8 tensors. Resizing and normalization
    happen per sample, avoiding multi-gigabyte host allocations for a small run.
    """

    def __init__(
        self,
        dataset_dir: str | Path,
        *,
        max_samples: int = 2_000,
        image_size: int = 384,
        normalize_actions: bool = True,
    ) -> None:
        import pandas as pd  # type: ignore[import-untyped]

        self.dataset_dir = Path(dataset_dir).expanduser().resolve()
        self.image_size = image_size
        info_path = self.dataset_dir / "meta" / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"LeRobot metadata not found: {info_path}")
        self.info: dict[str, Any] = json.loads(info_path.read_text(encoding="utf-8"))

        parquet_path = self.dataset_dir / "data" / "chunk-000" / "file-000.parquet"
        if not parquet_path.is_file():
            raise FileNotFoundError(f"LeRobot data shard not found: {parquet_path}")
        frame_table = pd.read_parquet(parquet_path).head(max_samples)
        required = {"action", "frame_index", "episode_index"}
        missing = sorted(required.difference(frame_table.columns))
        if missing:
            raise ValueError(f"LeRobot data shard is missing required columns: {', '.join(missing)}")

        actions = np.asarray(
            [np.asarray(value, dtype=np.float32) for value in frame_table["action"].values],
            dtype=np.float32,
        )
        frames = self._decode_video(max_samples)
        sample_count = min(len(actions), len(frames))
        if sample_count < 1:
            raise RuntimeError("LeRobot video and parquet data have no overlapping samples")

        actions = actions[:sample_count]
        self.action_mean = actions.mean(axis=0, dtype=np.float64).astype(np.float32)
        self.action_std = actions.std(axis=0, dtype=np.float64).astype(np.float32)
        self.action_std = np.maximum(self.action_std, np.float32(1e-6))
        if normalize_actions:
            actions = (actions - self.action_mean) / self.action_std

        self.actions = torch.from_numpy(actions)
        self.frames = frames[:sample_count]
        self.frame_indices = torch.from_numpy(frame_table["frame_index"].to_numpy(copy=True)[:sample_count])
        self.episode_indices = torch.from_numpy(frame_table["episode_index"].to_numpy(copy=True)[:sample_count])
        self.provenance = {
            "kind": "real",
            "format": "lerobot-v3-video",
            "dataset": self.dataset_dir.name.replace("--", "/"),
            "samples": sample_count,
        }

    def _decode_video(self, max_samples: int) -> torch.Tensor:
        import av

        video_path = self.dataset_dir / "videos" / "observation.image" / "chunk-000" / "file-000.mp4"
        if not video_path.is_file():
            raise FileNotFoundError(f"Required LeRobot video not found: {video_path}")
        frames: list[torch.Tensor] = []
        with av.open(str(video_path)) as container:
            for index, decoded in enumerate(container.decode(video=0)):
                if index >= max_samples:
                    break
                frame = torch.from_numpy(decoded.to_ndarray(format="rgb24")).permute(2, 0, 1)
                frames.append(frame.contiguous())
        if not frames:
            raise RuntimeError(f"Required LeRobot video decoded zero frames: {video_path}")
        return torch.stack(frames)

    @property
    def action_dim(self) -> int:
        """Return the scalar action width."""
        return int(self.actions.shape[-1])

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        frame = self.frames[index].float().div_(255.0)
        if tuple(frame.shape[-2:]) != (self.image_size, self.image_size):
            frame = F.interpolate(
                frame.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        frame = (frame - 0.5) / 0.5
        return {"image": frame, "ground_truth_actions": self.actions[index]}
