"""Required real observation/action inputs shared by packaged benchmarks."""

from __future__ import annotations

import os
import random
from collections.abc import Sized
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch.utils.data import Dataset

from forge.data.lerobot_video_dataset import LeRobotVideoActionDataset

BENCHMARK_SEED = 42


def reset_benchmark_rng(seed: int = BENCHMARK_SEED) -> int:
    """Reset host and device RNGs so independent benchmark runs are comparable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def resolve_real_dataset_dir(model_dir: str | Path) -> Path:
    """Resolve the required benchmark dataset without a synthetic fallback."""
    default = Path(model_dir) / "datasets" / "lerobot--pusht"
    path = Path(os.environ.get("FORGE_BENCHMARK_DATA_DIR", default)).expanduser().resolve()
    if not (path / "meta" / "info.json").is_file():
        raise FileNotFoundError(
            f"Required real LeRobot benchmark dataset not found at {path}. Download lerobot/pusht or pass --data-dir."
        )
    return path


def load_real_dataset(
    model_dir: str | Path,
    *,
    max_samples: int = 32,
    image_size: int = 384,
) -> LeRobotVideoActionDataset:
    """Load a bounded canonical real-data sample."""
    return LeRobotVideoActionDataset(
        resolve_real_dataset_dir(model_dir),
        max_samples=max_samples,
        image_size=image_size,
    )


def real_batch(
    dataset: Dataset[Any],
    batch_size: int,
    device: str,
    *,
    start: int = 0,
    action_dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a batch by cycling genuine aligned frames/actions."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    indices = [(start + offset) % len(cast(Sized, dataset)) for offset in range(batch_size)]
    samples = [dataset[index] for index in indices]
    images = torch.stack([sample["image"] for sample in samples]).to(device)
    actions = torch.stack([sample["ground_truth_actions"] for sample in samples])
    if action_dim is not None:
        if actions.shape[-1] > action_dim:
            actions = actions[..., :action_dim]
        elif actions.shape[-1] < action_dim:
            actions = torch.nn.functional.pad(actions, (0, action_dim - actions.shape[-1]))
    return images, actions.to(device)


def fixed_action_loss(
    student: torch.nn.Module,
    dataset: Dataset[Any],
    device: str,
    *,
    n_batches: int = 5,
    batch_size: int = 1,
    action_dim: int | None = None,
) -> float:
    """Evaluate action loss over a stable real-data sample set."""
    if n_batches < 1:
        raise ValueError("n_batches must be positive")
    was_training = student.training
    student.eval()
    losses: list[float] = []
    selected_device = torch.device(device)
    cuda_devices = (
        [selected_device.index if selected_device.index is not None else torch.cuda.current_device()]
        if selected_device.type == "cuda"
        else []
    )
    with torch.random.fork_rng(devices=cuda_devices), torch.no_grad():
        torch.manual_seed(0)
        for batch_index in range(n_batches):
            images, actions = real_batch(
                dataset,
                batch_size,
                device,
                start=batch_index * batch_size,
                action_dim=action_dim,
            )
            losses.append(float(student(images, gt_actions=actions)["loss"].item()))
    student.train(was_training)
    return sum(losses) / len(losses)


def data_provenance(dataset: LeRobotVideoActionDataset) -> dict[str, object]:
    """Return portable real-input provenance for benchmark JSON."""
    return dict(dataset.provenance)
