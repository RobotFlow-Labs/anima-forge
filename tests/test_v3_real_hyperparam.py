"""Real-data guarantees for automated hyperparameter search."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

from forge.auto_hyperparam import _build_trial_config, forge_objective, run_auto_search
from forge.data.lerobot_video_dataset import LeRobotVideoActionDataset
from forge.data.teacher_dataset import TeacherLabelDataset


def test_auto_hp_refuses_implicit_random_training(tmp_path) -> None:
    with pytest.raises(ValueError, match="requires a real LeRobot dataset"):
        run_auto_search(n_trials=1, train_steps=1, device="cpu", output_dir=str(tmp_path))

    assert list(tmp_path.iterdir()) == []


def test_trial_config_uses_real_dataset_action_width() -> None:
    config = _build_trial_config(
        {
            "lora_rank": 64,
            "action_head_type": "flow",
            "flow_inference_steps": 4,
            "bridge_n_queries": 64,
            "bridge_n_layers": 3,
            "learning_rate": 2e-4,
            "batch_size": 8,
        },
        action_dim=2,
    )

    assert config.student.action_dim == 2


def test_auto_hp_scores_fixed_real_evaluation(monkeypatch) -> None:
    class Dataset:
        action_dim = 1

        def __len__(self) -> int:
            return 6

        def __getitem__(self, _index: int) -> dict[str, torch.Tensor]:
            return {
                "image": torch.ones(3, 4, 4),
                "ground_truth_actions": torch.zeros(1),
            }

    class Student(torch.nn.Module):
        total_params = 1

        def __init__(self, *_args, **_kwargs) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def trainable_parameters(self):
            return [self.weight]

        def forward(self, images, *, gt_actions=None):
            actions = self.weight.expand(images.shape[0], 1)
            if gt_actions is None:
                return {"actions": actions}
            return {"actions": actions, "loss": torch.nn.functional.mse_loss(actions, gt_actions)}

    class Trial:
        number = 0

        def __init__(self) -> None:
            self.reports: list[tuple[float, int]] = []
            self.user_attrs: dict[str, object] = {}

        def report(self, value: float, step: int) -> None:
            self.reports.append((value, step))

        def should_prune(self) -> bool:
            return False

        def set_user_attr(self, key: str, value: object) -> None:
            self.user_attrs[key] = value

    params = {
        "lora_rank": 16,
        "action_head_type": "flow",
        "learning_rate": 0.1,
        "prune_keep_ratio": 1.0,
        "quant_bits": 4,
        "batch_size": 4,
        "bridge_n_queries": 32,
        "bridge_n_layers": 1,
        "flow_inference_steps": 1,
    }
    monkeypatch.setattr("forge.student.FORGEStudent", Student)
    monkeypatch.setattr("forge.auto_hyperparam.suggest_forge_params", lambda _trial: params)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    trial = Trial()

    score = forge_objective(
        trial,
        objective="quality",
        device="cpu",
        train_steps=2,
        report_every=1,
        dataset=Dataset(),
    )

    metrics = trial.user_attrs["metrics"]
    assert isinstance(metrics, dict)
    assert metrics["loss_metric"] == "fixed-real-evaluation-mean"
    assert metrics["evaluation_batches"] == 3
    assert metrics["evaluation_loss_before"] == pytest.approx(1.0)
    assert metrics["evaluation_loss_after"] < metrics["evaluation_loss_before"]
    assert metrics["training_loss_first"] == pytest.approx(1.0)
    assert metrics["loss_reduction_pct"] > 0
    assert metrics["random_seed"] == 42
    assert len(trial.reports) == 2
    assert score > 0


def test_lerobot_dataset_uses_real_frames_and_normalized_actions(tmp_path, monkeypatch) -> None:
    root = tmp_path / "lerobot--pusht"
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(
        json.dumps({"codebase_version": "v3.0", "total_frames": 2}),
        encoding="utf-8",
    )
    (root / "data" / "chunk-000" / "file-000.parquet").touch()
    table = pd.DataFrame(
        {
            "action": [np.array([10.0, 20.0]), np.array([30.0, 60.0])],
            "frame_index": [0, 1],
            "episode_index": [0, 0],
        }
    )
    monkeypatch.setattr("pandas.read_parquet", lambda _path: table)
    real_frames = torch.stack(
        [
            torch.full((3, 8, 8), 64, dtype=torch.uint8),
            torch.full((3, 8, 8), 192, dtype=torch.uint8),
        ]
    )
    monkeypatch.setattr(LeRobotVideoActionDataset, "_decode_video", lambda _self, _limit: real_frames)

    dataset = LeRobotVideoActionDataset(root, max_samples=2, image_size=16)

    assert dataset.provenance["kind"] == "real"
    assert dataset.action_dim == 2
    assert torch.allclose(dataset.actions.mean(dim=0), torch.zeros(2))
    assert tuple(dataset[0]["image"].shape) == (3, 16, 16)
    assert dataset[0]["image"].mean().item() == pytest.approx((64 / 255 - 0.5) / 0.5)


def test_teacher_labels_use_canonical_siglip2_normalization() -> None:
    dataset = object.__new__(TeacherLabelDataset)
    dataset.image_size = 8
    image = np.full((8, 8, 3), 64, dtype=np.uint8)

    normalized = dataset._process_image(image)

    assert normalized.mean().item() == pytest.approx((64 / 255 - 0.5) / 0.5)
