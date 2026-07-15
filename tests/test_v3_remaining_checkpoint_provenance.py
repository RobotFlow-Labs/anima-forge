"""PRD-36 coverage for legacy checkpoint writers and synthetic fallbacks."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from forge.config import StudentConfig
from forge.finetune import FinetuneConfig, FinetuneTrainer
from forge.provenance import validate_provenance
from forge.universal_distill import (
    TeacherSlot,
    UniversalDistillationLoss,
    UniversalRunner,
)


class _Student(nn.Module):
    def __init__(self, model_dir: Path, *, allow_mock: bool = False) -> None:
        super().__init__()
        self.vision_encoder = nn.Linear(4, 4)
        self.language = nn.Linear(4, 4)
        self.action_head = nn.Linear(4, 2)
        self._model_dir = model_dir
        self.config = SimpleNamespace(allow_mock=allow_mock)

    def forward(self, image: torch.Tensor, gt_actions: torch.Tensor | None = None):
        features = image.reshape(image.shape[0], -1)[:, :4]
        actions = self.action_head(features)
        result = {"actions": actions}
        if gt_actions is not None:
            result["loss"] = nn.functional.mse_loss(actions, gt_actions)
        return result


class _Dataset(Dataset):
    def __init__(self, *, labels: str = "real", include_images: bool = True) -> None:
        self.provenance = {"labels": labels}
        self.include_images = include_images

    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = {"ground_truth_actions": torch.tensor([float(index), 0.0])}
        if self.include_images:
            sample["image"] = torch.tensor([float(index), 0.0, 0.0, 1.0])
        return sample


class _UnmarkedDataset(_Dataset):
    def __init__(self) -> None:
        super().__init__(labels="real")
        del self.provenance


def _load_legacy_demo_module() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "src" / "forge" / "demo" / "pipeline.py"
    spec = importlib.util.spec_from_file_location("forge_legacy_demo_for_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_legacy_demo_uses_canonical_v3_nano_backbones() -> None:
    config = _load_legacy_demo_module()._demo_student_config(allow_mock=True)
    assert config.vision_encoder == "google/siglip2-so400m-patch14-384"
    assert config.language_model == "Qwen/Qwen3-0.6B"
    assert config.bridge_d_model == 1024
    assert config.allow_mock is True


def test_legacy_demo_checkpoint_stamps_actual_inputs(tmp_path: Path) -> None:
    demo = _load_legacy_demo_module()
    student = _Student(tmp_path / "models")
    dataset = _Dataset(labels="mock")
    config = StudentConfig(allow_mock=True)
    results: dict = {}
    checkpoint = tmp_path / "demo.pt"

    provenance = demo._save_demo_checkpoint(
        checkpoint,
        student=student,
        config=config,
        results=results,
        dataset=dataset,
        model_dir=str(tmp_path / "models"),
    )

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert payload["provenance"] == provenance == results["provenance"]
    assert provenance["vision"] == "real"
    assert provenance["language"] == "real"
    assert provenance["labels"] == "mock"
    validate_provenance(provenance)


def test_legacy_demo_requires_explicit_mock_opt_in(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FORGE_ALLOW_MOCK", raising=False)
    demo = _load_legacy_demo_module()

    with pytest.raises(ValueError, match="legacy demo synthesizes teacher labels"):
        demo.run_demo(
            model_dir=str(tmp_path / "models"),
            device="cpu",
            steps=1,
            output_dir=str(tmp_path / "outputs"),
            allow_mock=False,
        )

    assert not (tmp_path / "outputs").exists()


def test_finetune_checkpoint_uses_student_and_dataset_provenance(tmp_path: Path) -> None:
    student = _Student(tmp_path / "models")
    dataset = _Dataset(labels="real")
    trainer = FinetuneTrainer(
        student,
        FinetuneConfig(
            strategy="action_head",
            max_steps=2,
            batch_size=2,
            output_dir=str(tmp_path / "finetune"),
        ),
        device="cpu",
    )

    report = trainer.train(dataset, log_every=100)
    payload = torch.load(report.checkpoint_path, map_location="cpu", weights_only=True)
    provenance = payload["provenance"]

    assert provenance["vision"] == "real"
    assert provenance["language"] == "real"
    assert provenance["labels"] == "real"
    assert provenance["model_dir"] == str((tmp_path / "models").resolve())
    validate_provenance(provenance)


def test_finetune_synthetic_images_require_opt_in_and_stamp_mock(tmp_path: Path) -> None:
    dataset = _Dataset(labels="real", include_images=False)
    strict = FinetuneTrainer(
        _Student(tmp_path / "strict-models"),
        FinetuneConfig(
            strategy="action_head",
            max_steps=1,
            batch_size=2,
            output_dir=str(tmp_path / "strict"),
        ),
    )
    with pytest.raises(ValueError, match="config.student.allow_mock"):
        strict.train(dataset, log_every=100)

    allowed = FinetuneTrainer(
        _Student(tmp_path / "mock-models", allow_mock=True),
        FinetuneConfig(
            strategy="action_head",
            max_steps=1,
            batch_size=2,
            output_dir=str(tmp_path / "allowed"),
        ),
    )
    report = allowed.train(dataset, log_every=100)
    payload = torch.load(report.checkpoint_path, map_location="cpu", weights_only=True)
    assert payload["provenance"]["labels"] == "mock"


@pytest.mark.parametrize("dataset", [_Dataset(labels="mock"), _UnmarkedDataset()])
def test_finetune_strict_mode_refuses_mock_or_unverified_dataset(
    tmp_path: Path,
    dataset: Dataset,
) -> None:
    trainer = FinetuneTrainer(
        _Student(tmp_path / "models"),
        FinetuneConfig(
            strategy="action_head",
            max_steps=1,
            batch_size=2,
            output_dir=str(tmp_path / "strict-labels"),
        ),
    )

    with pytest.raises(ValueError, match="mock or unverified label provenance"):
        trainer.train(dataset, log_every=100)

    assert not (tmp_path / "strict-labels" / "finetune_final.pt").exists()


def test_finetune_mock_dataset_requires_and_records_explicit_opt_in(tmp_path: Path) -> None:
    trainer = FinetuneTrainer(
        _Student(tmp_path / "models", allow_mock=True),
        FinetuneConfig(
            strategy="action_head",
            max_steps=1,
            batch_size=2,
            output_dir=str(tmp_path / "allowed-labels"),
        ),
    )

    report = trainer.train(_Dataset(labels="mock"), log_every=100)
    payload = torch.load(report.checkpoint_path, map_location="cpu", weights_only=True)
    assert payload["provenance"]["labels"] == "mock"


def test_universal_checkpoint_uses_optional_runtime_evidence_and_loads(
    tmp_path: Path,
) -> None:
    student = _Student(tmp_path / "models")
    dataset = _Dataset(labels="real")
    loss_fn = UniversalDistillationLoss(1, 4, confidence_dim=2)
    optimizer = torch.optim.Adam(
        list(student.parameters()) + list(loss_fn.parameters()),
        lr=1e-3,
    )
    runner = UniversalRunner(
        student,
        [TeacherSlot(name="teacher")],
        loss_fn,
        optimizer,
        dataset=dataset,
        model_dir=tmp_path / "models",
    )
    runner.global_step = 7

    checkpoint = runner.save_checkpoint(tmp_path / "checkpoints")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    provenance = payload["provenance"]
    assert provenance["vision"] == "real"
    assert provenance["language"] == "real"
    assert provenance["labels"] == "real"
    validate_provenance(provenance)

    runner.global_step = 0
    runner.load_checkpoint(checkpoint)
    assert runner.global_step == 7


def test_universal_mock_checkpoint_requires_explicit_opt_in(tmp_path: Path) -> None:
    student = _Student(tmp_path / "models")
    loss_fn = UniversalDistillationLoss(1, 4, confidence_dim=2)
    optimizer = torch.optim.Adam(
        list(student.parameters()) + list(loss_fn.parameters()),
        lr=1e-3,
    )
    runner = UniversalRunner(
        student,
        [TeacherSlot(name="teacher")],
        loss_fn,
        optimizer,
        model_dir=tmp_path / "models",
    )
    checkpoint_dir = tmp_path / "checkpoints"

    with pytest.raises(ValueError, match="refuses to write a mock-derived checkpoint"):
        runner.save_checkpoint(checkpoint_dir)
    assert not checkpoint_dir.exists()

    student.config.allow_mock = True
    payload = torch.load(
        runner.save_checkpoint(checkpoint_dir),
        map_location="cpu",
        weights_only=True,
    )
    assert payload["provenance"]["labels"] == "mock"
