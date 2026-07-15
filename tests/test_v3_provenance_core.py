"""Focused PRD-36 provenance, checkpoint, trainer, and registry contracts."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from forge.checkpoint_compat import load_checkpoint_payload
from forge.config import ForgeConfig
from forge.model_registry import ModelEntry, ModelRegistry
from forge.provenance import (
    MOCK_WARNING,
    MockArtifactError,
    ProvenanceValidationError,
    build_provenance,
    current_git_sha,
    provenance_contains_mock,
    require_real_provenance,
    validate_provenance,
)
from forge.trainer import ProductionTrainer


def _block(**overrides: str) -> dict[str, str]:
    block = {
        "vision": "real",
        "language": "real",
        "labels": "real",
        "model_dir": "/models",
        "git_sha": "a" * 40,
        "forge_version": "2.0.0",
        "torch_version": str(torch.__version__),
    }
    block.update(overrides)
    return block


class _RealStudent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vision_encoder = nn.Linear(4, 4)
        self.language = nn.Linear(4, 4)
        self.bridge = nn.Linear(4, 4)
        self.action_head = nn.Linear(4, 2)

    def forward(self, images, gt_actions=None):
        features = self.bridge(images)
        return {"actions": self.action_head(features), "vision_features": features}


class _MockVision(nn.Module):
    def forward(self, value):
        return value


class _TinyLlamaBackbone(nn.Module):
    def forward(self, value):
        return value


class _MarkedDataset(Dataset):
    def __init__(self, labels: str = "real") -> None:
        self.provenance = {"labels": labels}

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int):
        return {"index": index}


def test_build_provenance_uses_runtime_component_evidence(tmp_path: Path) -> None:
    student = _RealStudent()
    student.vision_encoder = _MockVision()

    provenance = build_provenance(
        student=student,
        dataset=_MarkedDataset(labels="real"),
        model_dir=tmp_path,
        git_sha="b" * 40,
        forge_version="3.0.0-test",
        torch_version="test-torch",
    )

    assert provenance == {
        "vision": "mock",
        "language": "real",
        "labels": "real",
        "model_dir": str(tmp_path.resolve()),
        "git_sha": "b" * 40,
        "forge_version": "3.0.0-test",
        "torch_version": "test-torch",
    }


def test_tiny_model_name_is_not_implicitly_mock(tmp_path: Path) -> None:
    student = _RealStudent()
    student.vision_encoder = _TinyLlamaBackbone()

    provenance = build_provenance(
        student=student,
        dataset=_MarkedDataset(labels="real"),
        model_dir=tmp_path,
        git_sha="b" * 40,
        forge_version="3.0.0-test",
        torch_version="test-torch",
    )

    assert provenance["vision"] == "real"


def test_validate_provenance_rejects_incomplete_or_ambiguous_blocks() -> None:
    with pytest.raises(ProvenanceValidationError, match="missing: torch_version"):
        validate_provenance({k: v for k, v in _block().items() if k != "torch_version"})

    with pytest.raises(ProvenanceValidationError, match="must be 'real' or 'mock'"):
        validate_provenance(_block(labels="unknown"))


def test_mock_refusal_is_actionable_and_requires_explicit_opt_in() -> None:
    provenance = _block(language="mock", labels="mock")
    assert provenance_contains_mock(provenance)

    with pytest.raises(MockArtifactError) as error:
        require_real_provenance(provenance, action="export", allow_mock=False)

    message = str(error.value)
    assert "Refusing to export" in message
    assert "language, labels marked mock" in message
    assert "forge doctor" in message
    assert "--allow-mock" in message
    assert require_real_provenance(provenance, action="export", allow_mock=True) == provenance


def test_checkpoint_loader_opt_in_refuses_mock_and_keeps_legacy_compatible(tmp_path: Path) -> None:
    mock_path = tmp_path / "mock.pt"
    torch.save({"state_dict": {}, "provenance": _block(vision="mock")}, mock_path)

    assert load_checkpoint_payload(str(mock_path)) is not None
    with pytest.raises(MockArtifactError, match="Refusing to serve"):
        load_checkpoint_payload(str(mock_path), verify_provenance_for="serve")
    allowed = load_checkpoint_payload(
        str(mock_path),
        verify_provenance_for="serve",
        allow_mock=True,
    )
    assert allowed is not None

    legacy_path = tmp_path / "legacy.pt"
    torch.save({"state_dict": {}}, legacy_path)
    assert load_checkpoint_payload(str(legacy_path), verify_provenance_for="eval") is not None


def test_checkpoint_loader_refuses_present_malformed_provenance(tmp_path: Path) -> None:
    path = tmp_path / "malformed.pt"
    torch.save({"state_dict": {}, "provenance": {"vision": "real"}}, path)
    with pytest.raises(ProvenanceValidationError, match="incomplete"):
        load_checkpoint_payload(str(path), verify_provenance_for="export")

    non_mapping_path = tmp_path / "malformed-string.pt"
    torch.save({"state_dict": {}, "provenance": "mock"}, non_mapping_path)
    with pytest.raises(ProvenanceValidationError, match="must be a mapping"):
        load_checkpoint_payload(str(non_mapping_path), verify_provenance_for="eval")

    null_path = tmp_path / "malformed-null.pt"
    torch.save({"state_dict": {}, "provenance": None}, null_path)
    with pytest.raises(ProvenanceValidationError, match="must be a mapping"):
        load_checkpoint_payload(str(null_path), verify_provenance_for="serve")


def test_distribution_revision_is_used_outside_a_git_checkout(monkeypatch: pytest.MonkeyPatch) -> None:
    import forge._build_info as build_info

    revision = "d" * 40
    monkeypatch.setattr(build_info, "SOURCE_GIT_SHA", revision)
    for key in ("FORGE_GIT_SHA", "GITHUB_SHA", "GIT_COMMIT"):
        monkeypatch.delenv(key, raising=False)
    current_git_sha.cache_clear()

    assert current_git_sha() == revision
    current_git_sha.cache_clear()


def test_production_trainer_checkpoint_stamps_complete_real_provenance(tmp_path: Path) -> None:
    config = ForgeConfig.default()
    config.paths.output_dir = str(tmp_path / "outputs")
    config.paths.model_dir = str(tmp_path / "models")
    config.curriculum.enabled = False
    config.curriculum.plateau_window = 0
    config.curriculum.hard_example_mining = False

    trainer = ProductionTrainer(
        student=_RealStudent(),
        dataset=_MarkedDataset(labels="real"),
        loss_fn=nn.MSELoss(),
        config=config,
        checkpoint_dir=str(tmp_path),
    )
    checkpoint = trainer.save_checkpoint(tag="provenance")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    provenance = payload["provenance"]

    assert provenance["vision"] == "real"
    assert provenance["language"] == "real"
    assert provenance["labels"] == "real"
    assert provenance["model_dir"] == str((tmp_path / "models").resolve())
    assert re.fullmatch(r"[0-9a-f]{40}", provenance["git_sha"])
    validate_provenance(provenance)


def test_registry_extracts_serializes_and_flags_mock_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "mock.pt"
    provenance = _block(labels="mock")
    torch.save({"student_state_dict": {}, "provenance": provenance}, checkpoint)

    registry = ModelRegistry(tmp_path / "registry")
    entry = registry.register(checkpoint, variant="nano", name="fixture")

    assert entry.provenance == provenance
    assert entry.is_mock is True
    assert entry.mock_warning == f"[red]{MOCK_WARNING}[/red]"
    assert MOCK_WARNING in entry.summary()
    serialized = entry.to_dict()
    assert serialized["provenance"] == provenance
    assert serialized["is_mock"] is True
    assert serialized["provenance_warning"] == MOCK_WARNING

    restored = ModelRegistry(tmp_path / "registry").get(entry.model_id)
    assert restored is not None
    assert restored.provenance == provenance
    assert restored.is_mock is True


def test_registry_loads_legacy_entries_without_provenance(tmp_path: Path) -> None:
    registry_dir = tmp_path / "registry"
    registry_dir.mkdir()
    entry = ModelEntry(
        model_id="legacy-model",
        name="legacy",
        variant="nano",
        checkpoint_path="legacy.pt",
        created_at=time.time(),
    )
    legacy = entry.to_dict()
    legacy.pop("provenance")
    legacy.pop("is_mock")
    legacy.pop("provenance_warning")
    (registry_dir / "registry.json").write_text(
        json.dumps({"version": 1, "count": 1, "models": [legacy]}),
        encoding="utf-8",
    )

    restored = ModelRegistry(registry_dir).get("legacy")
    assert restored is not None
    assert restored.provenance is None
    assert restored.is_mock is False
    assert MOCK_WARNING not in restored.summary()
