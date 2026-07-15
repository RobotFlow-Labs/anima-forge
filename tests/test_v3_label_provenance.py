"""Focused PRD-36 fail-closed teacher-label provenance contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from forge.config import ForgeConfig
from forge.data.label_writer import LabelWriter
from forge.data.teacher_dataset import TeacherLabelDataset
from forge.distill import _create_mock_dataset, train_forge
from forge.errors import ForgeDataNotFoundError, ForgeModelNotFoundError
from forge.provenance import build_provenance
from forge.teacher import generate_teacher_labels


def _write_empty_metadata(path: Path, *, labels: str | None = None) -> None:
    path.mkdir(parents=True)
    metadata: dict[str, object] = {
        "total_episodes": 0,
        "episodes_per_file": 50,
    }
    if labels is not None:
        metadata["provenance"] = {"labels": labels}
    (path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def _strict_config(tmp_path: Path) -> ForgeConfig:
    config = ForgeConfig.default()
    config.student.allow_mock = False
    config.student.autosense = False
    config.paths.model_dir = str(tmp_path / "models")
    config.paths.data_dir = str(tmp_path / "data")
    config.paths.output_dir = str(tmp_path / "outputs")
    return config


def test_label_writer_defaults_to_mock_and_can_explicitly_stamp_real(tmp_path: Path) -> None:
    mock_dir = tmp_path / "mock"
    real_dir = tmp_path / "real"

    mock_metadata = LabelWriter(str(mock_dir)).finalize()
    real_metadata = LabelWriter(str(real_dir), labels_provenance="real").finalize()

    assert mock_metadata["provenance"] == {"labels": "mock"}
    assert real_metadata["provenance"] == {"labels": "real"}
    assert TeacherLabelDataset(mock_dir).labels_provenance == "mock"
    assert TeacherLabelDataset(real_dir).labels_provenance == "real"


def test_unmarked_label_metadata_is_fail_closed_mock(tmp_path: Path) -> None:
    labels_dir = tmp_path / "labels"
    _write_empty_metadata(labels_dir)

    dataset = TeacherLabelDataset(labels_dir)
    provenance = build_provenance(
        dataset=dataset,
        vision="real",
        language="real",
        model_dir=tmp_path / "models",
        git_sha="a" * 40,
        forge_version="test",
        torch_version="test",
    )

    assert dataset.labels_provenance == "mock"
    assert provenance["labels"] == "mock"


def test_zero_episode_real_generation_rejects_before_output(tmp_path: Path) -> None:
    config = _strict_config(tmp_path)

    with pytest.raises(ForgeDataNotFoundError) as exc_info:
        generate_teacher_labels(config, max_episodes=0, device="cpu")

    message = str(exc_info.value)
    assert "at least one episode" in message
    assert "--allow-mock" in message
    assert not (tmp_path / "data").exists()


def test_synthetic_teacher_generation_stamps_mock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _strict_config(tmp_path)
    config.student.allow_mock = True
    monkeypatch.setattr("forge.teacher.load_teacher", lambda *_args, **_kwargs: torch.nn.Linear(1, 1))
    monkeypatch.setattr("forge.teacher.load_processor", lambda *_args, **_kwargs: object())
    monkeypatch.setattr("forge.teacher._load_benchmark_tasks", lambda _benchmark: [])

    summary = generate_teacher_labels(config, max_episodes=0, device="cpu")
    metadata = json.loads((tmp_path / "data" / "teacher_labels" / "metadata.json").read_text(encoding="utf-8"))

    assert summary["provenance"] == {"labels": "mock"}
    assert metadata["provenance"] == {"labels": "mock"}


def test_create_mock_dataset_explicitly_stamps_mock(tmp_path: Path) -> None:
    dataset = _create_mock_dataset(tmp_path / "labels", n_episodes=0)

    assert dataset.labels_provenance == "mock"
    assert dataset.reader.metadata["provenance"] == {"labels": "mock"}


@pytest.mark.parametrize("labels", [None, "mock"])
def test_strict_distill_rejects_mock_or_unmarked_labels_without_artifacts(
    tmp_path: Path,
    labels: str | None,
) -> None:
    config = _strict_config(tmp_path)
    model_dir = Path(config.paths.model_dir)
    (model_dir / config.student.vision_encoder.replace("/", "--")).mkdir(parents=True)
    (model_dir / config.student.language_model.replace("/", "--")).mkdir(parents=True)
    _write_empty_metadata(Path(config.paths.data_dir) / "teacher_labels", labels=labels)

    with pytest.raises(ForgeDataNotFoundError) as exc_info:
        train_forge(config, device="cpu", max_steps=1)

    message = str(exc_info.value)
    assert "Regenerate" in message
    assert "--allow-mock" in message
    assert not Path(config.paths.output_dir).exists()


def test_missing_backbone_preflight_precedes_mock_label_rejection(tmp_path: Path) -> None:
    config = _strict_config(tmp_path)
    _write_empty_metadata(Path(config.paths.data_dir) / "teacher_labels")

    with pytest.raises(ForgeModelNotFoundError) as exc_info:
        train_forge(config, device="cpu", max_steps=1)

    message = str(exc_info.value)
    assert "forge models fetch google/siglip2-so400m-patch14-384" in message
    assert "forge doctor" in message
    assert not Path(config.paths.output_dir).exists()
