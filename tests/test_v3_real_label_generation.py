"""End-to-end contracts for genuine teacher-label HDF5 generation."""

from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from PIL import Image

from forge.config import ForgeConfig
from forge.data.label_writer import LabelReader
from forge.data.real_robot_episodes import load_real_robot_episodes
from forge.errors import ForgeModelNotFoundError
from forge.teacher import _infer_real_teacher_episode, generate_teacher_labels
from forge.teachers.base import ActionChunk, TeacherInfo


def _image_bytes(value: int) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.full((12, 14, 3), value, dtype=np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def _real_dataset(root: Path) -> Path:
    (root / "meta").mkdir(parents=True)
    data_dir = root / "data" / "chunk-000"
    data_dir.mkdir(parents=True)
    (root / "meta" / "info.json").write_text(
        json.dumps({"data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"}),
        encoding="utf-8",
    )
    (root / "meta" / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "place the block"}) + "\n",
        encoding="utf-8",
    )
    (root / "meta" / "episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "tasks": ["place the block"], "length": 2}) + "\n",
        encoding="utf-8",
    )
    rows = []
    for index, value in enumerate((10, 20)):
        rows.append(
            {
                "observation.images.image": {"bytes": _image_bytes(value), "path": f"frame_{index}.png"},
                "observation.state": [float(index + offset) for offset in range(8)],
                "action": [float(index) / 10 + offset / 100 for offset in range(7)],
                "episode_index": 0,
                "frame_index": index,
                "task_index": 0,
            }
        )
    pq.write_table(pa.Table.from_pylist(rows), data_dir / "episode_000000.parquet")
    return root


class _RealAdapter:
    def __init__(self, *, fail_after: int | None = None) -> None:
        self.fail_after = fail_after
        self.calls = 0
        self.loaded = False
        self.unloaded = False

    def load(self, path: Path, device: str, dtype: torch.dtype) -> None:
        assert path.is_dir()
        assert device == "cpu"
        assert dtype == torch.float32
        self.loaded = True

    def info(self) -> TeacherInfo:
        return TeacherInfo(
            name="fake-real",
            architecture="test-real",
            param_count=1.0,
            action_dim=7,
            action_horizon=1,
            vision_encoder="real",
            language_model="real",
            supports_chunking=False,
            supports_features=False,
        )

    def predict(self, image: np.ndarray, instruction: str, proprioception: np.ndarray) -> ActionChunk:
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("real inference failed")
        assert instruction == "place the block"
        values = np.arange(7, dtype=np.float32) + image[0, 0, 0] / 255.0 + proprioception[0] / 100.0
        actions = values[None, :]
        return ActionChunk(
            actions=actions,
            action_mean=actions.copy(),
            action_std=np.zeros_like(actions),
            confidence=np.ones_like(actions),
            metadata={"inference": "real"},
        )

    def unload(self) -> None:
        self.unloaded = True


def _config(tmp_path: Path, dataset: Path) -> ForgeConfig:
    config = ForgeConfig.default()
    config.student.allow_mock = False
    config.paths.model_dir = str(tmp_path / "models")
    config.paths.teacher = "fake--teacher"
    config.paths.data_dir = str(tmp_path / "generated")
    config.teacher.adapter = "fake-real"
    config.teacher.dataset = str(dataset)
    config.teacher.max_steps_per_episode = 2
    config.teacher.save_vision_features = False
    (Path(config.paths.model_dir) / config.paths.teacher).mkdir(parents=True)
    return config


def test_real_teacher_predictions_are_published_as_real_hdf5(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _real_dataset(tmp_path / "dataset")
    config = _config(tmp_path, dataset)
    adapter = _RealAdapter()
    monkeypatch.setattr(
        "forge.teachers.registry.get_registry",
        lambda: SimpleNamespace(create=lambda name: adapter if name == "fake-real" else None),
    )

    summary = generate_teacher_labels(config, max_episodes=1, device="cpu")

    labels_dir = Path(config.paths.data_dir) / "teacher_labels"
    reader = LabelReader(labels_dir)
    episode = reader[0]
    assert summary["provenance"] == {"labels": "real"}
    assert summary["teacher_adapter"] == "fake-real"
    assert summary["total_episodes"] == 1
    assert reader.metadata["provenance"] == {"labels": "real"}
    assert reader.metadata["source"]["teacher_adapter"] == "fake-real"
    assert reader.metadata["source"]["dataset_format"] == "lerobot-episode-parquet"
    assert episode["images"].shape == (2, 12, 14, 3)
    assert episode["ground_truth_actions"].shape == (2, 7)
    assert episode["teacher_action_mean"].shape == (2, 1, 7)
    assert episode["teacher_action_mean"][0, 0, 0] == pytest.approx(10 / 255)
    assert episode["teacher_action_mean"][1, 0, 0] == pytest.approx(20 / 255 + 0.01)
    assert episode["success"] is None
    assert summary["successful_episodes"] == 0
    assert summary["success_unknown_episodes"] == 1
    assert summary["success_rate"] is None
    assert reader.metadata["success_unknown_episodes"] == 1
    assert reader.metadata["success_rate"] is None
    assert adapter.calls == 2
    assert adapter.loaded and adapter.unloaded
    reader.close()


def test_failed_real_inference_preserves_previous_labels_and_removes_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _real_dataset(tmp_path / "dataset")
    config = _config(tmp_path, dataset)
    labels_dir = Path(config.paths.data_dir) / "teacher_labels"
    labels_dir.mkdir(parents=True)
    marker = labels_dir / "previous-labels"
    marker.write_text("keep", encoding="utf-8")
    adapter = _RealAdapter(fail_after=1)
    monkeypatch.setattr(
        "forge.teachers.registry.get_registry",
        lambda: SimpleNamespace(create=lambda _name: adapter),
    )

    with pytest.raises(RuntimeError, match="real inference failed"):
        generate_teacher_labels(config, max_episodes=1, device="cpu")

    assert marker.read_text(encoding="utf-8") == "keep"
    assert list(Path(config.paths.data_dir).glob(".teacher_labels-*")) == []
    assert adapter.unloaded


def test_missing_real_teacher_fails_before_label_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _real_dataset(tmp_path / "dataset")
    config = _config(tmp_path, dataset)
    teacher_path = Path(config.paths.model_dir) / config.paths.teacher
    teacher_path.rmdir()
    monkeypatch.setattr(
        "forge.teachers.registry.get_registry",
        lambda: pytest.fail("registry should not be used before checkpoint preflight"),
    )

    with pytest.raises(ForgeModelNotFoundError, match="forge models fetch fake/teacher"):
        generate_teacher_labels(config, max_episodes=1, device="cpu")

    assert not Path(config.paths.data_dir).exists()


def test_real_teacher_preserves_complete_action_chunks(tmp_path: Path) -> None:
    dataset = _real_dataset(tmp_path / "dataset")
    source = load_real_robot_episodes(dataset, max_episodes=1)[0]

    class ChunkAdapter(_RealAdapter):
        def predict(self, image, instruction, proprioception):
            first = super().predict(image, instruction, proprioception)
            actions = np.concatenate([first.actions + offset for offset in (0, 10, 20)], axis=0)
            return ActionChunk(
                actions=actions,
                action_mean=actions.copy(),
                action_std=np.zeros_like(actions),
                confidence=np.ones_like(actions),
                metadata={"inference": "real"},
            )

    episode = _infer_real_teacher_episode(ChunkAdapter(), source, save_vision_features=False)

    assert episode.teacher_action_mean.shape == (2, 3, 7)
    np.testing.assert_allclose(
        episode.teacher_action_mean[0, :, 0],
        [10 / 255, 10 + 10 / 255, 20 + 10 / 255],
    )


def test_partial_episode_vision_features_are_rejected(tmp_path: Path) -> None:
    dataset = _real_dataset(tmp_path / "dataset")
    source = load_real_robot_episodes(dataset, max_episodes=1)[0]

    class PartialFeaturesAdapter(_RealAdapter):
        def predict(self, image, instruction, proprioception):
            chunk = super().predict(image, instruction, proprioception)
            if self.calls == 1:
                chunk.vision_features = np.ones((2, 4), dtype=np.float16)
            return chunk

    with pytest.raises(ValueError, match="only part of the episode"):
        _infer_real_teacher_episode(PartialFeaturesAdapter(), source, save_vision_features=True)
