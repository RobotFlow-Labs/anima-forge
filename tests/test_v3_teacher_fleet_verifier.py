"""Tests for PRD-37's real teacher fleet verifier."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from forge.teacher_fleet import (
    VerificationFrame,
    _load_video_dataset_frames,
    build_fleet_report,
    build_isolated_fleet_report,
    combine_fleet_reports,
    verify_teacher,
    write_fleet_report,
)
from forge.teachers.base import ActionChunk, TeacherInfo


class _Adapter:
    def __init__(self, *, real_metadata: bool = True) -> None:
        self.real_metadata = real_metadata
        self.loaded = False
        self.unloaded = False

    def load(self, model_path: Path, device: str, dtype: torch.dtype) -> None:
        assert model_path.is_dir()
        assert device == "cpu"
        assert dtype == torch.float32
        self.loaded = True

    def info(self) -> TeacherInfo:
        return TeacherInfo(
            name="test-teacher",
            architecture="test",
            param_count=1.0,
            action_dim=3,
            action_horizon=2,
            vision_encoder="test",
            language_model="test",
            supports_chunking=True,
            supports_features=False,
        )

    def predict(self, image: np.ndarray, instruction: str, proprioception: np.ndarray) -> ActionChunk:
        assert image.dtype == np.uint8
        assert instruction
        value = float(proprioception[0]) if proprioception.size else 0.0
        actions = np.full((2, 3), value, dtype=np.float32)
        return ActionChunk(
            actions=actions,
            action_mean=actions,
            action_std=np.zeros_like(actions),
            confidence=np.ones_like(actions),
            metadata={"inference": "real"} if self.real_metadata else {},
        )

    def unload(self) -> None:
        self.unloaded = True


def _frames() -> list[VerificationFrame]:
    return [
        VerificationFrame(
            image=np.full((8, 9, 3), index, dtype=np.uint8),
            instruction="move the block",
            proprioception=np.array([index], dtype=np.float32),
            dataset="real-test",
            frame_index=index,
        )
        for index in range(3)
    ]


def test_verify_teacher_records_real_shapes_latency_and_actions(tmp_path: Path) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "model.safetensors").write_bytes(b"real")
    companion = tmp_path / "companion.bin"
    companion.write_bytes(b"weights")
    adapter = _Adapter()

    record = verify_teacher(
        "test-teacher",
        adapter,
        model_path,
        _frames(),
        device="cpu",
        dtype=torch.float32,
        companion_paths=(companion,),
    )

    assert record["status"] == "ok"
    assert record["predictions"] == 3
    assert record["actions"]["shape_per_prediction"] == [2, 3]
    assert record["actions"]["finite"] is True
    assert record["actions"]["min"] == 0.0
    assert record["actions"]["max"] == 2.0
    assert len(record["latency_ms"]["values"]) == 3
    assert record["cuda_memory_bytes"] == {"peak_allocated": 0, "peak_reserved": 0}
    assert record["frame_sources"][0]["image"] == [8, 9, 3]
    assert record["model_bytes"] == len(b"realweights")
    assert record["artifact_paths"] == [str(model_path.resolve()), str(companion.resolve())]
    assert adapter.loaded and adapter.unloaded


def test_verification_frames_use_episode_aligned_images_and_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "packed-dataset"
    dataset.mkdir()
    episodes = [
        SimpleNamespace(
            timesteps=1,
            instruction="first task",
            images=np.full((1, 4, 5, 3), 11, dtype=np.uint8),
            proprioception=np.array([[101.0]], dtype=np.float32),
        ),
        SimpleNamespace(
            timesteps=1,
            instruction="second task",
            images=np.full((1, 4, 5, 3), 22, dtype=np.uint8),
            proprioception=np.array([[202.0]], dtype=np.float32),
        ),
    ]
    calls = []

    def fake_load(path, **kwargs):
        calls.append((Path(path), kwargs))
        return episodes

    monkeypatch.setattr("forge.data.real_robot_episodes.load_real_robot_episodes", fake_load)

    frames = _load_video_dataset_frames(dataset, 2)

    assert calls == [(dataset, {"max_episodes": 2, "max_steps": 2})]
    assert [frame.instruction for frame in frames] == ["first task", "second task"]
    assert [int(frame.image[0, 0, 0]) for frame in frames] == [11, 22]
    assert [float(frame.proprioception[0]) for frame in frames] == [101.0, 202.0]


def test_verify_teacher_can_retain_real_predictions_for_distillation(tmp_path: Path) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()

    record = verify_teacher(
        "test-teacher",
        _Adapter(),
        model_path,
        _frames()[:2],
        device="cpu",
        dtype=torch.float32,
        include_predictions=True,
    )

    assert np.asarray(record["prediction_actions"]).shape == (2, 2, 3)
    assert np.asarray(record["prediction_confidences"]).shape == (2, 2, 3)
    assert record["prediction_actions"][1][0] == [1.0, 1.0, 1.0]


def test_verifier_rejects_adapter_without_real_inference_attestation(tmp_path: Path) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    adapter = _Adapter(real_metadata=False)

    with pytest.raises(ValueError, match="did not attest real inference"):
        verify_teacher(
            "test-teacher",
            adapter,
            model_path,
            _frames()[:1],
            device="cpu",
            dtype=torch.float32,
        )

    assert adapter.unloaded


@pytest.mark.parametrize("field", ["action_mean", "action_std", "confidence"])
def test_verifier_rejects_invalid_action_statistics(tmp_path: Path, field: str) -> None:
    class InvalidStatisticsAdapter(_Adapter):
        def predict(self, image, instruction, proprioception):
            chunk = super().predict(image, instruction, proprioception)
            setattr(chunk, field, np.full_like(chunk.actions, np.nan))
            return chunk

    model_path = tmp_path / "model"
    model_path.mkdir()
    adapter = InvalidStatisticsAdapter()

    with pytest.raises(ValueError, match="invalid"):
        verify_teacher(
            "test-teacher",
            adapter,
            model_path,
            _frames()[:1],
            device="cpu",
            dtype=torch.float32,
        )

    assert adapter.unloaded


def test_fleet_report_rejects_empty_teacher_selection(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="At least one teacher"):
        build_fleet_report(
            teacher_names=[],
            model_dir=tmp_path / "models",
            dataset_dir=tmp_path / "dataset",
            gpu_ids=[0],
        )


def test_report_writer_replaces_temporary_file_atomically(tmp_path: Path) -> None:
    output = tmp_path / "reports" / "fleet.json"
    report = {"all_real": True, "results": []}

    result = write_fleet_report(report, output)

    assert result == output
    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert not output.with_suffix(".json.tmp").exists()


def test_process_isolated_reports_are_combined_without_hiding_failures(tmp_path: Path) -> None:
    reports = [
        {"results": [{"teacher": "teacher-a", "status": "ok"}]},
        {"results": [{"teacher": "teacher-b", "status": "error", "error": "failed"}]},
    ]

    combined = combine_fleet_reports(
        reports,
        model_dir=tmp_path / "models",
        dataset_dir=tmp_path / "datasets",
        predictions=5,
    )

    assert combined["execution_isolation"] == "one-process-per-teacher"
    assert combined["teachers_requested"] == 2
    assert combined["teachers_verified"] == 1
    assert combined["all_real"] is False
    assert [record["teacher"] for record in combined["results"]] == ["teacher-a", "teacher-b"]


def test_isolated_fleet_builder_uses_packaged_worker_and_retains_predictions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(command, **_kwargs):
        output = Path(command[command.index("--output") + 1])
        teacher = command[command.index("--teacher") + 1]
        device = f"cuda:{command[command.index('--gpu') + 1]}"
        output.write_text(
            json.dumps(
                {
                    "results": [
                        {
                            "teacher": teacher,
                            "device": device,
                            "status": "ok",
                            "prediction_actions": [[[1.0]]],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        assert command[1:3] == ["-m", "forge.teacher_fleet"]
        assert "--include-predictions" in command
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("forge.teacher_fleet.subprocess.run", fake_run)
    report = build_isolated_fleet_report(
        teacher_names=["teacher-a", "teacher-b"],
        model_dir=tmp_path / "models",
        dataset_dir=tmp_path / "datasets",
        gpu_ids=[0, 1],
        predictions=1,
        include_predictions=True,
    )

    assert report["all_real"] is True
    assert report["execution_isolation"] == "one-process-per-teacher"
    assert report["results"][1]["device"] == "cuda:1"
    assert report["results"][0]["prediction_actions"] == [[[1.0]]]
