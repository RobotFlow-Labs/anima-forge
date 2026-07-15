"""Strict atomic persistence contracts for pipeline evidence summaries."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from forge.config import ForgeConfig
from forge.pipeline import _artifact_family_sha256, _checkpoint_sha256, _finalize


def _config(tmp_path: Path) -> ForgeConfig:
    config = ForgeConfig.default()
    config.paths.output_dir = str(tmp_path)
    return config


def _completed_results() -> dict[str, object]:
    return {
        "device": "cuda",
        "distill": {"status": "success", "total_steps": 10, "final_loss": 0.25},
        "provenance": {"vision": "real", "language": "real", "labels": "real"},
    }


def test_pipeline_artifact_hashes_cover_primary_and_sidecars(tmp_path: Path) -> None:
    import onnx

    primary = tmp_path / "forge.onnx"
    sidecar = tmp_path / "weights.bin"
    unrelated = tmp_path / "other.bin"
    tensor = onnx.numpy_helper.from_array(np.ones((1,), dtype=np.float32), name="weight")
    graph = onnx.helper.make_graph([], "external-fixture", [], [], [tensor])
    model = onnx.helper.make_model(graph)
    onnx.external_data_helper.convert_model_to_external_data(
        model,
        all_tensors_to_one_file=True,
        location=sidecar.name,
        size_threshold=0,
    )
    onnx.save_model(model, primary)
    unrelated.write_bytes(b"ignore")

    hashes = _artifact_family_sha256(primary)

    assert hashes == {
        primary.name: _checkpoint_sha256(primary),
        sidecar.name: _checkpoint_sha256(sidecar),
    }


def test_pipeline_artifact_family_requires_primary_file(tmp_path: Path) -> None:
    (tmp_path / "forge.onnx.data").write_bytes(b"weights")

    with pytest.raises(FileNotFoundError, match="Required ONNX graph is missing"):
        _artifact_family_sha256(tmp_path / "forge.onnx")


def test_pipeline_summary_persists_exact_completed_schema_and_fsyncs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from forge import json_artifacts

    fsync_calls: list[int] = []
    real_fsync = json_artifacts.os.fsync

    def recording_fsync(descriptor: int) -> None:
        fsync_calls.append(descriptor)
        real_fsync(descriptor)

    monkeypatch.setattr(json_artifacts.os, "fsync", recording_fsync)
    results = _completed_results()

    returned = _finalize(results, time.time() - 1.0, tmp_path, _config(tmp_path))

    summary = tmp_path / "pipeline_summary.json"
    assert json.loads(summary.read_text(encoding="utf-8")) == returned
    assert returned["status"] == "completed"
    assert returned["pipeline_summary_path"] == str(summary.resolve())
    assert fsync_calls
    assert list(tmp_path.iterdir()) == [summary]


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_non_finite_pipeline_summary_preserves_existing_completed_evidence(tmp_path: Path, value: float) -> None:
    summary = tmp_path / "pipeline_summary.json"
    original = '{"status": "completed", "run": "previous"}\n'
    summary.write_text(original, encoding="utf-8")
    results = _completed_results()
    results["metric"] = value

    with pytest.raises(ValueError, match="Out of range float values"):
        _finalize(results, time.time() - 1.0, tmp_path, _config(tmp_path))

    assert summary.read_text(encoding="utf-8") == original
    assert list(tmp_path.iterdir()) == [summary]


def test_unserializable_pipeline_summary_publishes_no_partial_target(tmp_path: Path) -> None:
    results = _completed_results()
    results["unsupported"] = object()

    with pytest.raises(TypeError, match="not JSON serializable"):
        _finalize(results, time.time() - 1.0, tmp_path, _config(tmp_path))

    assert not (tmp_path / "pipeline_summary.json").exists()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("preexisting", [False, True])
def test_pipeline_summary_replace_failure_preserves_target_and_cleans_temp(
    preexisting: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from forge import json_artifacts

    summary = tmp_path / "pipeline_summary.json"
    original = '{"status": "completed", "run": "previous"}\n'
    if preexisting:
        summary.write_text(original, encoding="utf-8")

    def fail_replace(_source, _target) -> None:
        raise OSError("injected pipeline summary replace failure")

    monkeypatch.setattr(json_artifacts.os, "replace", fail_replace)

    with pytest.raises(OSError, match="injected pipeline summary replace failure"):
        _finalize(_completed_results(), time.time() - 1.0, tmp_path, _config(tmp_path))

    if preexisting:
        assert summary.read_text(encoding="utf-8") == original
        assert list(tmp_path.iterdir()) == [summary]
    else:
        assert not summary.exists()
        assert list(tmp_path.iterdir()) == []
