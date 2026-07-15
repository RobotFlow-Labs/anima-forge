"""B4 contracts for pipeline output paths, progress, summaries, and exits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
from typer.testing import CliRunner

from forge.config import ForgeConfig
from forge.pipeline import (
    _clear_runtime_export_artifacts,
    _create_quant_profile,
    _export_stage_result,
    _latest_checkpoint,
    _packed_compression_payload,
    _require_runtime_success,
    run_pipeline,
)


def test_pipeline_quant_profile_uses_configured_uniform_bits(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_profile(*_args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(compressed_size_mb=1.0)

    monkeypatch.setattr("forge.quantize.create_quant_profile", fake_profile)
    config = ForgeConfig.default()
    config.quant.bits = 8

    _create_quant_profile(nn.Linear(2, 2), config)

    assert captured["uniform_bits"] == 8
    assert "8bit" in str(captured["name"])


def test_selected_runtime_skip_is_a_pipeline_failure() -> None:
    result = _require_runtime_success(
        {"status": "skipped", "reason": "CUDAExecutionProvider is unavailable"},
        target="ONNX Runtime",
    )

    assert result["status"] == "failed"
    assert result["error"] == "CUDAExecutionProvider is unavailable"


def test_selected_runtime_success_is_preserved() -> None:
    result = {"status": "success", "fps": 42.0}

    assert _require_runtime_success(result, target="TensorRT") is result


def test_requested_export_skip_fails_stage() -> None:
    result = _export_stage_result(
        {"export_tensorrt": {"status": "skipped", "reason": "not a CUDA device"}},
        {"tensorrt"},
    )

    assert result["status"] == "failed"
    assert "tensorrt" in result["error"]


def test_latest_checkpoint_finds_production_and_train_run_layouts(tmp_path: Path) -> None:
    production = tmp_path / "checkpoints" / "production" / "final.pt"
    production.parent.mkdir(parents=True)
    production.write_bytes(b"production")
    assert _latest_checkpoint(tmp_path) == production

    train_run = tmp_path / "train-runs" / "run-1" / "checkpoints" / "production" / "final.pt"
    train_run.parent.mkdir(parents=True)
    train_run.write_bytes(b"newer")
    train_run.touch()
    assert _latest_checkpoint(tmp_path) == train_run


def test_pipeline_export_clears_stale_and_partial_runtime_artifacts(tmp_path: Path) -> None:
    onnx_path = tmp_path / "forge.onnx"
    external_data_path = tmp_path / "forge.onnx.data"
    engine_path = tmp_path / "forge.engine"
    unrelated_path = tmp_path / "keep.json"
    for path in (onnx_path, external_data_path, engine_path, unrelated_path):
        path.write_text("stale", encoding="utf-8")

    _clear_runtime_export_artifacts(onnx_path, engine_path)

    assert not onnx_path.exists()
    assert not external_data_path.exists()
    assert not engine_path.exists()
    assert unrelated_path.is_file()


def test_pipeline_core_records_stage_progress_and_summary(tmp_path: Path, monkeypatch) -> None:
    import forge.distill

    checkpoint_bytes = b"durable-final-checkpoint"

    def fake_train(*_args, **kwargs):
        checkpoint = Path(kwargs["checkpoint_dir"]) / "checkpoints" / "final.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(checkpoint_bytes)
        return {"status": "success", "final_loss": 0.25}

    monkeypatch.setattr(forge.distill, "train_forge", fake_train)
    config = ForgeConfig.default()
    config.paths.output_dir = str(tmp_path)
    events: list[dict[str, object]] = []
    results = run_pipeline(
        config,
        device="cpu",
        stage="distill",
        max_distill_steps=5,
        progress_callback=events.append,
    )
    assert [event["status"] for event in events] == ["started", "completed"]
    assert all(event["stage"] == "distill" for event in events)
    assert results["status"] == "completed"
    assert results["execution"]["schema"] == "forge.pipeline-execution.v1"
    assert results["execution"]["git_sha"]
    assert results["execution"]["requested_stage"] == "distill"
    assert results["execution"]["device"] == "cpu"
    assert results["stage_timings_seconds"]["distill"] >= 0
    assert results["distill"]["checkpoint_path"] == str((tmp_path / "checkpoints" / "final.pt").resolve())
    assert results["distill"]["checkpoint_sha256"] == hashlib.sha256(checkpoint_bytes).hexdigest()
    summary = Path(results["pipeline_summary_path"])
    assert summary == (tmp_path / "pipeline_summary.json").resolve()
    assert json.loads(summary.read_text(encoding="utf-8"))["status"] == "completed"
    persisted = json.loads(summary.read_text(encoding="utf-8"))
    assert persisted["distill"]["checkpoint_sha256"] == hashlib.sha256(checkpoint_bytes).hexdigest()
    assert "config_path" not in persisted
    assert "config_sha256" not in persisted


def test_pipeline_summary_binds_loaded_config_file_bytes(tmp_path: Path, monkeypatch) -> None:
    import forge.distill

    output_dir = tmp_path / "output"
    config_path = tmp_path / "forge.yaml"
    config_bytes = f"paths:\n  output_dir: {output_dir}\nstudent:\n  variant: nano\n".encode()
    config_path.write_bytes(config_bytes)

    def fake_train(*_args, **kwargs):
        checkpoint = Path(kwargs["checkpoint_dir"]) / "checkpoints" / "final.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"checkpoint")
        return {"status": "success", "final_loss": 0.25}

    monkeypatch.setattr(forge.distill, "train_forge", fake_train)
    results = run_pipeline(
        ForgeConfig.from_yaml(config_path),
        device="cpu",
        stage="distill",
        max_distill_steps=1,
    )

    assert results["config_path"] == str(config_path.resolve())
    assert results["config_sha256"] == hashlib.sha256(config_bytes).hexdigest()
    persisted = json.loads((output_dir / "pipeline_summary.json").read_text(encoding="utf-8"))
    assert persisted["config_path"] == str(config_path.resolve())
    assert persisted["config_sha256"] == hashlib.sha256(config_bytes).hexdigest()


def test_primary_packed_compression_payload_binds_exact_pruned_source(tmp_path: Path) -> None:
    pruned = tmp_path / "pruned.pt"
    pruned.write_bytes(b"exact-pruned-checkpoint")
    digest = hashlib.sha256(pruned.read_bytes()).hexdigest()

    payload = _packed_compression_payload(
        packed_state={"weight": {"kind": "tensor", "value": torch.ones(1)}},
        quantization={"schema": "forge.packed-state.v1", "method": "qvla", "bits": 4},
        pruning={"removed_layers": [2, 4]},
        provenance={"vision": "real", "language": "real", "labels": "real"},
        source_checkpoint_sha256=digest,
        config_sha256="a" * 64,
    )
    artifact = tmp_path / "qvla_4bit.pt"
    torch.save(payload, artifact)

    restored = torch.load(artifact, map_location="cpu", weights_only=True)
    assert restored["source_checkpoint_sha256"] == digest
    assert restored["config_sha256"] == "a" * 64


@pytest.mark.parametrize("precision", ["fp16", "int8"])
def test_pipeline_export_always_persists_exact_runtime_inputs(
    precision: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import forge.export.onnx_export
    import forge.export.tensorrt_export
    import forge.student

    class FakeStudent:
        def to(self, _device: str):
            return self

    images = torch.ones((2, 3, 384, 384), dtype=torch.float32)
    language_ids = torch.arange(16, dtype=torch.int64).reshape(2, 8)
    export_call: dict[str, object] = {}

    monkeypatch.setattr(forge.student, "FORGEStudent", lambda *_args, **_kwargs: FakeStudent())
    monkeypatch.setattr(
        "forge.pipeline._load_export_runtime_inputs",
        lambda *_args, **_kwargs: (images, language_ids, "real"),
    )

    def fake_export_onnx(_student, path: Path, **_kwargs) -> Path:
        import onnx

        graph = onnx.helper.make_graph([], "pipeline-fixture", [], [])
        onnx.save_model(onnx.helper.make_model(graph), path)
        return path

    def fake_export_tensorrt(_onnx_path, path: Path, **kwargs) -> Path:
        export_call.update(kwargs)
        path.write_bytes(b"engine")
        return path

    monkeypatch.setattr(forge.export.onnx_export, "export_onnx", fake_export_onnx)
    monkeypatch.setattr(
        forge.export.onnx_export,
        "benchmark_onnx_runtime",
        lambda *_args, **_kwargs: {"status": "success", "provider": "CUDAExecutionProvider"},
    )
    monkeypatch.setattr(forge.export.tensorrt_export, "check_tensorrt_available", lambda: True)
    monkeypatch.setattr(forge.export.tensorrt_export, "export_tensorrt", fake_export_tensorrt)
    monkeypatch.setattr(
        forge.export.tensorrt_export,
        "benchmark_tensorrt_runtime",
        lambda *_args, **_kwargs: {"status": "success", "provider": "TensorRT"},
    )

    config = ForgeConfig.default()
    config.student.allow_mock = True
    config.paths.output_dir = str(tmp_path / precision)
    config.export.formats = ["onnx", "tensorrt"]
    config.export.tensorrt_precision = precision
    results = run_pipeline(config, device="cuda", stage="export")

    runtime_path = Path(results["export_runtime_inputs"]["path"])
    assert results["status"] == "completed"
    assert runtime_path == (tmp_path / precision / "tensorrt_calibration.npz").resolve()
    assert results["export_runtime_inputs"]["sha256"] == hashlib.sha256(runtime_path.read_bytes()).hexdigest()
    with np.load(runtime_path, allow_pickle=False) as runtime_inputs:
        assert np.array_equal(runtime_inputs["images"], images.numpy())
        assert np.array_equal(runtime_inputs["language_ids"], language_ids.numpy())
    assert export_call["calibration_data"] == (str(runtime_path) if precision == "int8" else None)


def test_pipeline_core_marks_stage_failure(tmp_path: Path, monkeypatch) -> None:
    import forge.distill

    def fail(*_args, **_kwargs):
        raise RuntimeError("distillation exploded")

    monkeypatch.setattr(forge.distill, "train_forge", fail)
    config = ForgeConfig.default()
    config.paths.output_dir = str(tmp_path)
    events: list[dict[str, object]] = []
    results = run_pipeline(
        config,
        device="cpu",
        stage="distill",
        max_distill_steps=5,
        progress_callback=events.append,
    )
    assert results["status"] == "failed"
    assert results["distill"] == {"status": "failed", "error": "distillation exploded"}
    assert events[-1]["status"] == "failed"


def test_pipeline_cli_overrides_output_and_prints_live_progress(tmp_path: Path, monkeypatch) -> None:
    from forge import pipeline as pipeline_module
    from forge.cli_v2 import app

    captured: dict[str, object] = {}

    def fake_run(config, **kwargs):
        captured["output_dir"] = config.paths.output_dir
        captured["data_dir"] = config.paths.data_dir
        captured["batch_size"] = config.distill.batch_size
        captured["gradient_accumulation_steps"] = config.distill.gradient_accumulation_steps
        captured["max_label_episodes"] = kwargs["max_label_episodes"]
        captured["checkpoint_path"] = kwargs["checkpoint_path"]
        callback = kwargs["progress_callback"]
        callback({"stage": "distill", "title": "Knowledge Distillation", "status": "started"})
        callback(
            {
                "stage": "distill",
                "status": "progress",
                "step": 3,
                "total_steps": 5,
                "loss": 0.5,
                "steps_per_second": 2.0,
                "eta_seconds": 1.0,
                "vram_gib": 14.5,
            }
        )
        callback({"stage": "distill", "status": "completed", "elapsed_seconds": 1.25})
        summary = Path(config.paths.output_dir) / "pipeline_summary.json"
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text("{}\n", encoding="utf-8")
        return {
            "distill": {"status": "success"},
            "total_time_seconds": 1.25,
            "status": "completed",
            "pipeline_summary_path": str(summary),
        }

    monkeypatch.setattr(pipeline_module, "run_pipeline", fake_run)
    checkpoint = tmp_path / "source.pt"
    checkpoint.write_bytes(b"checkpoint")
    result = CliRunner().invoke(
        app,
        [
            "pipeline",
            "--device",
            "cpu",
            "--stage",
            "distill",
            "--skip-labels",
            "--max-steps",
            "5",
            "--max-episodes",
            "3",
            "--output-dir",
            str(tmp_path),
            "--data-dir",
            str(tmp_path / "labels"),
            "--checkpoint",
            str(checkpoint),
            "--batch-size",
            "12",
            "--gradient-accumulation-steps",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["output_dir"] == str(tmp_path.resolve())
    assert captured["data_dir"] == str((tmp_path / "labels").resolve())
    assert captured["batch_size"] == 12
    assert captured["gradient_accumulation_steps"] == 2
    assert captured["max_label_episodes"] == 3
    assert captured["checkpoint_path"] == checkpoint.resolve()
    assert "Knowledge Distillation" in result.output
    assert "DONE" in result.output
    assert "step 3/5" in result.output
    assert "VRAM 14.50 GiB" in result.output
    assert "Pipeline summary:" in result.output


def test_pipeline_cli_rejects_missing_checkpoint(tmp_path: Path) -> None:
    from forge.cli_v2 import app

    missing = tmp_path / "missing.pt"
    result = CliRunner().invoke(
        app,
        ["pipeline", "--device", "cpu", "--stage", "compress", "--checkpoint", str(missing)],
    )
    assert result.exit_code == 2
    assert "does not exist" in result.output


def test_pipeline_cli_exits_two_for_failed_summary(tmp_path: Path, monkeypatch) -> None:
    from forge import pipeline as pipeline_module
    from forge.cli_v2 import app

    def fake_run(config, **kwargs):
        summary = Path(config.paths.output_dir) / "pipeline_summary.json"
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text("{}\n", encoding="utf-8")
        return {
            "distill": {"status": "failed", "error": "boom"},
            "total_time_seconds": 0.1,
            "status": "failed",
            "pipeline_summary_path": str(summary),
        }

    monkeypatch.setattr(pipeline_module, "run_pipeline", fake_run)
    result = CliRunner().invoke(
        app,
        ["pipeline", "--device", "cpu", "--stage", "distill", "--output-dir", str(tmp_path)],
    )
    assert result.exit_code == 2
    assert "FAILED" in result.output
