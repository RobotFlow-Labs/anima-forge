"""Artifact-driven validation matrix contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from forge.benchmark.matrix import (
    MANIFEST_SCHEMA,
    _prepare_variant,
    _publish_results_directory,
    _quantized_architecture_evidence,
    _run_pytorch_benchmark,
    _selected_compression_metrics,
    _selected_export_metrics,
    _selected_training_metrics,
    _validated_backend_result,
    load_validation_manifest,
    run_validation_matrix,
)
from forge.cli_commands.shared import load_forge_config


def _write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _real_provenance() -> dict[str, str]:
    return {"vision": "real", "language": "real", "labels": "real"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_inline_onnx(path: Path) -> None:
    import onnx

    graph = onnx.helper.make_graph([], "matrix-fixture", [], [])
    onnx.save_model(onnx.helper.make_model(graph), path)


def _write_external_onnx(path: Path, *, location: str = "weights.bin") -> Path:
    import onnx

    tensor = onnx.numpy_helper.from_array(np.ones((1,), dtype=np.float32), name="weight")
    graph = onnx.helper.make_graph([], "external-fixture", [], [], [tensor])
    model = onnx.helper.make_model(graph)
    onnx.external_data_helper.convert_model_to_external_data(
        model,
        all_tensors_to_one_file=True,
        location=location,
        size_threshold=0,
    )
    onnx.save_model(model, path)
    return path.parent / location


def _build_matrix_fixture(tmp_path: Path, *, variant: str = "nano", strict: bool = True) -> dict[str, Path]:
    root = tmp_path / "artifacts"
    training_dir = root / variant
    compression_dir = root / f"{variant}-d2-qvla4"
    export_dir = root / f"{variant}-d2-export"
    alternate_dir = root / f"{variant}-d2-quant"
    for directory in (training_dir / "checkpoints", compression_dir / "compressed", export_dir, alternate_dir):
        directory.mkdir(parents=True, exist_ok=True)
    config = tmp_path / "forge.yaml"
    config.write_text(f"student:\n  variant: {variant}\n", encoding="utf-8")
    config_sha256 = _sha256(config)
    checkpoint = training_dir / "checkpoints" / "final.pt"
    configured = load_forge_config(config, required=True)
    torch.save(
        {
            "step": 2000,
            "student_config": asdict(configured.student),
            "provenance": _real_provenance(),
        },
        checkpoint,
    )
    pruned = compression_dir / "compressed" / "pruned.pt"
    pruned.write_bytes(b"pruned")
    pruning = {"removed_layers": [2, 4], "target_layers": 8, "calibration_provenance": "real"}
    quantized = {
        "qvla_int4": compression_dir / "compressed" / "qvla_4bit.pt",
        "qvla_int8": alternate_dir / "qvla_8bit.pt",
        "turboquant_mse_int4": alternate_dir / "turboquant_mse_4bit.pt",
        "turboquant_mse_int8": alternate_dir / "turboquant_mse_8bit.pt",
    }
    for name, path in quantized.items():
        if strict:
            method = "qvla" if name.startswith("qvla_") else "turboquant-mse"
            bits = 4 if name.endswith("int4") else 8
            payload = {
                "config_sha256": config_sha256,
                "source_checkpoint_sha256": _sha256(pruned),
                "pruning": pruning,
                "quantization": {
                    "schema": "forge.packed-state.v1",
                    "method": method,
                    "bits": bits,
                },
            }
            torch.save(payload, path)
        else:
            path.write_bytes(name.encode())
    training = _write_json(
        training_dir / "pipeline_summary.json",
        {
            "status": "completed",
            "pipeline_summary_path": str(training_dir / "pipeline_summary.json"),
            "distill": {
                "device": "cuda",
                "total_steps": 2000,
                "elapsed_seconds": 2.0,
                "steps_per_second": 1.0,
                "initial_loss": 2.0,
                "final_loss": 1.0,
                "best_loss": 0.9,
                "loss_reduction_percent": 50.0,
                "provenance": _real_provenance(),
                "cuda_memory": {"target_60_80_percent_met": True},
                "checkpoint_dir": str(training_dir / "checkpoints"),
            },
        },
    )
    compression = _write_json(
        compression_dir / "pipeline_summary.json",
        {
            "status": "completed",
            "pipeline_summary_path": str(compression_dir / "pipeline_summary.json"),
            "config_sha256": config_sha256,
            "source_checkpoint": str(checkpoint),
            "source_checkpoint_sha256": _sha256(checkpoint) if strict else None,
            "provenance": _real_provenance(),
            "pruning": {
                "status": "success",
                "n_removed": 2,
                "removed_layers": [2, 4],
                "path": str(pruned),
                "sha256": _sha256(pruned) if strict else None,
            },
            "compression": {
                "status": "success",
                "path": str(quantized["qvla_int4"]),
                "sha256": _sha256(quantized["qvla_int4"]) if strict else None,
            },
            "quantization": {"status": "success", "serialization_schema": "forge.packed-state.v1"},
        },
    )
    runtime_inputs = export_dir / "tensorrt_calibration.npz"
    np.savez(runtime_inputs, images=np.ones((1, 3, 4, 4), dtype=np.float32), language_ids=np.ones((1, 2)))
    onnx = export_dir / "forge.onnx"
    _write_inline_onnx(onnx)
    tensorrt = export_dir / "forge.engine"
    tensorrt.write_bytes(b"engine")
    export = _write_json(
        export_dir / "pipeline_summary.json",
        {
            "status": "completed",
            "pipeline_summary_path": str(export_dir / "pipeline_summary.json"),
            "config_sha256": config_sha256,
            "source_checkpoint": str(quantized["qvla_int4"]),
            "source_checkpoint_sha256": _sha256(quantized["qvla_int4"]) if strict else None,
            "provenance": _real_provenance(),
            "export_runtime_inputs": {
                "status": "success",
                "labels_provenance": "real",
                "path": str(runtime_inputs) if strict else None,
                "sha256": _sha256(runtime_inputs) if strict else None,
            },
            "export_onnx": {
                "status": "success",
                "path": str(onnx),
                "artifacts_sha256": {onnx.name: _sha256(onnx)} if strict else None,
            },
            "export_tensorrt": {
                "status": "success",
                "precision": "int8",
                "path": str(tensorrt),
                "sha256": _sha256(tensorrt) if strict else None,
            },
        },
    )
    entry: dict[str, object] = {
        "variant": variant,
        "expected_training_steps": 2000,
        "config": str(config),
        "config_sha256": config_sha256,
        "checkpoint": str(checkpoint),
        "training_summary": str(training),
        "training_config_binding": "checkpoint-contract-v1",
        "compression_summary": str(compression),
        "export_summary": str(export),
        "runtime_inputs": str(runtime_inputs),
        "onnx": str(onnx),
        "tensorrt": str(tensorrt),
        "quantized": {name: str(path) for name, path in quantized.items()},
    }
    if strict:
        entry.update(
            evidence_profile="sha256-v1",
            training_summary_sha256=_sha256(training),
            training_checkpoint_sha256=_sha256(checkpoint),
            quantized_sha256={name: _sha256(path) for name, path in quantized.items()},
        )
    manifest = _write_json(tmp_path / "manifest.json", {"schema": MANIFEST_SCHEMA, "variants": [entry]})
    return {
        "manifest": manifest,
        "config": config,
        "checkpoint": checkpoint,
        "training": training,
        "compression": compression,
        "export": export,
        "pruned": pruned,
        "runtime_inputs": runtime_inputs,
        "onnx": onnx,
        "tensorrt": tensorrt,
        **{f"quantized_{name}": path for name, path in quantized.items()},
    }


def _add_external_onnx_family(fixture: dict[str, Path]) -> Path:
    onnx_path = fixture["onnx"]
    sidecar = _write_external_onnx(onnx_path)
    export = json.loads(fixture["export"].read_text(encoding="utf-8"))
    export["export_onnx"]["artifacts_sha256"] = {
        onnx_path.name: _sha256(onnx_path),
        sidecar.name: _sha256(sidecar),
    }
    fixture["export"].write_text(json.dumps(export), encoding="utf-8")
    return sidecar


def _patch_successful_matrix_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "forge.benchmark.matrix._run_pytorch_benchmark",
        lambda **_kwargs: {
            "status": "success",
            "device": "cuda",
            "execution": {"requested_device": "cuda", "resolved_device": "cuda"},
            "actions_finite": True,
            "actions_shape": [1, 7],
            "action_samples": 1,
            "input_provenance": {"kind": "real"},
            "compression": {"student_params_m": 10.0},
        },
    )
    monkeypatch.setattr(
        "forge.benchmark.matrix.benchmark_onnx_runtime",
        lambda *_args, **_kwargs: {
            "status": "success",
            "provider": "CUDAExecutionProvider",
            "device": "cuda",
            "provider_device_id": 0,
            "actions_finite": True,
            "actions_shape": [1, 7],
            "action_samples": 1,
        },
    )
    monkeypatch.setattr(
        "forge.benchmark.matrix.benchmark_tensorrt_runtime",
        lambda *_args, **kwargs: {
            "status": "success",
            "provider": "TensorRT",
            "precision": kwargs["precision"],
            "actions_finite": True,
        },
    )


def test_manifest_rejects_obsolete_or_empty_schemas(tmp_path: Path) -> None:
    manifest = _write_json(tmp_path / "manifest.json", {"schema": "old", "variants": []})
    with pytest.raises(ValueError, match="Manifest schema"):
        load_validation_manifest(manifest)


def test_manifest_accepts_canonical_variant_matching_config(tmp_path: Path) -> None:
    config = tmp_path / "forge_micro.yaml"
    config.write_text("student:\n  variant: micro\n", encoding="utf-8")
    manifest = _write_json(
        tmp_path / "manifest.json",
        {
            "schema": MANIFEST_SCHEMA,
            "variants": [
                {
                    "variant": "micro",
                    "config": config.name,
                    "expected_training_steps": 2000,
                    "evidence_profile": "sha256-v1",
                    "config_sha256": _sha256(config),
                }
            ],
        },
    )

    resolved, variants = load_validation_manifest(manifest)

    assert resolved == manifest.resolve()
    assert variants[0]["variant"] == "micro"


@pytest.mark.parametrize(
    ("manifest_variant", "config_variant", "match"),
    [
        ("nano", "micro", "does not match config student.variant"),
        ("nano_flagship", "nano", "Validation variant must be one of"),
    ],
)
def test_manifest_rejects_mismatched_or_free_form_variant_before_output_creation(
    manifest_variant: str,
    config_variant: str,
    match: str,
    tmp_path: Path,
) -> None:
    config = tmp_path / "forge.yaml"
    config.write_text(f"student:\n  variant: {config_variant}\n", encoding="utf-8")
    manifest = _write_json(
        tmp_path / "manifest.json",
        {
            "schema": MANIFEST_SCHEMA,
            "variants": [
                {
                    "variant": manifest_variant,
                    "config": config.name,
                    "expected_training_steps": 2000,
                }
            ],
        },
    )
    results_dir = tmp_path / "must-not-exist"

    with pytest.raises(ValueError, match=match):
        run_validation_matrix(manifest, results_dir=results_dir)

    assert not results_dir.exists()


def test_manifest_rejects_duplicate_variant_before_output_creation(tmp_path: Path) -> None:
    config = tmp_path / "forge_nano.yaml"
    config.write_text("student:\n  variant: nano\n", encoding="utf-8")
    manifest = _write_json(
        tmp_path / "manifest.json",
        {
            "schema": MANIFEST_SCHEMA,
            "variants": [
                {
                    "variant": "nano",
                    "config": config.name,
                    "expected_training_steps": 2000,
                    "evidence_profile": "sha256-v1",
                    "config_sha256": _sha256(config),
                },
                {
                    "variant": "nano",
                    "config": config.name,
                    "expected_training_steps": 2000,
                    "evidence_profile": "sha256-v1",
                    "config_sha256": _sha256(config),
                },
            ],
        },
    )
    results_dir = tmp_path / "must-not-exist"

    with pytest.raises(ValueError, match="duplicate variant 'nano'"):
        run_validation_matrix(manifest, results_dir=results_dir)

    assert not results_dir.exists()


def test_pytorch_matrix_command_forwards_real_input_contract(tmp_path: Path, monkeypatch) -> None:
    observed: list[str] = []

    def fake_run(command, **_kwargs):
        observed.extend(command)
        stdout = json.dumps(
            {
                "source_checkpoint": str(tmp_path / "final.pt"),
                "provenance": {"model_dir": str(tmp_path / "models")},
            }
        )
        return type("Completed", (), {"returncode": 0, "stdout": stdout, "stderr": ""})()

    monkeypatch.setattr("forge.benchmark.matrix.subprocess.run", fake_run)
    output = tmp_path / "report.json"
    report = _run_pytorch_benchmark(
        config=tmp_path / "forge.yaml",
        checkpoint=tmp_path / "final.pt",
        output=output,
        device="cuda",
        samples=3,
        duration=0.25,
        data_dir=tmp_path / "real-data",
        instruction="push the block to the target",
    )

    assert report["status"] == "success"
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "source_checkpoint": "final.pt",
        "provenance": {"model_dir": "models"},
        "status": "success",
    }
    assert observed[observed.index("--data-dir") + 1] == str(tmp_path / "real-data")
    assert observed[observed.index("--instruction") + 1] == "push the block to the target"


def test_matrix_pytorch_rewrite_rejects_non_finite_report_without_replacing_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "nano_pytorch.json"
    original = '{"status": "previous"}\n'
    output.write_text(original, encoding="utf-8")
    completed = type(
        "Completed",
        (),
        {"returncode": 0, "stdout": json.dumps({"latency_ms": float("nan")}), "stderr": ""},
    )()
    monkeypatch.setattr("forge.benchmark.matrix.subprocess.run", lambda *_args, **_kwargs: completed)

    result = _run_pytorch_benchmark(
        config=tmp_path / "forge.yaml",
        checkpoint=tmp_path / "final.pt",
        output=output,
        device="cuda",
        samples=1,
        duration=0.1,
        data_dir=None,
        instruction=None,
    )

    assert result["status"] == "failed"
    assert "non-finite JSON constant NaN" in result["error"]
    assert output.read_text(encoding="utf-8") == original
    assert list(tmp_path.iterdir()) == [output]


def test_matrix_pytorch_does_not_promote_declared_failure(tmp_path: Path, monkeypatch) -> None:
    completed = type(
        "Completed",
        (),
        {"returncode": 0, "stdout": '{"status":"failed","error":"runtime rejected actions"}', "stderr": ""},
    )()
    monkeypatch.setattr("forge.benchmark.matrix.subprocess.run", lambda *_args, **_kwargs: completed)
    output = tmp_path / "report.json"

    result = _run_pytorch_benchmark(
        config=tmp_path / "forge.yaml",
        checkpoint=tmp_path / "final.pt",
        output=output,
        device="cuda",
        samples=1,
        duration=0.1,
        data_dir=None,
        instruction=None,
    )

    assert result == {"status": "failed", "error": "runtime rejected actions"}
    assert not output.exists()


def test_onnx_matrix_evidence_must_match_selected_cuda_index() -> None:
    result = _validated_backend_result(
        {
            "status": "success",
            "provider": "CUDAExecutionProvider",
            "device": "cuda:0",
            "provider_device_id": 0,
            "actions_finite": True,
            "actions_shape": [1, 7],
            "action_samples": 1,
        },
        target="ONNX",
        expected_device="cuda:2",
    )

    assert result["status"] == "failed"
    assert "selected device cuda:2" in result["error"]


@pytest.mark.parametrize("preexisting", [False, True])
def test_matrix_artifact_replace_failure_never_publishes_partial_target(
    preexisting: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from forge import json_artifacts

    target = tmp_path / "summary.json"
    if preexisting:
        target.write_text('{"status": "previous"}\n', encoding="utf-8")

    def fail_replace(_source, _target) -> None:
        raise OSError("injected replace failure")

    monkeypatch.setattr(json_artifacts.os, "replace", fail_replace)

    with pytest.raises(OSError, match="injected replace failure"):
        json_artifacts.write_json_artifact(target, {"status": "completed"})

    if preexisting:
        assert target.read_text(encoding="utf-8") == '{"status": "previous"}\n'
        assert list(tmp_path.iterdir()) == [target]
    else:
        assert not target.exists()
        assert list(tmp_path.iterdir()) == []


def test_matrix_routes_every_persisted_json_through_atomic_writer() -> None:
    source = Path("src/forge/benchmark/matrix.py").read_text(encoding="utf-8")

    assert source.count("write_json_artifact(") == 3
    assert ".write_text(" not in source


def test_matrix_uses_real_artifact_paths_and_writes_fresh_reports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "forge_nano.yaml"
    checkpoint = tmp_path / "final.pt"
    quantized = {
        "qvla_int4": tmp_path / "qvla_4bit.pt",
        "qvla_int8": tmp_path / "qvla_8bit.pt",
        "turboquant_mse_int4": tmp_path / "turboquant_mse_4bit.pt",
        "turboquant_mse_int8": tmp_path / "turboquant_mse_8bit.pt",
    }
    onnx = tmp_path / "forge.onnx"
    tensorrt = tmp_path / "forge.engine"
    pruned = tmp_path / "pruned.pt"
    data_dir = tmp_path / "real-data"
    data_dir.mkdir()
    for path, contents in ((config, "student: {}\n"), (pruned, "pruned"), (tensorrt, "engine")):
        path.write_text(contents, encoding="utf-8")
    _write_inline_onnx(onnx)
    config_sha256 = _sha256(config)
    configured = load_forge_config(config, required=True)
    torch.save(
        {
            "step": 2000,
            "student_config": asdict(configured.student),
            "provenance": _real_provenance(),
        },
        checkpoint,
    )
    pruning_metadata = {
        "removed_layers": [1, 2, 3, 4],
        "target_layers": 4,
        "calibration_provenance": "real",
    }
    for name, path in quantized.items():
        method = "qvla" if name.startswith("qvla_") else "turboquant-mse"
        bits = 4 if name.endswith("int4") else 8
        torch.save(
            {
                "config_sha256": config_sha256,
                "source_checkpoint_sha256": _sha256(pruned),
                "pruning": pruning_metadata,
                "quantization": {
                    "schema": "forge.packed-state.v1",
                    "method": method,
                    "bits": bits,
                },
            },
            path,
        )
    training = _write_json(
        tmp_path / "training.json",
        {
            "status": "completed",
            "config_sha256": config_sha256,
            "device": "cuda",
            "distill": {
                "total_steps": 2000,
                "elapsed_seconds": 4000.0,
                "steps_per_second": 0.5,
                "initial_loss": 1.5,
                "final_loss": 0.25,
                "best_loss": 0.2,
                "loss_reduction_percent": 42.0,
                "cuda_memory": {"target_60_80_percent_met": True},
                "device": "cuda",
                "provenance": _real_provenance(),
                "checkpoint_dir": str(tmp_path),
            },
        },
    )
    compression = _write_json(
        tmp_path / "compression.json",
        {
            "status": "completed",
            "config_sha256": config_sha256,
            "execution": {"schema": "forge.pipeline-execution.v1", "git_sha": "compress-sha"},
            "provenance": _real_provenance(),
            "pruning": {
                "status": "success",
                "n_removed": 4,
                "removed_layers": [1, 2, 3, 4],
                "path": str(pruned),
                "sha256": _sha256(pruned),
            },
            "compression": {"status": "success"},
            "quantization": {"status": "success", "serialization_schema": "forge.packed-state.v1"},
        },
    )
    export = _write_json(
        tmp_path / "export.json",
        {
            "status": "completed",
            "config_sha256": config_sha256,
            "execution": {"schema": "forge.pipeline-execution.v1", "git_sha": "export-sha"},
            "provenance": _real_provenance(),
            "export_runtime_inputs": {
                "status": "success",
                "samples": 1,
                "labels_provenance": "real",
            },
            "source_checkpoint": str(quantized["qvla_int4"]),
            "export_onnx": {"status": "success", "path": str(onnx)},
            "export_tensorrt": {"status": "success", "precision": "int8", "path": str(tensorrt)},
        },
    )
    compression_payload = json.loads(compression.read_text(encoding="utf-8"))
    compression_payload["source_checkpoint"] = str(checkpoint)
    compression_payload["source_checkpoint_sha256"] = _sha256(checkpoint)
    compression_payload["compression"]["path"] = str(quantized["qvla_int4"])
    compression_payload["compression"]["sha256"] = _sha256(quantized["qvla_int4"])
    compression.write_text(json.dumps(compression_payload), encoding="utf-8")
    runtime_inputs = tmp_path / "tensorrt_calibration.npz"
    np.savez(
        runtime_inputs,
        images=np.ones((1, 3, 8, 8), dtype=np.float32),
        language_ids=np.ones((1, 4), dtype=np.int64),
    )
    export_payload = json.loads(export.read_text(encoding="utf-8"))
    export_payload["source_checkpoint_sha256"] = _sha256(quantized["qvla_int4"])
    export_payload["export_runtime_inputs"].update(
        path=str(runtime_inputs),
        sha256=_sha256(runtime_inputs),
    )
    export_payload["export_onnx"]["artifacts_sha256"] = {onnx.name: _sha256(onnx)}
    export_payload["export_tensorrt"]["sha256"] = _sha256(tensorrt)
    export.write_text(json.dumps(export_payload), encoding="utf-8")
    manifest = _write_json(
        tmp_path / "manifest.json",
        {
            "schema": MANIFEST_SCHEMA,
            "variants": [
                {
                    "variant": "nano",
                    "expected_training_steps": 2000,
                    "evidence_profile": "sha256-v1",
                    "config": config.name,
                    "config_sha256": config_sha256,
                    "checkpoint": checkpoint.name,
                    "training_summary": training.name,
                    "compression_summary": compression.name,
                    "export_summary": export.name,
                    "runtime_inputs": runtime_inputs.name,
                    "data_dir": data_dir.name,
                    "instruction": "push the block to the target",
                    "onnx": onnx.name,
                    "tensorrt": tensorrt.name,
                    "quantized": {name: path.name for name, path in quantized.items()},
                    "quantized_sha256": {name: _sha256(path) for name, path in quantized.items()},
                    "training_summary_sha256": _sha256(training),
                    "training_checkpoint_sha256": _sha256(checkpoint),
                }
            ],
        },
    )

    pytorch_calls: list[dict[str, object]] = []

    def fake_pytorch_benchmark(**kwargs):
        pytorch_calls.append(kwargs)
        return {
            "status": "success",
            "device": "cuda",
            "execution": {"requested_device": "cuda", "resolved_device": "cuda"},
            "actions_finite": True,
            "actions_shape": [1, 7],
            "action_samples": 1,
            "input_provenance": {"kind": "real"},
            "latency": {"mean_ms": 5.0},
            "compression": {"student_params_m": 123.0},
        }

    monkeypatch.setattr("forge.benchmark.matrix._run_pytorch_benchmark", fake_pytorch_benchmark)
    onnx_call: dict[str, object] = {}
    tensorrt_call: dict[str, object] = {}

    def fake_onnx_runtime(*_args, **kwargs):
        onnx_call.update(kwargs)
        return {
            "status": "success",
            "provider": "CUDAExecutionProvider",
            "device": "cuda",
            "provider_device_id": 0,
            "actions_finite": True,
            "actions_shape": [1, 7],
            "action_samples": 1,
            "fps": 100.0,
        }

    def fake_tensorrt_runtime(*_args, **kwargs):
        tensorrt_call.update(kwargs)
        return {
            "status": "success",
            "provider": "TensorRT",
            "precision": "int8",
            "fps": 250.0,
            "actions_finite": True,
        }

    monkeypatch.setattr("forge.benchmark.matrix.benchmark_onnx_runtime", fake_onnx_runtime)
    monkeypatch.setattr("forge.benchmark.matrix.benchmark_tensorrt_runtime", fake_tensorrt_runtime)

    results_dir = tmp_path / "results"
    summary = run_validation_matrix(
        manifest,
        results_dir=results_dir,
        device="cuda",
        samples=2,
        duration=0.1,
        onnx_warmup=1,
        onnx_runs=2,
    )

    result = summary["variants"]["nano"]
    assert summary["status"] == "completed"
    assert summary["execution"]["schema"] == "forge.benchmark-execution.v1"
    assert summary["execution"]["command"] == "matrix"
    assert summary["execution"]["git_sha"]
    assert result["execution"] == summary["execution"]
    assert result["training"]["total_steps"] == 2000
    assert result["compression"]["execution"]["git_sha"] == "compress-sha"
    assert result["export"]["execution"]["git_sha"] == "export-sha"
    assert result["runtime_inputs"]["provenance"]["labels_provenance"] == "real"
    assert len(pytorch_calls) == 5
    assert {call["checkpoint"] for call in pytorch_calls} == {checkpoint, *quantized.values()}
    assert all(call["data_dir"] == data_dir for call in pytorch_calls)
    assert all(call["instruction"] == "push the block to the target" for call in pytorch_calls)
    assert torch.equal(onnx_call["images"], torch.ones(1, 3, 8, 8))
    assert torch.equal(onnx_call["language_ids"], torch.ones(1, 4, dtype=torch.int64))
    assert tensorrt_call["images"] is onnx_call["images"]
    assert tensorrt_call["language_ids"] is onnx_call["language_ids"]
    assert tensorrt_call["precision"] == "int8"
    assert result["onnxruntime"]["provider"] == "CUDAExecutionProvider"
    assert result["tensorrt"]["provider"] == "TensorRT"
    assert result["tensorrt"]["actions_finite"] is True
    assert result["quantized_artifacts"]["qvla_int4"]["size_mb"] == quantized["qvla_int4"].stat().st_size / 1e6
    assert result["quantized_artifacts"]["qvla_int4"]["artifact"] == quantized["qvla_int4"].name
    assert result["quantized_artifacts"]["qvla_int4"]["benchmark"]["status"] == "success"
    assert result["quantized_architecture"] == {
        "status": "success",
        "consistent": True,
        "student_params_m": {
            "qvla_int4": 123.0,
            "qvla_int8": 123.0,
            "turboquant_mse_int4": 123.0,
            "turboquant_mse_int8": 123.0,
        },
        "reference_student_params_m": 123.0,
    }
    assert str(tmp_path) not in json.dumps(summary)
    assert json.loads((results_dir / "nano_validation.json").read_text())["status"] == "completed"
    assert json.loads((results_dir / "summary.json").read_text())["status"] == "completed"

    with pytest.raises(ValueError, match="requires --device cuda"):
        run_validation_matrix(
            manifest,
            results_dir=tmp_path / "cpu-results",
            device="cpu",
            samples=2,
            duration=0.1,
            onnx_warmup=1,
            onnx_runs=2,
        )
    assert not (tmp_path / "cpu-results").exists()


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (lambda summary: summary.update(status="failed"), "status must be completed"),
        (lambda summary: summary["distill"].update(device="cpu"), "prove CUDA execution"),
        (lambda summary: summary["distill"].update(final_loss=2.0), "positive loss improvement"),
        (lambda summary: summary["distill"]["provenance"].update(labels="mock"), "real vision, language"),
        (
            lambda summary: summary["distill"]["cuda_memory"].update(target_60_80_percent_met=False),
            "60–80% CUDA memory gate",
        ),
    ],
)
def test_training_evidence_fails_closed(mutation, error: str) -> None:
    summary = {
        "status": "completed",
        "device": "cuda",
        "distill": {
            "device": "cuda",
            "total_steps": 2000,
            "elapsed_seconds": 4000.0,
            "steps_per_second": 0.5,
            "initial_loss": 1.5,
            "final_loss": 0.25,
            "best_loss": 0.2,
            "loss_reduction_percent": 83.3,
            "provenance": _real_provenance(),
            "cuda_memory": {"target_60_80_percent_met": True},
        },
    }
    mutation(summary)

    with pytest.raises(ValueError, match=error):
        _selected_training_metrics(summary, expected_steps=2000)


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (lambda summary: summary["pruning"].update(status="failed"), "successful pruning"),
        (lambda summary: summary["compression"].update(status="failed"), "successful compression"),
        (lambda summary: summary["quantization"].update(serialization_schema="raw"), "packed quantization"),
        (lambda summary: summary["provenance"].update(vision="mock"), "real vision, language"),
    ],
)
def test_compression_evidence_fails_closed(mutation, error: str) -> None:
    summary = {
        "status": "completed",
        "provenance": _real_provenance(),
        "pruning": {
            "status": "success",
            "n_removed": 2,
            "removed_layers": [3, 7],
            "path": "/artifacts/pruned.pt",
        },
        "compression": {"status": "success"},
        "quantization": {"status": "success", "serialization_schema": "forge.packed-state.v1"},
    }
    mutation(summary)

    with pytest.raises(ValueError, match=error):
        _selected_compression_metrics(summary)


def test_compression_accepts_complete_legacy_pruning_evidence() -> None:
    summary = {
        "status": "completed",
        "provenance": _real_provenance(),
        "pruning": {"n_removed": 2, "removed_layers": [3, 7], "path": "/artifacts/pruned.pt"},
        "compression": {"status": "success"},
        "quantization": {"status": "success", "serialization_schema": "forge.packed-state.v1"},
    }

    selected = _selected_compression_metrics(summary)

    assert selected["pruning"] == {"n_removed": 2, "removed_layers": [3, 7], "path": "pruned.pt"}


@pytest.mark.parametrize(
    "pruning",
    [
        {"n_removed": 0, "removed_layers": [], "path": "/artifacts/pruned.pt"},
        {"n_removed": 2, "removed_layers": [3], "path": "/artifacts/pruned.pt"},
        {"n_removed": 2, "removed_layers": [3, 7], "path": ""},
        {"status": "success", "n_removed": 2, "removed_layers": [3, 7]},
    ],
)
def test_compression_rejects_incomplete_legacy_pruning_evidence(pruning: dict[str, object]) -> None:
    summary = {
        "status": "completed",
        "provenance": _real_provenance(),
        "pruning": pruning,
        "compression": {"status": "success"},
        "quantization": {"status": "success", "serialization_schema": "forge.packed-state.v1"},
    }

    with pytest.raises(ValueError, match="complete successful pruning evidence"):
        _selected_compression_metrics(summary)


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (lambda summary: summary["export_onnx"].update(status="failed"), "successful ONNX"),
        (lambda summary: summary["export_tensorrt"].update(status="failed"), "successful TensorRT"),
        (lambda summary: summary["export_runtime_inputs"].update(labels_provenance="mock"), "real runtime inputs"),
        (lambda summary: summary["provenance"].update(language="mock"), "real vision, language"),
    ],
)
def test_export_evidence_fails_closed(mutation, error: str) -> None:
    summary = {
        "status": "completed",
        "provenance": _real_provenance(),
        "export_runtime_inputs": {"status": "success", "labels_provenance": "real"},
        "export_onnx": {"status": "success"},
        "export_tensorrt": {"status": "success", "precision": "int8"},
    }
    mutation(summary)

    with pytest.raises(ValueError, match=error):
        _selected_export_metrics(summary)


def test_quantized_architecture_rejects_mixed_pruned_and_unpruned_candidates() -> None:
    evidence = _quantized_architecture_evidence(
        {
            "qvla_int4": {"benchmark": {"compression": {"student_params_m": 797.236567}}},
            "qvla_int8": {"benchmark": {"compression": {"student_params_m": 1086.787143}}},
        }
    )

    assert evidence["status"] == "failed"
    assert evidence["consistent"] is False
    assert evidence["errors"] == ["quantized candidates do not share one pruned student architecture"]


@pytest.mark.parametrize(
    ("expected_steps", "acceptance", "error"),
    [
        (None, None, "explicitly declare integer expected_training_steps"),
        (1, None, "expected_training_steps must be 2000"),
        (5000, None, "requires acceptance"),
        (2000, {"kind": "flagship"}, "must not declare flagship"),
    ],
)
def test_manifest_training_step_contract_fails_closed(
    expected_steps: int | None,
    acceptance: dict[str, str] | None,
    error: str,
    tmp_path: Path,
) -> None:
    config = tmp_path / "forge.yaml"
    config.write_text("student:\n  variant: nano\n", encoding="utf-8")
    entry: dict[str, object] = {"variant": "nano", "config": config.name}
    if expected_steps is not None:
        entry["expected_training_steps"] = expected_steps
    if acceptance is not None:
        entry["acceptance"] = acceptance
    manifest = _write_json(tmp_path / "manifest.json", {"schema": MANIFEST_SCHEMA, "variants": [entry]})

    with pytest.raises(ValueError, match=error):
        load_validation_manifest(manifest)


def test_manifest_accepts_only_typed_nano_flagship_contract(tmp_path: Path) -> None:
    config = tmp_path / "forge_nano_flagship.yaml"
    config.write_text(
        "student:\n  variant: nano\n  action_head_type: flow\n  lora_rank: 64\ndistill:\n  max_steps: 5000\n",
        encoding="utf-8",
    )
    manifest = _write_json(
        tmp_path / "manifest.json",
        {
            "schema": MANIFEST_SCHEMA,
            "variants": [
                {
                    "variant": "nano",
                    "config": config.name,
                    "expected_training_steps": 5000,
                    "acceptance": {"kind": "flagship"},
                    "evidence_profile": "sha256-v1",
                    "config_sha256": _sha256(config),
                }
            ],
        },
    )

    _, variants = load_validation_manifest(manifest)

    assert variants[0]["variant"] == "nano"


def test_training_selector_rejects_short_but_improving_run() -> None:
    summary = {
        "status": "completed",
        "distill": {
            "device": "cuda",
            "total_steps": 1,
            "elapsed_seconds": 1.0,
            "steps_per_second": 1.0,
            "initial_loss": 2.0,
            "final_loss": 1.0,
            "best_loss": 1.0,
            "loss_reduction_percent": 50.0,
            "provenance": _real_provenance(),
            "cuda_memory": {"target_60_80_percent_met": True},
        },
    }

    with pytest.raises(ValueError, match="exactly 2000 total_steps"):
        _selected_training_metrics(summary, expected_steps=2000)


@pytest.mark.parametrize("mutation", ["missing", "traversal"])
def test_quantized_candidate_keys_are_exact_and_cannot_traverse(
    mutation: str,
    tmp_path: Path,
) -> None:
    fixture = _build_matrix_fixture(tmp_path)
    manifest = json.loads(fixture["manifest"].read_text(encoding="utf-8"))
    quantized = manifest["variants"][0]["quantized"]
    if mutation == "missing":
        quantized.pop("qvla_int8")
    else:
        quantized["x/../../../escaped"] = quantized["qvla_int8"]
    fixture["manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    _, variants = load_validation_manifest(fixture["manifest"])

    with pytest.raises(ValueError, match="quantized must contain exactly"):
        _prepare_variant(variants[0], base_dir=fixture["manifest"].parent)

    assert not (tmp_path.parent / "escaped.json").exists()


def test_cross_run_compression_summary_swap_is_rejected_before_output(tmp_path: Path) -> None:
    fixture = _build_matrix_fixture(tmp_path)
    compression = json.loads(fixture["compression"].read_text(encoding="utf-8"))
    compression["source_checkpoint"] = str(fixture["quantized_qvla_int8"])
    fixture["compression"].write_text(json.dumps(compression), encoding="utf-8")
    _, variants = load_validation_manifest(fixture["manifest"])

    with pytest.raises(ValueError, match="Compression source checkpoint evidence path does not match"):
        _prepare_variant(variants[0], base_dir=fixture["manifest"].parent)


def test_strict_micro_accepts_manifest_hash_for_pre_hash_training_summary(tmp_path: Path) -> None:
    fixture = _build_matrix_fixture(tmp_path, variant="micro", strict=True)
    _, variants = load_validation_manifest(fixture["manifest"])

    prepared = _prepare_variant(variants[0], base_dir=fixture["manifest"].parent)

    assert prepared["evidence_profile"] == "sha256-v1"
    assert prepared["artifact_evidence"]["checkpoint"]["sha256"]


def test_pre_hash_training_summary_requires_typed_checkpoint_binding(tmp_path: Path) -> None:
    fixture = _build_matrix_fixture(tmp_path, variant="micro", strict=True)
    manifest = json.loads(fixture["manifest"].read_text(encoding="utf-8"))
    del manifest["variants"][0]["training_config_binding"]
    fixture["manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    _, variants = load_validation_manifest(fixture["manifest"])

    with pytest.raises(ValueError, match="training_config_binding='checkpoint-contract-v1'"):
        _prepare_variant(variants[0], base_dir=fixture["manifest"].parent)


def test_training_summary_config_hash_rejects_legacy_binding(tmp_path: Path) -> None:
    fixture = _build_matrix_fixture(tmp_path, variant="micro", strict=True)
    training = json.loads(fixture["training"].read_text(encoding="utf-8"))
    training["config_sha256"] = _sha256(fixture["config"])
    fixture["training"].write_text(json.dumps(training), encoding="utf-8")
    manifest = json.loads(fixture["manifest"].read_text(encoding="utf-8"))
    manifest["variants"][0]["training_summary_sha256"] = _sha256(fixture["training"])
    fixture["manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    _, variants = load_validation_manifest(fixture["manifest"])

    with pytest.raises(ValueError, match="only valid when the training summary lacks"):
        _prepare_variant(variants[0], base_dir=fixture["manifest"].parent)


def test_strict_quantized_alternate_must_derive_from_exact_pruned_hash(tmp_path: Path) -> None:
    fixture = _build_matrix_fixture(tmp_path, variant="micro", strict=True)
    alternate = fixture["quantized_qvla_int8"]
    payload = torch.load(alternate, weights_only=True)
    payload["source_checkpoint_sha256"] = "0" * 64
    torch.save(payload, alternate)
    manifest = json.loads(fixture["manifest"].read_text(encoding="utf-8"))
    manifest["variants"][0]["quantized_sha256"]["qvla_int8"] = _sha256(alternate)
    fixture["manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    _, variants = load_validation_manifest(fixture["manifest"])

    with pytest.raises(ValueError, match="qvla_int8 pruned-source lineage SHA-256 does not match"):
        _prepare_variant(variants[0], base_dir=fixture["manifest"].parent)


def test_strict_quantized_key_must_match_internal_method_and_width(tmp_path: Path) -> None:
    fixture = _build_matrix_fixture(tmp_path, variant="micro", strict=True)
    alternate = fixture["quantized_qvla_int8"]
    payload = torch.load(alternate, weights_only=True)
    payload["quantization"]["bits"] = 4
    torch.save(payload, alternate)
    manifest = json.loads(fixture["manifest"].read_text(encoding="utf-8"))
    manifest["variants"][0]["quantized_sha256"]["qvla_int8"] = _sha256(alternate)
    fixture["manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    _, variants = load_validation_manifest(fixture["manifest"])

    with pytest.raises(ValueError, match="qvla_int8 internal method/width metadata"):
        _prepare_variant(variants[0], base_dir=fixture["manifest"].parent)


def test_matrix_execution_exception_preserves_previous_results_transactionally(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build_matrix_fixture(tmp_path)
    results = tmp_path / "results"
    results.mkdir()
    marker = results / "accepted.json"
    marker.write_text('{"status":"completed"}\n', encoding="utf-8")
    calls = 0

    def fail_second_benchmark(**_kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected late benchmark failure")
        return {
            "status": "success",
            "device": "cuda",
            "execution": {"requested_device": "cuda", "resolved_device": "cuda"},
            "compression": {"student_params_m": 10.0},
        }

    _patch_successful_matrix_runtime(monkeypatch)
    monkeypatch.setattr("forge.benchmark.matrix._run_pytorch_benchmark", fail_second_benchmark)

    with pytest.raises(RuntimeError, match="injected late benchmark failure"):
        run_validation_matrix(fixture["manifest"], results_dir=results)

    assert list(results.iterdir()) == [marker]
    assert not list(tmp_path.glob(".results.staging-*"))


def test_directory_publish_failure_restores_previous_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "results"
    destination.mkdir()
    marker = destination / "accepted.json"
    marker.write_text("accepted", encoding="utf-8")
    staging = tmp_path / ".results.staging-test"
    staging.mkdir()
    (staging / "summary.json").write_text("new", encoding="utf-8")
    real_replace = __import__("os").replace

    def fail_staging_publish(source, target) -> None:
        if Path(source) == staging:
            raise OSError("injected directory publication failure")
        real_replace(source, target)

    monkeypatch.setattr("forge.benchmark.matrix.os.replace", fail_staging_publish)

    with pytest.raises(OSError, match="injected directory publication failure"):
        _publish_results_directory(staging, destination)

    assert marker.read_text(encoding="utf-8") == "accepted"
    assert staging.is_dir()
    assert not list(tmp_path.glob(".results.previous-*"))


def test_failed_or_skipped_backend_never_replaces_accepted_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build_matrix_fixture(tmp_path)
    results = tmp_path / "results"
    results.mkdir()
    marker = results / "accepted.json"
    marker.write_text('{"status":"completed"}\n', encoding="utf-8")
    _patch_successful_matrix_runtime(monkeypatch)
    monkeypatch.setattr(
        "forge.benchmark.matrix.benchmark_onnx_runtime",
        lambda *_args, **_kwargs: {"status": "skipped", "reason": "provider unavailable"},
    )

    summary = run_validation_matrix(fixture["manifest"], results_dir=results)

    assert summary["status"] == "failed"
    assert summary["variants"]["nano"]["onnxruntime"]["status"] == "failed"
    assert list(results.iterdir()) == [marker]
    assert not list(tmp_path.glob(".results.staging-*"))


@pytest.mark.parametrize(
    "report",
    [
        {
            "status": "success",
            "device": "cpu",
            "execution": {"requested_device": "cuda", "resolved_device": "cuda"},
            "compression": {"student_params_m": 10.0},
        },
        {
            "status": "success",
            "device": "cuda",
            "execution": {"requested_device": "cuda", "resolved_device": "cpu"},
            "compression": {"student_params_m": 10.0},
        },
    ],
)
def test_pytorch_and_quantized_reports_must_prove_cuda(
    report: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build_matrix_fixture(tmp_path)
    _patch_successful_matrix_runtime(monkeypatch)
    monkeypatch.setattr("forge.benchmark.matrix._run_pytorch_benchmark", lambda **_kwargs: report)

    summary = run_validation_matrix(fixture["manifest"], results_dir=tmp_path / "results")

    assert summary["status"] == "failed"
    assert summary["variants"]["nano"]["pytorch"]["status"] == "failed"
    assert all(
        candidate["benchmark"]["status"] == "failed"
        for candidate in summary["variants"]["nano"]["quantized_artifacts"].values()
    )
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize(
    "tensorrt_result",
    [
        {"status": "success", "provider": "CPU", "precision": "int8", "actions_finite": True},
        {"status": "success", "provider": "TensorRT", "precision": "fp16", "actions_finite": True},
        {"status": "success", "provider": "TensorRT", "precision": "int8", "actions_finite": False},
    ],
)
def test_tensorrt_success_contract_fails_closed(
    tensorrt_result: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build_matrix_fixture(tmp_path)
    _patch_successful_matrix_runtime(monkeypatch)
    monkeypatch.setattr(
        "forge.benchmark.matrix.benchmark_tensorrt_runtime",
        lambda *_args, **_kwargs: tensorrt_result,
    )

    summary = run_validation_matrix(fixture["manifest"], results_dir=tmp_path / "results")

    assert summary["status"] == "failed"
    assert summary["variants"]["nano"]["tensorrt"]["status"] == "failed"
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize("missing_field", ["evidence_profile", "config_sha256"])
def test_every_variant_requires_strict_profile_and_config_hash(missing_field: str, tmp_path: Path) -> None:
    fixture = _build_matrix_fixture(tmp_path)
    manifest = json.loads(fixture["manifest"].read_text(encoding="utf-8"))
    manifest["variants"][0].pop(missing_field)
    fixture["manifest"].write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=missing_field):
        load_validation_manifest(fixture["manifest"])


def test_pre_hash_d1_checkpoint_binds_every_student_config_field(tmp_path: Path) -> None:
    fixture = _build_matrix_fixture(tmp_path)
    checkpoint = fixture["checkpoint"]
    payload = torch.load(checkpoint, weights_only=True)
    payload["student_config"]["action_diffusion_steps"] += 1
    torch.save(payload, checkpoint)
    manifest = json.loads(fixture["manifest"].read_text(encoding="utf-8"))
    manifest["variants"][0]["training_checkpoint_sha256"] = _sha256(checkpoint)
    fixture["manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    _, variants = load_validation_manifest(fixture["manifest"])

    with pytest.raises(ValueError, match="architecture differs.*action_diffusion_steps"):
        _prepare_variant(variants[0], base_dir=fixture["manifest"].parent)


def test_pre_hash_d1_checkpoint_requires_matching_real_provenance(tmp_path: Path) -> None:
    fixture = _build_matrix_fixture(tmp_path)
    checkpoint = fixture["checkpoint"]
    payload = torch.load(checkpoint, weights_only=True)
    payload["provenance"]["labels"] = "mock"
    torch.save(payload, checkpoint)
    manifest = json.loads(fixture["manifest"].read_text(encoding="utf-8"))
    manifest["variants"][0]["training_checkpoint_sha256"] = _sha256(checkpoint)
    fixture["manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    _, variants = load_validation_manifest(fixture["manifest"])

    with pytest.raises(ValueError, match="checkpoint provenance labels must be real"):
        _prepare_variant(variants[0], base_dir=fixture["manifest"].parent)


def test_primary_qvla_must_derive_from_exact_pruned_hash(tmp_path: Path) -> None:
    fixture = _build_matrix_fixture(tmp_path)
    primary = fixture["quantized_qvla_int4"]
    payload = torch.load(primary, weights_only=True)
    payload["source_checkpoint_sha256"] = "0" * 64
    torch.save(payload, primary)
    primary_sha = _sha256(primary)
    compression = json.loads(fixture["compression"].read_text(encoding="utf-8"))
    compression["compression"]["sha256"] = primary_sha
    fixture["compression"].write_text(json.dumps(compression), encoding="utf-8")
    export = json.loads(fixture["export"].read_text(encoding="utf-8"))
    export["source_checkpoint_sha256"] = primary_sha
    fixture["export"].write_text(json.dumps(export), encoding="utf-8")
    manifest = json.loads(fixture["manifest"].read_text(encoding="utf-8"))
    manifest["variants"][0]["quantized_sha256"]["qvla_int4"] = primary_sha
    fixture["manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    _, variants = load_validation_manifest(fixture["manifest"])

    with pytest.raises(ValueError, match="qvla_int4 pruned-source lineage SHA-256 does not match"):
        _prepare_variant(variants[0], base_dir=fixture["manifest"].parent)


def test_matrix_accepts_exact_arbitrary_external_data_family(tmp_path: Path) -> None:
    fixture = _build_matrix_fixture(tmp_path)
    sidecar = _add_external_onnx_family(fixture)
    _, variants = load_validation_manifest(fixture["manifest"])

    prepared = _prepare_variant(variants[0], base_dir=fixture["manifest"].parent)

    assert prepared["artifact_evidence"]["onnx_artifact_family"] == {
        fixture["onnx"].name: _sha256(fixture["onnx"]),
        sidecar.name: _sha256(sidecar),
    }


def test_matrix_rejects_tampered_external_data(tmp_path: Path) -> None:
    fixture = _build_matrix_fixture(tmp_path)
    sidecar = _add_external_onnx_family(fixture)
    sidecar.write_bytes(b"tampered")
    _, variants = load_validation_manifest(fixture["manifest"])

    with pytest.raises(ValueError, match="weights.bin SHA-256 does not match"):
        _prepare_variant(variants[0], base_dir=fixture["manifest"].parent)


@pytest.mark.parametrize("declared_change", ["undeclared", "extra"])
def test_matrix_requires_exact_declared_external_data_family(tmp_path: Path, declared_change: str) -> None:
    fixture = _build_matrix_fixture(tmp_path)
    sidecar = _add_external_onnx_family(fixture)
    export = json.loads(fixture["export"].read_text(encoding="utf-8"))
    family = export["export_onnx"]["artifacts_sha256"]
    if declared_change == "undeclared":
        family.pop(sidecar.name)
    else:
        unused = fixture["onnx"].parent / "unused.bin"
        unused.write_bytes(b"unused")
        family[unused.name] = _sha256(unused)
    fixture["export"].write_text(json.dumps(export), encoding="utf-8")
    _, variants = load_validation_manifest(fixture["manifest"])

    with pytest.raises(ValueError, match="does not match graph references"):
        _prepare_variant(variants[0], base_dir=fixture["manifest"].parent)


def test_pytorch_finite_action_evidence_is_mandatory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _build_matrix_fixture(tmp_path)
    _patch_successful_matrix_runtime(monkeypatch)

    def nonfinite_report(**_kwargs):
        return {
            "status": "success",
            "device": "cuda",
            "execution": {"requested_device": "cuda", "resolved_device": "cuda"},
            "actions_finite": False,
            "actions_shape": [1, 7],
            "action_samples": 1,
            "input_provenance": {"kind": "real"},
            "compression": {"student_params_m": 10.0},
        }

    monkeypatch.setattr("forge.benchmark.matrix._run_pytorch_benchmark", nonfinite_report)

    summary = run_validation_matrix(fixture["manifest"], results_dir=tmp_path / "results")

    assert summary["status"] == "failed"
    assert "finite actions" in summary["variants"]["nano"]["pytorch"]["error"]
    assert not (tmp_path / "results").exists()


def test_onnx_finite_action_evidence_is_mandatory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _build_matrix_fixture(tmp_path)
    _patch_successful_matrix_runtime(monkeypatch)
    monkeypatch.setattr(
        "forge.benchmark.matrix.benchmark_onnx_runtime",
        lambda *_args, **_kwargs: {
            "status": "success",
            "provider": "CUDAExecutionProvider",
            "device": "cuda",
            "provider_device_id": 0,
            "actions_finite": False,
            "actions_shape": [1, 7],
            "action_samples": 1,
        },
    )

    summary = run_validation_matrix(fixture["manifest"], results_dir=tmp_path / "results")

    assert summary["status"] == "failed"
    assert "finite actions" in summary["variants"]["nano"]["onnxruntime"]["error"]
    assert not (tmp_path / "results").exists()


def test_atomic_current_pointer_keeps_previous_summary_visible_on_commit_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build_matrix_fixture(tmp_path)
    results = tmp_path / "results"
    _patch_successful_matrix_runtime(monkeypatch)
    first = run_validation_matrix(fixture["manifest"], results_dir=results)
    accepted_summary = (results / "summary.json").read_text(encoding="utf-8")
    accepted_pointer = (results / "current").readlink()
    real_replace = __import__("os").replace

    def fail_pointer_commit(source, target) -> None:
        if Path(target) == results / "current":
            raise OSError("injected current-pointer commit failure")
        real_replace(source, target)

    monkeypatch.setattr("forge.benchmark.matrix.os.replace", fail_pointer_commit)

    with pytest.raises(OSError, match="current-pointer commit failure"):
        run_validation_matrix(fixture["manifest"], results_dir=results)

    assert first["status"] == "completed"
    assert (results / "summary.json").read_text(encoding="utf-8") == accepted_summary
    assert (results / "current").readlink() == accepted_pointer
    assert (results / "summary.json").is_symlink()
