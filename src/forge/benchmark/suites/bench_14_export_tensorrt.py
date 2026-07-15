"""Benchmark 14: ONNX Export + TensorRT Conversion.

Tests the full export pipeline:
1. Build FORGE student
2. Export to ONNX
3. Validate ONNX output matches PyTorch
4. Convert to TensorRT (if available)
5. Benchmark inference: PyTorch vs ONNX vs TensorRT
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from forge.benchmark.artifacts import write_json_artifact
from forge.benchmark.suites.real_data import data_provenance, load_real_dataset, real_batch
from forge.benchmark.suites.runtime import export_dir, results_dir
from forge.training_safety import backward_with_finite_gradients

RESULTS_DIR = results_dir()

MODEL_DIR = Path(os.environ.get("FORGE_MODEL_DIR", "./models"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = export_dir()

REQUIRED_STAGE_STATUSES = {
    "onnx_export": "success",
    "onnx_validation": "passed",
    "onnx_runtime": "success",
    "tensorrt_export": "success",
    "tensorrt_runtime": "success",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_size_metrics(primary: Path, artifacts: list[Path]) -> dict[str, object]:
    """Report decimal MB consistently for a graph/engine and its full artifact family."""
    unique = list(dict.fromkeys(path.resolve() for path in artifacts))
    return {
        "graph_size_mb": primary.stat().st_size / 1e6,
        "artifact_size_mb": sum(path.stat().st_size for path in unique) / 1e6,
        "artifact_files": [path.name for path in unique],
        "artifacts_sha256": {path.name: _sha256(path) for path in unique},
    }


def export_onnx_model(student, output_path, image_size=384):
    """Export through FORGE's production Torch-dynamo ONNX path."""
    from forge.export.onnx_export import _onnx_artifact_files, export_onnx

    print(f"  Exporting ONNX to {output_path}...")
    artifact = export_onnx(student, output_path, image_size=image_size, optimize=False)
    metrics = artifact_size_metrics(artifact, _onnx_artifact_files(artifact))
    print(f"  ONNX exported: {metrics['artifact_size_mb']:.1f} MB total")
    return metrics


def validate_onnx(student, onnx_path, images, n_samples=10, image_size=384):
    """Validate through FORGE's production ONNX comparison path."""
    del image_size
    from forge.export.onnx_export import validate_onnx as validate_production_onnx

    return validate_production_onnx(student, onnx_path, n_samples=n_samples, images=images)


def benchmark_onnx(onnx_path, images, image_size=384, n_warmup=10, n_runs=50):
    """Benchmark the required CUDA ONNX Runtime provider."""
    from forge.export.onnx_export import benchmark_onnx_runtime

    return benchmark_onnx_runtime(
        onnx_path,
        device="cuda",
        n_warmup=n_warmup,
        n_runs=n_runs,
        image_size=image_size,
        images=images,
    )


def try_tensorrt_export(onnx_path, trt_path, precision="fp16"):
    """Convert ONNX through FORGE's required production TensorRT exporter."""
    from forge.export.tensorrt_export import export_tensorrt

    try:
        started = time.perf_counter()
        artifact = export_tensorrt(onnx_path, trt_path, precision=precision)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        return {"status": "failed", "reason": str(exc)}
    return {
        "status": "success",
        "precision": precision,
        "artifact": artifact.name,
        "artifact_size_mb": artifact.stat().st_size / 1e6,
        "sha256": _sha256(artifact),
        "build_time_s": round(time.perf_counter() - started, 1),
    }


def clear_previous_export_artifacts(onnx_path: Path, trt_path: Path) -> None:
    """Remove outputs that could make a failed export appear fresh."""
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    for artifact in onnx_path.parent.glob(f"{onnx_path.name}*"):
        if artifact.is_file():
            artifact.unlink()
    trt_path.unlink(missing_ok=True)


def export_tensorrt_after_onnx(
    onnx_result: dict[str, object],
    onnx_path: Path,
    trt_path: Path,
    *,
    precision: str = "fp16",
) -> dict[str, object]:
    """Require a successful export from this run before building TensorRT."""
    if onnx_result.get("status") != "success":
        return {
            "status": "failed",
            "reason": "TensorRT export requires a successful fresh ONNX export",
        }
    return try_tensorrt_export(onnx_path, trt_path, precision=precision)


def benchmark_tensorrt(trt_path, images, image_size=384, n_warmup=10, n_runs=50, precision="unknown"):
    """Benchmark TensorRT with torch-owned CUDA tensors and streams."""
    from forge.export.tensorrt_export import benchmark_tensorrt_runtime

    return benchmark_tensorrt_runtime(
        trt_path,
        image_size=image_size,
        n_warmup=n_warmup,
        n_runs=n_runs,
        images=images,
        precision=precision,
    )


def _failed_stage(exc: Exception) -> dict[str, str]:
    return {"status": "failed", "error": str(exc)[:200]}


def run_onnx_stages(student, onnx_path: Path, images: torch.Tensor, *, device: str) -> dict[str, dict[str, Any]]:
    """Run and attribute ONNX export, validation, and runtime independently."""
    stages: dict[str, dict[str, Any]] = {}

    print("\n4. ONNX Export...")
    started = time.perf_counter()
    try:
        onnx_sizes = export_onnx_model(student, onnx_path)
    except Exception as exc:
        stages["onnx_export"] = _failed_stage(exc)
        stages["onnx_validation"] = {
            "status": "blocked",
            "reason": "ONNX validation requires a successful fresh ONNX export",
        }
        stages["onnx_runtime"] = {
            "status": "blocked",
            "reason": "ONNX runtime requires a successful fresh ONNX export",
        }
        print(f"   ONNX Export failed: {exc}")
        return stages

    stages["onnx_export"] = {
        "status": "success",
        **onnx_sizes,
        "export_time_s": round(time.perf_counter() - started, 1),
        "artifact": onnx_path.name,
    }

    print("\n5. ONNX Validation...")
    restore_error: Exception | None = None
    try:
        student_cpu = student.cpu()
        validation = validate_onnx(student_cpu, onnx_path, images.cpu())
        stages["onnx_validation"] = dict(validation)
        print(f"   Validation: {validation['status']} (max_diff={validation.get('max_diff', 'N/A')})")
    except Exception as exc:
        stages["onnx_validation"] = _failed_stage(exc)
        print(f"   ONNX Validation failed: {exc}")
    finally:
        try:
            student.to(device)
        except Exception as exc:
            restore_error = exc

    print("\n6. ONNX Runtime Benchmark...")
    if restore_error is not None:
        stages["onnx_runtime"] = {
            "status": "failed",
            "error": f"Could not restore student to {device}: {str(restore_error)[:150]}",
        }
        return stages
    try:
        runtime = dict(benchmark_onnx(onnx_path, images))
        runtime.pop("onnx_path", None)
        runtime.pop("artifact_files", None)
        stages["onnx_runtime"] = runtime
        if runtime.get("fps"):
            print(
                f"   ONNX Runtime: {runtime['mean_ms']:.1f}ms ({runtime['fps']} FPS) [{runtime.get('provider', '?')}]"
            )
        else:
            print(f"   ONNX Runtime: {runtime.get('status', 'error')}")
    except Exception as exc:
        stages["onnx_runtime"] = _failed_stage(exc)
        print(f"   ONNX Runtime failed: {exc}")

    return stages


def required_stage_summary(result: dict[str, Any]) -> dict[str, Any]:
    """Return the mandatory export-pipeline aggregate without hiding stage results."""
    failed_stages = [
        stage
        for stage, required_status in REQUIRED_STAGE_STATUSES.items()
        if not isinstance(result.get(stage), dict) or result[stage].get("status") != required_status
    ]
    return {
        "status": "failed" if failed_stages else "success",
        "required_stages": list(REQUIRED_STAGE_STATUSES),
        "failed_stages": failed_stages,
    }


def main():
    if DEVICE == "cpu":
        print("SKIP: No CUDA")
        sys.exit(0)

    print("=== ONNX/TensorRT Export Benchmark ===")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    from forge.config import ForgeConfig
    from forge.student import FORGEStudent

    dataset = load_real_dataset(MODEL_DIR, max_samples=10)
    image_b1, _ = real_batch(dataset, 1, DEVICE)

    result: dict[str, Any] = {
        "benchmark": "export_tensorrt",
        "timestamp": datetime.now(UTC).isoformat(),
        "device": DEVICE,
        "gpu": torch.cuda.get_device_name(0),
        "data_provenance": data_provenance(dataset),
    }

    # Build model (best balanced config: flow + LoRA64)
    config = ForgeConfig.default()
    config.student.variant = "nano"
    config.student.language_model = "Qwen/Qwen3-0.6B"
    config.student.lora_rank = 64
    config.student.action_head_type = "flow"
    config.student.action_dim = dataset.action_dim

    print("\n1. Building model...")
    t0 = time.perf_counter()
    student = FORGEStudent(config.student, model_dir=str(MODEL_DIR)).to(DEVICE)
    build_time = time.perf_counter() - t0
    result["build"] = {
        "time_s": round(build_time, 1),
        "total_params_m": round(student.total_params / 1e6, 1),
    }
    print(f"   Built in {build_time:.1f}s, {result['build']['total_params_m']}M params")

    # Quick real-data training (10 steps) before export.
    print("\n2. Quick fine-tune (10 steps)...")
    student.train()
    opt = torch.optim.AdamW(student.trainable_parameters(), lr=2e-4)
    for step in range(10):
        img, gt = real_batch(dataset, 2, DEVICE, start=step * 2)
        out = student(img, gt_actions=gt)
        backward_with_finite_gradients(out["loss"], student.trainable_parameters())
        opt.step()
        opt.zero_grad()
    del opt
    student.eval()
    print("   Done")

    # PyTorch baseline benchmark
    print("\n3. PyTorch inference benchmark...")
    times_pt = []
    with torch.no_grad():
        for _ in range(5):
            student(image_b1)
        for _ in range(30):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = student(image_b1)
            torch.cuda.synchronize()
            times_pt.append((time.perf_counter() - t0) * 1000)

    pt_arr = np.array(times_pt[5:])
    result["pytorch"] = {
        "p50_ms": round(float(np.percentile(pt_arr, 50)), 2),
        "p95_ms": round(float(np.percentile(pt_arr, 95)), 2),
        "fps": round(float(1000 / pt_arr.mean()), 1),
    }
    print(f"   PyTorch: {pt_arr.mean():.1f}ms ({1000 / pt_arr.mean():.1f} FPS)")

    # ONNX Export. Clear both formats first so no failure can reuse a previous
    # run's graph, external-data file, or engine as release evidence.
    onnx_path = OUTPUT_DIR / "forge_nano_flow.onnx"
    trt_path = OUTPUT_DIR / "forge_nano_flow.engine"
    clear_previous_export_artifacts(onnx_path, trt_path)
    result.update(run_onnx_stages(student, onnx_path, image_b1, device=DEVICE))
    if result.get("onnx_runtime", {}).get("fps", 0) > 0:
        speedup = result["onnx_runtime"]["fps"] / result["pytorch"]["fps"]
        result["onnx_runtime"]["speedup_vs_pytorch"] = round(speedup, 2)

    # TensorRT
    print("\n7. TensorRT Export...")
    trt_result = export_tensorrt_after_onnx(
        result.get("onnx_export", {}),
        onnx_path,
        trt_path,
        precision="fp16",
    )
    result["tensorrt_export"] = trt_result
    print(f"   TensorRT: {trt_result.get('status', 'unknown')}")

    if trt_result.get("status") == "success":
        print("\n8. TensorRT Benchmark...")
        try:
            trt_bench = dict(benchmark_tensorrt(trt_path, image_b1, precision=str(trt_result["precision"])))
            trt_bench.pop("engine_path", None)
            result["tensorrt_runtime"] = trt_bench
            if trt_bench.get("fps"):
                print(f"   TensorRT: {trt_bench['mean_ms']:.1f}ms ({trt_bench['fps']} FPS)")
        except Exception as exc:
            result["tensorrt_runtime"] = _failed_stage(exc)
            print(f"   TensorRT Runtime failed: {exc}")
    else:
        result["tensorrt_runtime"] = {
            "status": "blocked",
            "reason": "TensorRT runtime requires a successful fresh TensorRT export",
        }

    result["pipeline"] = required_stage_summary(result)

    # Summary
    print(f"\n{'=' * 60}")
    print("Export Pipeline Summary")
    print(f"{'=' * 60}")
    print(f"  PyTorch:  {result['pytorch']['p50_ms']:.1f}ms | {result['pytorch']['fps']} FPS")
    if result.get("onnx_export", {}).get("status") == "success":
        print(f"  ONNX:     {result['onnx_export']['artifact_size_mb']:.1f} MB")
        if result.get("onnx_runtime", {}).get("fps"):
            print(f"  ORT:      {result['onnx_runtime']['mean_ms']:.1f}ms | {result['onnx_runtime']['fps']} FPS")
    if result.get("tensorrt_export", {}).get("status") == "success":
        print(f"  TensorRT: {result['tensorrt_export']['artifact_size_mb']:.1f} MB")
        if result.get("tensorrt_runtime", {}).get("fps"):
            print(
                f"  TRT:      {result['tensorrt_runtime']['mean_ms']:.1f}ms | {result['tensorrt_runtime']['fps']} FPS"
            )

    out_path = RESULTS_DIR / "bench_14_export_tensorrt.json"
    write_json_artifact(out_path, result)

    print(f"\nResults saved to {out_path}")
    print("BENCH 14: DONE")

    del student
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
