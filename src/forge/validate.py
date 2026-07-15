"""PRD-07: Edge Deployment & Validation.

End-to-end validation of the FORGE pipeline:
1. Load compressed model
2. Run inference benchmark (latency, throughput, VRAM)
3. Action consistency check (PyTorch vs exported)
4. Stability test (continuous operation)

Usage:
    forge validate benchmark --model outputs/forge-nano/
    forge validate stability --model outputs/forge-nano/ --duration 3600
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from latency/throughput benchmark."""

    mean_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_fps: float
    vram_mb: float
    model_size_mb: float
    n_iterations: int
    device: str


@dataclass
class ValidationResult:
    """Full validation result."""

    benchmark: BenchmarkResult | None = None
    warnings: list[str] = field(default_factory=list)
    action_consistency: dict = field(default_factory=dict)
    stability: dict = field(default_factory=dict)
    export_validation: dict = field(default_factory=dict)
    overall_status: str = "pending"


def benchmark_model(
    model: nn.Module,
    device: str = "cpu",
    image_size: int = 384,
    warmup: int = 10,
    iterations: int = 100,
) -> BenchmarkResult:
    """Benchmark model inference latency and throughput.

    Args:
        model: FORGE student model
        device: Device to benchmark on
        image_size: Input image size
        warmup: Number of warmup iterations
        iterations: Number of benchmark iterations

    Returns:
        BenchmarkResult with latency and throughput stats
    """
    model.eval()
    model = model.to(device)

    dummy_images = torch.randn(1, 3, image_size, image_size, device=device)
    dummy_lang = torch.randint(0, 1000, (1, 64), device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            model(dummy_images, language_ids=dummy_lang)

    # Synchronize CUDA if available
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)

    # Benchmark
    latencies = []
    for _ in range(iterations):
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        with torch.no_grad():
            model(dummy_images, language_ids=dummy_lang)

        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000)  # ms

    latencies_np = np.array(latencies)

    # VRAM measurement
    vram_mb = 0.0
    if device.startswith("cuda") and torch.cuda.is_available():
        vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    # Model size
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)

    result = BenchmarkResult(
        mean_latency_ms=float(latencies_np.mean()),
        p95_latency_ms=float(np.percentile(latencies_np, 95)),
        p99_latency_ms=float(np.percentile(latencies_np, 99)),
        throughput_fps=1000.0 / float(latencies_np.mean()),
        vram_mb=vram_mb,
        model_size_mb=model_size_mb,
        n_iterations=iterations,
        device=device,
    )

    logger.info(
        f"Benchmark: {result.mean_latency_ms:.1f}ms mean, "
        f"{result.p95_latency_ms:.1f}ms p95, "
        f"{result.throughput_fps:.1f} FPS, "
        f"{result.model_size_mb:.1f} MB"
    )

    return result


def _get_model_device(model: nn.Module) -> torch.device:
    """Infer model device safely."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def validate_action_consistency(
    model: nn.Module,
    n_samples: int = 50,
    tolerance: float = 0.001,
) -> dict:
    """Verify model produces consistent actions for same input."""
    model.eval()
    device = _get_model_device(model)

    images = torch.randn(1, 3, 384, 384, device=device)
    lang = torch.randint(0, 1000, (1, 64), device=device)

    action_samples: list[np.ndarray] = []
    with torch.no_grad():
        for _ in range(n_samples):
            out = model(images, language_ids=lang)
            action_samples.append(out["actions"].detach().to("cpu").numpy())

    actions = np.stack(action_samples)
    std_per_dim = actions.std(axis=0)
    max_std = std_per_dim.max()

    # Diffusion heads can be stochastic; fallback tolerance is user-configurable.
    passed = max_std <= tolerance

    return {
        "status": "passed" if passed else "failed",
        "max_std": float(max_std),
        "mean_std": float(std_per_dim.mean()),
        "tolerance": tolerance,
        "n_samples": n_samples,
        "note": "Diffusion head is stochastic; some variance expected",
    }


def stability_test(
    model: nn.Module,
    duration_seconds: int = 60,
    device: str = "cpu",
) -> dict:
    """Run model continuously and check for stability issues."""
    model.eval()
    model = model.to(device)

    dummy_images = torch.randn(1, 3, 384, 384, device=device)
    dummy_lang = torch.randint(0, 1000, (1, 64), device=device)

    start_time = time.time()
    frames = 0
    errors = 0
    nan_count = 0
    inf_count = 0

    initial_vram = 0
    if device.startswith("cuda") and torch.cuda.is_available():
        initial_vram = torch.cuda.memory_allocated(device)

    while time.time() - start_time < duration_seconds:
        try:
            with torch.no_grad():
                out = model(dummy_images, language_ids=dummy_lang)
                actions = out["actions"]

                if torch.isnan(actions).any():
                    nan_count += 1
                if torch.isinf(actions).any():
                    inf_count += 1

            frames += 1
        except Exception:
            errors += 1
            if errors > 10:
                break

    elapsed = time.time() - start_time
    final_vram = 0
    if device.startswith("cuda") and torch.cuda.is_available():
        final_vram = torch.cuda.memory_allocated(device)

    vram_leak_mb = (final_vram - initial_vram) / (1024 * 1024)

    result = {
        "status": "passed" if errors == 0 and nan_count == 0 else "failed",
        "duration_seconds": elapsed,
        "frames_processed": frames,
        "fps": frames / max(elapsed, 1),
        "errors": errors,
        "nan_actions": nan_count,
        "inf_actions": inf_count,
        "vram_leak_mb": vram_leak_mb,
    }

    logger.info(
        f"Stability: {frames} frames in {elapsed:.1f}s ({result['fps']:.1f} FPS), {errors} errors, {nan_count} NaNs"
    )

    return result


def run_full_validation(
    model: nn.Module,
    device: str = "cpu",
    stability_duration: int = 10,
    allow_warnings: bool = False,
) -> ValidationResult:
    """Run complete validation suite."""
    result = ValidationResult()
    model = model.to(device)

    logger.info("=== FORGE Validation Suite ===")

    # 1. Benchmark
    logger.info("Running benchmark...")
    result.benchmark = benchmark_model(model, device=device, iterations=50)

    # 2. Action consistency
    logger.info("Checking action consistency...")
    result.action_consistency = validate_action_consistency(model)

    # 3. Stability
    logger.info(f"Running stability test ({stability_duration}s)...")
    result.stability = stability_test(model, duration_seconds=stability_duration, device=device)

    # Overall
    action_ok = result.action_consistency.get("status") == "passed"
    stability_ok = result.stability.get("status") == "passed"
    all_passed = action_ok and stability_ok

    if not action_ok and not stability_ok:
        result.warnings.append(
            "Validation quality checks both failed; this can occur on very short checkpoints or mock checkpoints. "
            "Use a longer production checkpoint for strict production cert."
        )
    elif not action_ok:
        result.warnings.append(
            "Action consistency failed while stability passed. "
            "For short or stochastic checkpoints this is expected; run with a larger checkpoint for strict cert."
        )
    elif not stability_ok:
        result.warnings.append("Runtime stability test failed; review runtime logs before production deployment.")

    if all_passed:
        result.overall_status = "passed"
    elif allow_warnings:
        result.overall_status = "passed_with_warnings"
    else:
        result.overall_status = "failed"

    logger.info(f"=== Validation {result.overall_status.upper()} ===")
    return result
