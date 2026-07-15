"""Benchmark metrics collection and computation.

Provides functions to profile latency, throughput, compression, and
action quality for FORGE models. Results are captured in typed dataclasses.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional


@dataclass
class LatencyMetrics:
    """Latency profiling results."""

    mean_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    samples: int = 0


@dataclass
class ThroughputMetrics:
    """Throughput measurement results."""

    actions_per_second: float = 0.0
    frames_per_second: float = 0.0
    chunk_gain: float = 1.0  # Throughput with chunking / without
    batch_size: int = 1


@dataclass
class CompressionMetrics:
    """Compression quality metrics."""

    teacher_params_b: float = 0.0  # Billions
    student_params_m: float = 0.0  # Millions
    compression_ratio: float = 0.0
    model_size_mb: float = 0.0
    vram_mb: float = 0.0


@dataclass
class QualityMetrics:
    """Action quality metrics."""

    action_mse: float = 0.0
    action_mae: float = 0.0
    temporal_coherence: float = 0.0
    per_dim_mse: list[float] = field(default_factory=list)


@dataclass
class BenchmarkReport:
    """Complete benchmark report."""

    model_name: str = ""
    variant: str = ""
    action_head_type: str = ""
    action_horizon: int = 1
    device: str = ""
    latency: LatencyMetrics = field(default_factory=LatencyMetrics)
    throughput: ThroughputMetrics = field(default_factory=ThroughputMetrics)
    compression: CompressionMetrics = field(default_factory=CompressionMetrics)
    quality: QualityMetrics | None = None
    timestamp: str = ""
    source_checkpoint: str | None = None
    artifact_size_mb: float | None = None
    provenance: dict[str, str] = field(default_factory=dict)
    input_provenance: dict[str, object] = field(default_factory=dict)
    execution: dict[str, str] = field(default_factory=dict)
    actions_finite: bool = False
    actions_shape: list[int] = field(default_factory=list)
    action_samples: int = 0

    def to_dict(self) -> dict[str, object]:
        """Convert to JSON-serializable dict."""
        return cast(dict[str, object], dataclasses.asdict(self))


def validate_action_output(
    model: nn.Module,
    *,
    device: str,
    images: torch.Tensor | None = None,
    language_text: str | list[str] | None = None,
) -> tuple[bool, list[int], int]:
    """Execute one real inference and report finite tensor action evidence."""
    model = model.to(device)
    model.eval()
    benchmark_input = torch.randn(1, 3, 384, 384, device=device) if images is None else images.to(device)
    with torch.no_grad():
        output = (
            model(benchmark_input) if language_text is None else model(benchmark_input, language_text=language_text)
        )
    actions = output.get("actions") if isinstance(output, Mapping) else None
    if not torch.is_tensor(actions) or actions.numel() < 1:
        return False, [], 0
    shape = [int(dimension) for dimension in actions.shape]
    samples = int(actions.shape[0]) if actions.ndim > 0 else 1
    return bool(torch.isfinite(actions).all().item()), shape, samples


def profile_latency(
    model: nn.Module,
    input_shape: tuple[int, ...] = (1, 3, 384, 384),
    n_warmup: int = 10,
    n_samples: int = 100,
    device: str = "cpu",
    images: torch.Tensor | None = None,
    language_text: str | list[str] | None = None,
) -> LatencyMetrics:
    """Profile model inference latency.

    Args:
        model: FORGE student model
        input_shape: Input image tensor shape
        n_warmup: Warmup iterations (not counted)
        n_samples: Measurement iterations
        device: Device to profile on

    Returns:
        LatencyMetrics with percentile statistics
    """
    model = model.to(device)
    model.eval()

    benchmark_input = torch.randn(*input_shape, device=device) if images is None else images.to(device)

    def forward() -> object:
        if language_text is None:
            return model(benchmark_input)
        return model(benchmark_input, language_text=language_text)

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            forward()

    # Synchronize if CUDA
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)

    # Measure
    latencies = []
    with torch.no_grad():
        for _ in range(n_samples):
            if device.startswith("cuda"):
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            forward()
            if device.startswith("cuda"):
                torch.cuda.synchronize(device)
            elapsed = (time.perf_counter() - t0) * 1000  # ms
            latencies.append(elapsed)

    arr = np.array(latencies)
    return LatencyMetrics(
        mean_ms=float(arr.mean()),
        p50_ms=float(np.percentile(arr, 50)),
        p95_ms=float(np.percentile(arr, 95)),
        p99_ms=float(np.percentile(arr, 99)),
        min_ms=float(arr.min()),
        max_ms=float(arr.max()),
        samples=n_samples,
    )


def measure_throughput(
    model: nn.Module,
    action_horizon: int = 1,
    batch_size: int = 1,
    duration_seconds: float = 2.0,
    device: str = "cpu",
    images: torch.Tensor | None = None,
    language_text: str | list[str] | None = None,
) -> ThroughputMetrics:
    """Measure sustained throughput.

    Args:
        model: FORGE student model
        action_horizon: Actions per chunk (H)
        batch_size: Batch size for throughput test
        duration_seconds: Duration to run
        device: Device

    Returns:
        ThroughputMetrics with actions/s and chunk gain
    """
    model = model.to(device)
    model.eval()

    benchmark_input = torch.randn(batch_size, 3, 384, 384, device=device) if images is None else images.to(device)
    effective_batch_size = int(benchmark_input.shape[0])

    def forward() -> object:
        if language_text is None:
            return model(benchmark_input)
        return model(benchmark_input, language_text=language_text)

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            forward()

    if device.startswith("cuda"):
        torch.cuda.synchronize(device)

    # Measure
    n_frames = 0
    t_start = time.perf_counter()
    with torch.no_grad():
        while time.perf_counter() - t_start < duration_seconds:
            forward()
            n_frames += effective_batch_size

    if device.startswith("cuda"):
        torch.cuda.synchronize(device)

    elapsed = time.perf_counter() - t_start
    fps = n_frames / elapsed
    aps = fps * action_horizon  # Each frame produces H actions

    return ThroughputMetrics(
        actions_per_second=aps,
        frames_per_second=fps,
        chunk_gain=float(action_horizon),
        batch_size=effective_batch_size,
    )


def measure_compression(
    model: nn.Module,
    teacher_params_b: float = 7.6,
) -> CompressionMetrics:
    """Measure compression stats.

    Args:
        model: FORGE student model
        teacher_params_b: Teacher model size in billions of params

    Returns:
        CompressionMetrics with ratio, sizes
    """
    total_params = sum(p.numel() for p in model.parameters())
    student_params_m = total_params / 1e6

    # Estimate model size (assuming fp32)
    model_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    model_size_mb = model_size_bytes / (1024 * 1024)

    # Compression ratio
    teacher_params = teacher_params_b * 1e9
    compression_ratio = teacher_params / max(total_params, 1)

    # VRAM estimation (params + gradients + optimizer states ≈ 4x params)
    vram_mb = model_size_mb  # Inference only = params only

    return CompressionMetrics(
        teacher_params_b=teacher_params_b,
        student_params_m=student_params_m,
        compression_ratio=compression_ratio,
        model_size_mb=model_size_mb,
        vram_mb=vram_mb,
    )


def measure_quality(
    model: nn.Module,
    test_data: list[dict[str, Any]],
    device: str = "cpu",
) -> QualityMetrics:
    """Measure action quality against teacher labels.

    Args:
        model: FORGE student model
        test_data: List of dicts with 'image' and 'teacher_action_mean' keys
        device: Device

    Returns:
        QualityMetrics with MSE, MAE, temporal coherence
    """
    from forge.prune_v2 import temporal_coherence_score

    model = model.to(device)
    model.eval()

    all_mse = []
    all_mae = []
    all_tc = []

    with torch.no_grad():
        for batch in test_data:
            images = batch["image"].to(device)
            if images.dim() == 3:
                images = images.unsqueeze(0)

            teacher_actions = batch.get("teacher_action_mean")
            if teacher_actions is None:
                continue
            teacher_actions = teacher_actions.to(device)

            out = model(images)
            student_actions = out["actions"]

            # Flatten if needed for MSE
            if student_actions.dim() == 3:
                student_flat = student_actions.reshape(-1, student_actions.shape[-1])
                teacher_flat = teacher_actions.reshape(-1, teacher_actions.shape[-1])
            else:
                student_flat = student_actions
                teacher_flat = teacher_actions

            mse = functional.mse_loss(student_flat, teacher_flat).item()
            mae = functional.l1_loss(student_flat, teacher_flat).item()
            all_mse.append(mse)
            all_mae.append(mae)

            if student_actions.dim() == 3 and student_actions.shape[1] > 1:
                tc = temporal_coherence_score(student_actions)
                all_tc.append(tc)

    return QualityMetrics(
        action_mse=float(np.mean(all_mse)) if all_mse else 0.0,
        action_mae=float(np.mean(all_mae)) if all_mae else 0.0,
        temporal_coherence=float(np.mean(all_tc)) if all_tc else 0.0,
    )
