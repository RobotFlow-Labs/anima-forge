"""Tests for PRD-07: Edge Deployment & Validation."""

import torch.nn as nn


class MiniModel(nn.Module):
    """Tiny model for fast validation tests."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, 7)

    def forward(self, images, language_ids=None, **kwargs):
        x = self.conv(images)
        x = self.pool(x).flatten(1)
        return {"actions": self.fc(x)}


def test_benchmark():
    """Verify benchmark produces valid metrics."""
    from forge.validate import benchmark_model

    model = MiniModel()
    result = benchmark_model(model, device="cpu", image_size=32, warmup=2, iterations=10)

    assert result.mean_latency_ms > 0
    assert result.p95_latency_ms >= result.mean_latency_ms
    assert result.throughput_fps > 0
    assert result.model_size_mb > 0
    assert result.n_iterations == 10
    assert result.device == "cpu"


def test_action_consistency():
    """Verify action consistency check (deterministic model)."""
    from forge.validate import validate_action_consistency

    model = MiniModel()
    result = validate_action_consistency(model, n_samples=10)

    assert result["status"] == "passed"
    # Deterministic model should have zero std
    assert result["max_std"] < 0.001


def test_stability():
    """Verify stability test runs and reports metrics."""
    from forge.validate import stability_test

    model = MiniModel()
    result = stability_test(model, duration_seconds=2, device="cpu")

    assert result["status"] == "passed"
    assert result["frames_processed"] > 0
    assert result["fps"] > 0
    assert result["errors"] == 0
    assert result["nan_actions"] == 0


def test_full_validation():
    """Verify full validation suite runs end-to-end."""
    from forge.validate import run_full_validation

    model = MiniModel()
    result = run_full_validation(model, device="cpu", stability_duration=1)

    assert result.overall_status == "passed"
    assert result.benchmark is not None
    assert result.benchmark.throughput_fps > 0
    assert result.action_consistency["status"] == "passed"
    assert result.stability["status"] == "passed"


def test_benchmark_result_dataclass():
    """Verify BenchmarkResult dataclass."""
    from forge.validate import BenchmarkResult

    result = BenchmarkResult(
        mean_latency_ms=35.0,
        p95_latency_ms=42.0,
        p99_latency_ms=48.0,
        throughput_fps=28.5,
        vram_mb=487.0,
        model_size_mb=150.0,
        n_iterations=1000,
        device="cuda",
    )
    assert result.throughput_fps == 28.5
