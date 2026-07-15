"""PRD-17: Benchmark Suite & Metrics Dashboard tests."""

import json
import tempfile

import pytest

from forge.benchmark.metrics import (
    BenchmarkReport,
    CompressionMetrics,
    LatencyMetrics,
    ThroughputMetrics,
    measure_compression,
    profile_latency,
    validate_action_output,
)
from forge.benchmark.runner import BenchmarkRunner
from forge.config import ForgeConfig


def test_latency_metrics_dataclass():
    """LatencyMetrics stores all percentile fields."""
    metrics = LatencyMetrics(
        mean_ms=10.5,
        p50_ms=10.0,
        p95_ms=15.0,
        p99_ms=20.0,
        min_ms=8.0,
        max_ms=25.0,
        samples=100,
    )
    assert metrics.mean_ms == 10.5
    assert metrics.p95_ms == 15.0
    assert metrics.samples == 100


def test_throughput_metrics_dataclass():
    """ThroughputMetrics stores throughput fields."""
    metrics = ThroughputMetrics(
        actions_per_second=1000.0,
        frames_per_second=125.0,
        chunk_gain=8.0,
        batch_size=1,
    )
    assert metrics.actions_per_second == 1000.0
    assert metrics.chunk_gain == 8.0


def test_compression_metrics_dataclass():
    """CompressionMetrics stores compression fields."""
    metrics = CompressionMetrics(
        teacher_params_b=7.6,
        student_params_m=50.0,
        compression_ratio=152.0,
        model_size_mb=200.0,
        vram_mb=300.0,
    )
    assert metrics.compression_ratio == 152.0
    assert metrics.student_params_m == 50.0


def test_benchmark_report_to_dict():
    """BenchmarkReport serializes to JSON-compatible dict."""
    report = BenchmarkReport(
        model_name="FORGE-nano",
        variant="nano",
        action_head_type="flow",
        action_horizon=8,
        device="cpu",
    )
    d = report.to_dict()
    assert isinstance(d, dict)
    assert d["model_name"] == "FORGE-nano"
    assert d["action_horizon"] == 8
    assert "latency" in d
    assert "throughput" in d
    assert "compression" in d
    assert d["quality"] is None
    assert d["input_provenance"] == {}

    # Should be JSON serializable
    json_str = json.dumps(d)
    assert isinstance(json_str, str)


def test_profile_latency_cpu():
    """profile_latency works on CPU with real model."""
    from forge.config import StudentConfig
    from forge.student import FORGEStudent

    config = StudentConfig()
    model = FORGEStudent(config)

    metrics = profile_latency(model, n_warmup=2, n_samples=5, device="cpu")

    assert metrics.samples == 5
    assert metrics.mean_ms > 0
    assert metrics.p50_ms > 0
    assert metrics.p95_ms >= metrics.p50_ms
    assert metrics.min_ms <= metrics.mean_ms
    assert metrics.max_ms >= metrics.mean_ms


def test_measure_compression_mock():
    """measure_compression returns correct stats for model."""
    from forge.config import StudentConfig
    from forge.student import FORGEStudent

    config = StudentConfig()
    model = FORGEStudent(config)

    metrics = measure_compression(model, teacher_params_b=7.6)

    assert metrics.teacher_params_b == 7.6
    assert metrics.student_params_m > 0
    assert metrics.compression_ratio > 1.0
    assert metrics.model_size_mb > 0


def test_benchmark_runner_init():
    """BenchmarkRunner initializes with model and config."""
    from forge.student import FORGEStudent

    config = ForgeConfig.default()
    model = FORGEStudent(config.student)

    runner = BenchmarkRunner(model, config, device="cpu")
    assert runner.model is model
    assert runner.config is config
    assert runner.device == "cpu"


def test_benchmark_runner_run_cpu():
    """BenchmarkRunner.run() produces complete report on CPU."""
    from forge.student import FORGEStudent

    config = ForgeConfig.default()
    model = FORGEStudent(config.student)

    runner = BenchmarkRunner(model, config, device="cpu")
    report = runner.run(n_latency_samples=3, throughput_duration=0.5)

    assert report.model_name == "FORGE-nano"
    assert report.latency.samples == 3
    assert report.latency.mean_ms > 0
    assert report.input_provenance == {"kind": "synthetic", "purpose": "structural-latency-only"}
    assert report.throughput.frames_per_second > 0
    assert report.compression.compression_ratio > 0
    assert report.actions_finite is True
    assert report.actions_shape
    assert report.action_samples == 1
    assert report.timestamp != ""

    # Test export
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        runner.export(report, f.name)
        with open(f.name) as f2:
            loaded = json.load(f2)
        assert loaded["model_name"] == "FORGE-nano"


def test_benchmark_runner_export_rejects_non_finite_metrics_without_replacing_artifact(tmp_path):
    report = BenchmarkReport(model_name="FORGE-nano")
    report.latency.mean_ms = float("nan")
    artifact = tmp_path / "benchmark.json"
    original = '{"status": "accepted"}\n'
    artifact.write_text(original, encoding="utf-8")
    runner = object.__new__(BenchmarkRunner)

    with pytest.raises(ValueError, match="Out of range float values"):
        runner.export(report, str(artifact))

    assert artifact.read_text(encoding="utf-8") == original


def test_validate_action_output_reports_non_finite_actions() -> None:
    import torch
    from torch import nn

    class NonFiniteModel(nn.Module):
        def forward(self, images, **_kwargs):
            return {"actions": torch.full((len(images), 7), float("nan"), device=images.device)}

    finite, shape, samples = validate_action_output(NonFiniteModel(), device="cpu", images=torch.zeros(1, 3, 8, 8))

    assert finite is False
    assert shape == [1, 7]
    assert samples == 1


def test_validate_action_output_rejects_missing_actions() -> None:
    import torch
    from torch import nn

    class MissingActionsModel(nn.Module):
        def forward(self, images, **_kwargs):
            return {"features": images.mean(dim=(2, 3))}

    finite, shape, samples = validate_action_output(
        MissingActionsModel(),
        device="cpu",
        images=torch.zeros(1, 3, 8, 8),
    )

    assert finite is False
    assert shape == []
    assert samples == 0
