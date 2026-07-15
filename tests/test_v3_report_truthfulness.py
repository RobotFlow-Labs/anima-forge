from __future__ import annotations

from forge.report import generate_report


def test_empty_report_marks_metrics_unmeasured_without_invented_defaults(tmp_path) -> None:
    report = generate_report(results={}, output_path=tmp_path / "report.md")

    assert "Not measured" in report
    assert "~968" not in report
    assert "~129" not in report
    assert "Qwen2.5" not in report
    assert "FORGE 3.0.1" in report


def test_report_renders_supplied_metrics_and_real_provenance(tmp_path) -> None:
    report = generate_report(
        results={
            "device": "cuda",
            "model": {"variant": "small", "total_params_M": 2100.5},
            "inference": {"latency_avg_ms": 12.5, "throughput_fps": 80.0},
            "provenance": {"vision": "real", "language": "real", "labels": "real"},
            "teacher_comparison": {"measured_speedup": "1.8×"},
        },
        output_path=tmp_path / "report.md",
    )

    assert "small" in report
    assert "2100.5 M" in report
    assert "12.5 ms" in report
    assert "80.0 FPS" in report
    assert report.count("| real |") == 3
    assert "measured_speedup" in report


def test_report_renders_public_benchmark_artifact_schema(tmp_path) -> None:
    report = generate_report(
        results={
            "model_name": "FORGE-small",
            "device": "cuda",
            "latency": {"mean_ms": 12.5, "p50_ms": 12.0, "p99_ms": 14.0},
            "throughput": {"actions_per_second": 80.0},
            "compression": {
                "student_params_m": 2481.7,
                "compression_ratio": 3.06,
                "model_size_mb": 6185.3,
                "vram_mb": 6200.0,
            },
            "input_provenance": {
                "kind": "real",
                "dataset": "lerobot/pusht",
                "instruction_source": "cli",
            },
            "source_checkpoint": "/artifacts/small/final.pt",
        },
        output_path=tmp_path / "report.md",
    )

    assert "FORGE-small" in report
    assert "12.5 ms" in report
    assert "80.0 actions/s" in report
    assert "2481.7 M" in report
    assert "6185.3 MB" in report
    assert "lerobot/pusht" in report
    assert "/artifacts/small/final.pt" in report


def test_report_renders_pipeline_summary_distillation_schema(tmp_path) -> None:
    report = generate_report(
        results={
            "config": "small",
            "distill": {
                "total_steps": 2000,
                "elapsed_seconds": 4944.0,
                "steps_per_second": 0.4,
                "initial_loss": 1.36,
                "final_loss": 0.29,
                "loss_reduction_percent": 78.7,
            },
        },
        output_path=tmp_path / "report.md",
    )

    assert "small" in report
    assert "2000" in report
    assert "4944.0 s" in report
    assert "78.7%" in report
