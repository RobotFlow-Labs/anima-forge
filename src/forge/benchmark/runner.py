"""Benchmark runner — orchestrates all metrics collection.

Runs full benchmark suite and provides rich terminal dashboard + JSON export.

Usage:
    runner = BenchmarkRunner(model, config)
    report = runner.run()
    runner.display(report)
    runner.export(report, "benchmark_report.json")
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from forge.benchmark.artifacts import write_json_artifact
from forge.benchmark.metrics import (
    BenchmarkReport,
    measure_compression,
    measure_throughput,
    profile_latency,
    validate_action_output,
)
from forge.config import ForgeConfig

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Runs full benchmark suite and displays results.

    Orchestrates latency profiling, throughput measurement, and
    compression analysis. Outputs results as rich terminal tables
    or JSON export.
    """

    def __init__(
        self,
        model: Any,
        config: ForgeConfig,
        device: str = "cpu",
        teacher_params_b: float = 7.6,
    ) -> None:
        self.model = model
        self.config = config
        self.device = device
        self.teacher_params_b = teacher_params_b

    def run(
        self,
        n_latency_samples: int = 100,
        throughput_duration: float = 2.0,
        *,
        images: Any | None = None,
        language_text: str | list[str] | None = None,
        input_provenance: dict[str, object] | None = None,
    ) -> BenchmarkReport:
        """Run all benchmarks.

        Args:
            n_latency_samples: Number of latency measurements
            throughput_duration: Duration for throughput test (seconds)

        Returns:
            Complete BenchmarkReport
        """
        action_horizon = getattr(self.config.student, "action_horizon", 1)
        action_head_type = getattr(self.config.student, "action_head_type", "diffusion")

        report = BenchmarkReport(
            model_name=f"FORGE-{self.config.student.variant}",
            variant=self.config.student.variant,
            action_head_type=action_head_type,
            action_horizon=action_horizon,
            device=self.device,
            timestamp=datetime.now(UTC).isoformat(),
            input_provenance=(
                dict(input_provenance)
                if input_provenance is not None
                else {"kind": "synthetic", "purpose": "structural-latency-only"}
            ),
        )

        logger.info("Profiling latency...")
        report.latency = profile_latency(
            self.model,
            n_samples=n_latency_samples,
            device=self.device,
            images=images,
            language_text=language_text,
        )

        logger.info("Measuring throughput...")
        report.throughput = measure_throughput(
            self.model,
            action_horizon=action_horizon,
            device=self.device,
            duration_seconds=throughput_duration,
            images=images,
            language_text=language_text,
        )

        logger.info("Measuring compression...")
        report.compression = measure_compression(
            self.model,
            teacher_params_b=self.teacher_params_b,
        )

        logger.info("Validating finite action output...")
        report.actions_finite, report.actions_shape, report.action_samples = validate_action_output(
            self.model,
            device=self.device,
            images=images,
            language_text=language_text,
        )

        return report

    def display(self, report: BenchmarkReport) -> None:
        """Rich terminal dashboard."""
        try:
            self._display_rich(report)
        except ImportError:
            self._display_plain(report)

    def _display_rich(self, report: BenchmarkReport) -> None:
        """Display using rich library."""
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table

        console = Console()

        # Header
        console.print(
            Panel(
                f"[bold cyan]FORGE Benchmark Report[/bold cyan]\n"
                f"Model: {report.model_name} | Head: {report.action_head_type} | "
                f"Horizon: {report.action_horizon} | Device: {report.device}",
                border_style="cyan",
            )
        )

        # Latency table
        table = Table(title="Latency")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Mean", f"{report.latency.mean_ms:.2f} ms")
        table.add_row("P50", f"{report.latency.p50_ms:.2f} ms")
        table.add_row("P95", f"{report.latency.p95_ms:.2f} ms")
        table.add_row("P99", f"{report.latency.p99_ms:.2f} ms")
        table.add_row("Min", f"{report.latency.min_ms:.2f} ms")
        table.add_row("Max", f"{report.latency.max_ms:.2f} ms")
        console.print(table)

        # Throughput table
        table2 = Table(title="Throughput")
        table2.add_column("Metric", style="cyan")
        table2.add_column("Value", style="green")
        table2.add_row("Actions/s", f"{report.throughput.actions_per_second:.0f}")
        table2.add_row("Frames/s", f"{report.throughput.frames_per_second:.0f}")
        table2.add_row("Chunk Gain", f"{report.throughput.chunk_gain:.1f}x")
        console.print(table2)

        # Compression table
        table3 = Table(title="Compression")
        table3.add_column("Metric", style="cyan")
        table3.add_column("Value", style="green")
        table3.add_row("Teacher", f"{report.compression.teacher_params_b:.1f}B params")
        table3.add_row("Student", f"{report.compression.student_params_m:.1f}M params")
        table3.add_row("Ratio", f"{report.compression.compression_ratio:.0f}x")
        table3.add_row("Size", f"{report.compression.model_size_mb:.0f} MB")
        table3.add_row("VRAM", f"{report.compression.vram_mb:.0f} MB")
        console.print(table3)

    def _display_plain(self, report: BenchmarkReport) -> None:
        """Plain text fallback when rich is not available."""
        print(f"=== FORGE Benchmark: {report.model_name} ===")
        print(f"Latency: mean={report.latency.mean_ms:.2f}ms p95={report.latency.p95_ms:.2f}ms")
        print(f"Throughput: {report.throughput.actions_per_second:.0f} actions/s")
        print(
            f"Compression: {report.compression.compression_ratio:.0f}x "
            f"({report.compression.student_params_m:.1f}M params)"
        )

    def export(self, report: BenchmarkReport, path: str) -> None:
        """Export report to JSON.

        Args:
            report: Benchmark report to export
            path: Output JSON file path
        """
        write_json_artifact(path, report.to_dict())
        logger.info(f"Benchmark report exported to {path}")
