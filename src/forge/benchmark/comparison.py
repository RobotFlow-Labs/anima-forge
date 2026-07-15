"""Compare benchmark reports across configurations.

Load and compare multiple JSON benchmark reports side-by-side.

Usage:
    reports = [load_report("report_a.json"), load_report("report_b.json")]
    compare_reports(reports)
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping, Sequence
from typing import Any

logger = logging.getLogger(__name__)


def load_report(path: str) -> dict[str, Any]:
    """Load a benchmark report from JSON.

    Args:
        path: Path to JSON report file

    Returns:
        Parsed report dict
    """
    with open(path) as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Benchmark report must contain a JSON object: {path}")
    return payload


_METRICS = (
    ("latency.mean_ms", "Latency (mean)", "ms", "lower"),
    ("latency.p95_ms", "Latency (P95)", "ms", "lower"),
    ("throughput.actions_per_second", "Actions/s", "actions/s", "higher"),
    ("throughput.chunk_gain", "Chunk gain", "x", "higher"),
    ("compression.compression_ratio", "Compression", "x", "higher"),
    ("compression.model_size_mb", "Size", "MB", "lower"),
    ("compression.student_params_m", "Parameters", "M", "lower"),
)


def _finite_metric(report: Mapping[str, Any], path: str) -> float | None:
    value: object = report
    for key in path.split("."):
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def build_comparison(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Build a machine-readable comparison without inventing missing values."""
    sources = [
        {
            "model_name": str(report.get("model_name", "Unknown")),
            "action_head_type": report.get("action_head_type"),
            "source_checkpoint": report.get("source_checkpoint"),
        }
        for report in reports
    ]
    metrics: dict[str, dict[str, Any]] = {}
    for path, label, unit, preferred in _METRICS:
        values = [_finite_metric(report, path) for report in reports]
        delta = None
        relative_change_pct = None
        if len(values) == 2 and values[0] is not None and values[1] is not None:
            delta = values[1] - values[0]
            if values[0] != 0:
                relative_change_pct = delta / abs(values[0]) * 100.0
        metrics[path] = {
            "label": label,
            "unit": unit,
            "preferred": preferred,
            "values": values,
            "delta_report2_minus_report1": delta,
            "relative_change_pct": relative_change_pct,
        }
    return {
        "schema_version": 1,
        "reports": sources,
        "metrics": metrics,
    }


def compare_reports(reports: list[dict], *, comparison: dict[str, Any] | None = None) -> None:
    """Display side-by-side comparison of benchmark reports.

    Args:
        reports: List of report dicts (from load_report or BenchmarkReport.to_dict)
    """
    if not reports:
        print("No reports to compare.")
        return

    payload = comparison or build_comparison(reports)
    try:
        _compare_rich(reports, payload)
    except ImportError:
        _compare_plain(reports, payload)


def _format_metric(value: float | None, unit: str) -> str:
    if value is None:
        return "Not measured"
    if unit == "ms":
        return f"{value:.2f} ms"
    if unit in {"x", "MB", "M"}:
        return f"{value:.1f}{unit}"
    return f"{value:.2f} {unit}"


def _compare_rich(reports: list[dict], comparison: Mapping[str, Any]) -> None:
    """Rich table comparison."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title="Benchmark Comparison")

    table.add_column("Metric", style="cyan")
    for r in reports:
        name = r.get("model_name", "Unknown")
        head = r.get("action_head_type", "?")
        table.add_column(f"{name}\n({head})", style="green")

    metrics = comparison.get("metrics", {})
    if isinstance(metrics, Mapping):
        for metric in metrics.values():
            if not isinstance(metric, Mapping):
                continue
            values = metric.get("values", [])
            if not isinstance(values, list):
                values = []
            row = [str(metric.get("label", "Unknown"))]
            unit = str(metric.get("unit", ""))
            row.extend(
                _format_metric(values[index] if index < len(values) else None, unit) for index in range(len(reports))
            )
            table.add_row(*row)

    console.print(table)


def _compare_plain(reports: list[dict], comparison: Mapping[str, Any]) -> None:
    """Plain text comparison fallback."""
    metrics = comparison.get("metrics", {})
    for i, r in enumerate(reports):
        name = r.get("model_name", "Unknown")
        rendered = []
        if isinstance(metrics, Mapping):
            for metric in metrics.values():
                if not isinstance(metric, Mapping):
                    continue
                values = metric.get("values", [])
                value = values[i] if isinstance(values, list) and i < len(values) else None
                rendered.append(
                    f"{metric.get('label', 'Unknown')}={_format_metric(value, str(metric.get('unit', '')))}"
                )
        print(f"[{i + 1}] {name}: {', '.join(rendered)}")
