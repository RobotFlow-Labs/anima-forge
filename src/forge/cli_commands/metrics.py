"""Training metrics and monitoring commands."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from forge.cli_commands.shared import emit_cli_error, emit_json

console = Console()
metrics_app = typer.Typer(name="metrics", help="Training metrics & monitoring (PRD-24)")


@metrics_app.command("summary")
def metrics_summary(
    log_dir: str = typer.Option("./logs", help="Metrics log directory"),
    output_json: bool = typer.Option(False, "--json", help="JSON output"),
):
    """Show summary of training metrics from log files."""
    from forge.metrics import JSONLogger

    log_path = Path(log_dir) / "metrics.jsonl"
    if not log_path.is_file():
        emit_cli_error(
            f"No metrics log found at {log_path}",
            output_json=output_json,
            exit_code=2,
        )

    records = JSONLogger.load(log_path)
    if not records:
        emit_cli_error(
            "Empty metrics log",
            output_json=output_json,
            exit_code=1,
        )

    info = {
        "total_records": len(records),
        "first_step": records[0].get("step", 0),
        "last_step": records[-1].get("step", 0),
        "metrics": list(set(k for r in records for k in r if k != "step")),
    }
    info["latest"] = {k: v for k, v in records[-1].items() if k != "step"}

    if output_json:
        emit_json(info)
    else:
        table = Table(title="Training Metrics Summary")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Total records", str(info["total_records"]))
        table.add_row("Step range", f"{info['first_step']} → {info['last_step']}")
        table.add_row("Metrics tracked", ", ".join(info["metrics"]))
        console.print(table)

        if info["latest"]:
            latest_table = Table(title="Latest Values")
            latest_table.add_column("Metric", style="cyan")
            latest_table.add_column("Value", style="green")
            for k, v in info["latest"].items():
                if isinstance(v, float):
                    latest_table.add_row(k, f"{v:.6f}")
                else:
                    latest_table.add_row(k, str(v))
            console.print(latest_table)


@metrics_app.command("export")
def metrics_export(
    log_dir: str = typer.Option("./logs", help="Metrics log directory"),
    output: str = typer.Option("metrics_export.json", help="Output JSON path"),
    output_json: bool = typer.Option(False, "--json", help="JSON output"),
):
    """Export training metrics to JSON."""
    from forge.metrics import JSONLogger

    log_path = Path(log_dir) / "metrics.jsonl"
    if not log_path.is_file():
        emit_cli_error(
            f"No metrics log found at {log_path}",
            output_json=output_json,
            exit_code=2,
        )

    records = JSONLogger.load(log_path)
    try:
        Path(output).write_text(json.dumps(records, indent=2), encoding="utf-8")
    except OSError as exc:
        emit_cli_error(
            f"Could not write metrics export: {exc}",
            output_json=output_json,
            exit_code=2,
        )

    info = {"exported": len(records), "output": output}
    if output_json:
        emit_json(info)
    else:
        console.print(f"[green]Exported {len(records)} records to {output}[/green]")
