"""Inference telemetry commands."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from forge.cli_commands.shared import emit_cli_error, emit_json

telemetry_app = typer.Typer(name="telemetry", help="Inference telemetry (PRD-29)")
console = Console()


@telemetry_app.command("summary")
def telemetry_summary(
    export_path: str = typer.Option("", "--export-path", help="Path to telemetry JSON export"),
    output_json: bool = typer.Option(False, "--json", help="JSON output"),
):
    """Show telemetry summary from exported JSON."""
    if not export_path:
        emit_cli_error(
            "Use --export-path to specify a telemetry JSON file.",
            output_json=output_json,
            exit_code=2,
        )

    path = Path(export_path)
    if not path.is_file():
        emit_cli_error(
            f"File not found: {export_path}",
            output_json=output_json,
            exit_code=2,
        )

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        emit_cli_error(
            f"Could not read telemetry JSON: {exc}",
            output_json=output_json,
            exit_code=2,
        )
    if output_json:
        emit_json(payload)
    else:
        table = Table(title="Inference Telemetry")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        for key, value in payload.items():
            if isinstance(value, dict):
                for nested_key, nested_value in value.items():
                    table.add_row(f"  {key}.{nested_key}", str(nested_value))
            else:
                table.add_row(key, str(value))
        console.print(table)
