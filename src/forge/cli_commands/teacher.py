"""Teacher model registry commands."""

from __future__ import annotations

from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from forge.cli_commands.shared import emit_json

console = Console()
teacher_app = typer.Typer(name="teacher", help="Teacher model management")


@teacher_app.command("list")
def teacher_list(
    output_json: bool = typer.Option(False, "--json", help="JSON output"),
):
    """List available teacher models."""
    from forge.teachers.registry import get_registry

    registry = get_registry()
    teachers: list[dict[str, Any]] = []
    for name in registry.list_teachers():
        adapter = registry.create(name)
        info = adapter.info()
        teachers.append(
            {
                "name": info.name,
                "architecture": info.architecture,
                "params_b": info.param_count,
                "action_dim": info.action_dim,
                "supports_chunking": info.supports_chunking,
            }
        )

    if output_json:
        emit_json(teachers)
    else:
        table = Table(title="Available Teachers")
        table.add_column("Name", style="cyan")
        table.add_column("Architecture", style="green")
        table.add_column("Params", style="yellow")
        table.add_column("Action Dim")
        table.add_column("Chunking")
        for t in teachers:
            table.add_row(
                t["name"],
                t["architecture"],
                f"{t['params_b']:.1f}B",
                str(t["action_dim"]),
                "Yes" if t["supports_chunking"] else "No",
            )
        console.print(table)
