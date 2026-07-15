"""Embodiment profile commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from forge.cli_commands.shared import emit_json

console = Console()
embodiment_app = typer.Typer(name="embodiment", help="Robot embodiment profiles")


@embodiment_app.command("list")
def embodiment_list(
    output_json: bool = typer.Option(False, "--json", help="JSON output"),
):
    """List available robot embodiment profiles."""
    from forge.embodiments.registry import EmbodimentRegistry

    registry = EmbodimentRegistry()
    profiles: list[dict[str, Any]] = []
    for name in registry.list_embodiments():
        profile = registry.get(name)
        profiles.append(
            {
                "name": name,
                "dof": profile.dof,
                "action_dim": profile.action_dim,
                "control_frequency_hz": profile.control_frequency_hz,
                "recommended_variant": profile.recommended_variant,
                "recommended_action_head": profile.recommended_action_head,
            }
        )

    if output_json:
        emit_json(profiles)
    else:
        table = Table(title="Robot Embodiment Profiles")
        table.add_column("Name", style="cyan")
        table.add_column("DoF", style="yellow")
        table.add_column("Freq (Hz)")
        table.add_column("Variant", style="green")
        table.add_column("Action Head")
        for p in profiles:
            table.add_row(
                p["name"],
                str(p["dof"]),
                str(p["control_frequency_hz"]),
                p["recommended_variant"],
                p["recommended_action_head"],
            )
        console.print(table)


@embodiment_app.command("config")
def embodiment_config(
    name: str = typer.Argument(..., help="Embodiment name"),
    output: str = typer.Option(None, help="Output file path"),
):
    """Generate YAML config for an embodiment."""
    from forge.embodiments.registry import EmbodimentRegistry

    registry = EmbodimentRegistry()
    yaml_str = registry.generate_yaml_config(name)

    if output:
        Path(output).write_text(yaml_str)
        console.print(f"[green]Config saved to {output}[/green]")
    else:
        typer.echo(yaml_str)
