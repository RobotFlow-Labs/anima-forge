"""Trained-student aliases backed by the real model registry."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from forge.cli_commands.models import models_list
from forge.cli_commands.shared import emit_cli_error, emit_json

students_app = typer.Typer(name="students", help="Registered trained student models")
students_app.command("list")(models_list)
console = Console()


@students_app.command("package")
def students_package(
    checkpoint: Path = typer.Argument(..., help="Accepted trained checkpoint"),
    training_summary: Path = typer.Option(
        ...,
        "--training-summary",
        help="Completed pipeline_summary.json that passed the VRAM gate",
    ),
    output_dir: Path = typer.Option(..., "--output-dir", help="New or empty Hub staging directory"),
    repo_id: str = typer.Option("robotflowlabs/forge-nano", help="Target Hugging Face model repository"),
    output_json: bool = typer.Option(False, "--json", help="Emit one JSON result"),
) -> None:
    """Build a privacy-safe, inference-only Hugging Face checkpoint package."""
    from forge.hub_package import package_hub_checkpoint

    try:
        result = package_hub_checkpoint(
            checkpoint,
            training_summary,
            output_dir,
            repo_id=repo_id,
        )
    except Exception as exc:
        emit_cli_error(str(exc), output_json=output_json, exit_code=2)
    if output_json:
        emit_json(result)
        return
    console.print(f"[green]Hub package ready:[/green] {result['output_dir']}")
    console.print(f"  Artifact SHA-256: {result['artifact_sha256']}")
    console.print(f"  Target repository: {result['repo_id']}")


__all__ = ["students_app", "students_package"]
