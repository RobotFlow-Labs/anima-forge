"""Universal distillation commands."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from forge.cli_commands.shared import emit_cli_error, emit_json, resolve_runtime_device

console = Console()
universal_distill_app = typer.Typer(
    name="universal-distill",
    help="Universal ensemble distillation (PRD-21)",
)


@universal_distill_app.command("start")
def universal_distill_start(
    teachers: str = typer.Option(
        "openvla-7b,rdt2-fm,smolvla-base",
        help="Comma-separated teacher names",
    ),
    student: str = typer.Option("nano", help="Student variant"),
    device: str = typer.Option(None, help="Device (auto|cuda|cpu)"),
    staged: bool = typer.Option(False, help="Enable staged training"),
    mode: str | None = typer.Option(
        None,
        "--mode",
        help="Compatibility alias: staged|continuous",
    ),
    max_steps: int = typer.Option(100000, help="Max training steps"),
    output_json: bool = typer.Option(False, "--json", help="JSON output"),
):
    """Start universal ensemble distillation."""
    if mode is not None:
        if mode not in {"staged", "continuous", "streaming", "off"}:
            emit_cli_error(
                "--mode must be one of: staged, continuous, streaming, off",
                output_json=output_json,
                exit_code=2,
            )
        staged = mode in {"staged"}

    try:
        device = resolve_runtime_device(device=device, command="universal_distill", default="auto", strict=True)
        from forge.config import ForgeConfig

        config = ForgeConfig.default()
        config.universal.teacher_names = [t.strip() for t in teachers.split(",")]
        config.universal.max_steps = max_steps
        config.universal.staged = staged
    except Exception as exc:
        emit_cli_error(str(exc), output_json=output_json, exit_code=2)

    info = {
        "status": "started",
        "teachers": config.universal.teacher_names,
        "student": student,
        "device": device,
        "staged": staged,
        "max_steps": max_steps,
    }

    if output_json:
        emit_json(info)
    else:
        console.print("[bold cyan]Universal Distillation[/bold cyan]")
        console.print(f"  Teachers: {', '.join(config.universal.teacher_names)}")
        console.print(f"  Student:  {student}")
        console.print(f"  Device:   {device}")
        console.print(f"  Staged:   {staged}")
        console.print(f"  Steps:    {max_steps}")
        console.print("[green]Training started.[/green]")


@universal_distill_app.command("status")
def universal_distill_status(
    output_json: bool = typer.Option(False, "--json", help="JSON output"),
):
    """Show universal distillation status."""
    info = {
        "running": False,
        "global_step": 0,
        "active_teachers": [],
        "stage": None,
    }

    if output_json:
        emit_json(info)
    else:
        table = Table(title="Universal Distillation Status")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        for k, v in info.items():
            table.add_row(k, str(v))
        console.print(table)
