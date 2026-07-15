"""Curriculum learning commands."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from forge.cli_commands.shared import emit_cli_error, emit_json

console = Console()
curriculum_app = typer.Typer(
    name="curriculum",
    help="Curriculum learning & adaptive training (PRD-22)",
)


@curriculum_app.command("status")
def curriculum_status(
    run_dir: Path | None = typer.Option(None, "--run-dir", help="Specific training run"),
    output_dir: Path = typer.Option(Path("./outputs"), "--output-dir", help="Training output root"),
    output_json: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Show real curriculum state from a persisted production-training run."""
    from forge.training_runtime import TrainingRuntimeError
    from forge.training_status import read_training_run_status

    try:
        selected, state = read_training_run_status(
            run_dir=run_dir,
            output_dir=output_dir,
        )
    except (OSError, TrainingRuntimeError) as exc:
        emit_cli_error(str(exc), output_json=output_json, exit_code=2)

    curriculum = state.get("curriculum")
    if not isinstance(curriculum, dict):
        emit_cli_error(
            f"Curriculum state is unavailable in legacy heartbeat: {selected / 'train_state.json'}",
            output_json=output_json,
            exit_code=1,
        )

    info = {
        "run_dir": str(selected),
        "run_status": state.get("status"),
        "step": state.get("step"),
        "process_running": state.get("process_running"),
        **curriculum,
    }
    if output_json:
        emit_json(info)
        return

    table = Table(title="Curriculum Training Status")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    for key, value in info.items():
        table.add_row(key, str(value))
    console.print(table)


@curriculum_app.command("simulate")
def curriculum_simulate(
    steps: int = typer.Option(100000, help="Total training steps"),
    schedule: str = typer.Option("linear", help="Schedule: linear/cosine/step"),
    output_json: bool = typer.Option(False, "--json", help="JSON output"),
):
    """Simulate curriculum difficulty schedule."""
    from forge.curriculum import CurriculumScheduler

    try:
        scheduler = CurriculumScheduler(
            initial_difficulty=0.3,
            final_difficulty=1.0,
            ramp_steps=steps,
            schedule=schedule,
        )
    except ValueError as exc:
        emit_cli_error(
            str(exc),
            output_json=output_json,
            exit_code=2,
        )

    checkpoints = [0, steps // 4, steps // 2, 3 * steps // 4, steps]
    schedule_data = [{"step": s, "difficulty": round(scheduler.get_difficulty(s), 4)} for s in checkpoints]

    if output_json:
        emit_json(schedule_data)
    else:
        table = Table(title=f"Curriculum Schedule ({schedule})")
        table.add_column("Step", style="cyan")
        table.add_column("Difficulty", style="green")
        for d in schedule_data:
            table.add_row(str(d["step"]), f"{d['difficulty']:.1%}")
        console.print(table)
