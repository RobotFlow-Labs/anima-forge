"""Model registry commands."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from forge.cli_commands.shared import emit_cli_error, emit_json

console = Console()
models_app = typer.Typer(name="models", help="Trained student model registry (PRD-26)")


@models_app.command("list")
def models_list(
    variant: str = typer.Option(None, help="Filter by variant (nano/small/micro)"),
    tag: str = typer.Option(None, help="Filter by tag (e.g., production)"),
    registry_dir: str = typer.Option("./outputs/registry", help="Registry directory"),
    output_json: bool = typer.Option(False, "--json", help="JSON output"),
):
    """List registered trained models."""
    from forge.model_registry import ModelRegistry

    registry = ModelRegistry(registry_dir)
    entries = registry.list_models(variant=variant, tag=tag)

    if output_json:
        emit_json([e.to_dict() for e in entries])
    else:
        if not entries:
            console.print("[yellow]No models registered[/yellow]")
            return
        table = Table(title=f"Registered Models ({len(entries)})")
        table.add_column("ID", style="cyan", max_width=10)
        table.add_column("Name", style="green")
        table.add_column("Variant", style="yellow")
        table.add_column("Steps")
        table.add_column("Best Loss")
        table.add_column("Tags", style="magenta")
        for e in entries:
            table.add_row(
                e.model_id[:8],
                e.display_name,
                e.variant,
                str(e.total_steps),
                f"{e.best_loss:.4f}" if e.best_loss < float("inf") else "—",
                ", ".join(e.tags) if e.tags else "—",
            )
        console.print(table)
        for entry in entries:
            if entry.is_mock:
                console.print(f"{entry.model_id[:8]} {entry.mock_warning}")


@models_app.command("show")
def models_show(
    model_id: str = typer.Argument(..., help="Model ID (prefix match supported)"),
    registry_dir: str = typer.Option("./outputs/registry", help="Registry directory"),
    output_json: bool = typer.Option(False, "--json", help="JSON output"),
):
    """Show details for a registered model."""
    from forge.model_registry import ModelRegistry

    registry = ModelRegistry(registry_dir)
    entry = registry.get(model_id)

    if entry is None:
        emit_cli_error(
            f"Model {model_id} not found",
            output_json=output_json,
            exit_code=2,
        )

    if output_json:
        emit_json(entry.to_dict())
    else:
        console.print(entry.summary())


@models_app.command("best")
def models_best(
    by: str = typer.Option("best_loss", help="Metric to rank by"),
    variant: str = typer.Option(None, help="Filter by variant"),
    higher: bool = typer.Option(False, help="Higher is better (default: lower)"),
    registry_dir: str = typer.Option("./outputs/registry", help="Registry directory"),
    output_json: bool = typer.Option(False, "--json", help="JSON output"),
):
    """Find the best model by a metric."""
    from forge.model_registry import ModelRegistry

    registry = ModelRegistry(registry_dir)
    entry = registry.best(by=by, variant=variant, lower_is_better=not higher)

    if entry is None:
        emit_cli_error(
            "No models found",
            output_json=output_json,
            exit_code=1,
        )

    if output_json:
        emit_json(entry.to_dict())
    else:
        console.print(f"[bold green]Best by {by}:[/bold green]")
        console.print(entry.summary())


@models_app.command("promote")
def models_promote(
    model_id: str = typer.Argument(..., help="Model ID"),
    tag: str = typer.Option("production", help="Tag to assign"),
    registry_dir: str = typer.Option("./outputs/registry", help="Registry directory"),
    output_json: bool = typer.Option(False, "--json", help="JSON output"),
):
    """Promote a model by adding a tag (removes tag from others)."""
    from forge.model_registry import ModelRegistry

    registry = ModelRegistry(registry_dir)
    entry = registry.promote(model_id, tag=tag)

    if entry is None:
        emit_cli_error(
            f"Model {model_id} not found",
            output_json=output_json,
            exit_code=2,
        )

    if output_json:
        emit_json(entry.to_dict())
    else:
        console.print(f"[green]Promoted [{entry.model_id[:8]}] → {tag}[/green]")


@models_app.command("compare")
def models_compare(
    id1: str = typer.Argument(..., help="First model ID"),
    id2: str = typer.Argument(..., help="Second model ID"),
    registry_dir: str = typer.Option("./outputs/registry", help="Registry directory"),
    output_json: bool = typer.Option(False, "--json", help="JSON output"),
):
    """Compare two registered models."""
    from forge.model_registry import ModelRegistry

    registry = ModelRegistry(registry_dir)
    result = registry.compare(id1, id2)
    if "error" in result:
        emit_cli_error(
            str(result["error"]),
            output_json=output_json,
            exit_code=2,
        )

    if output_json:
        emit_json(result)
    else:
        console.print("[bold cyan]Comparing models[/bold cyan]")
        console.print(f"  Model 1: {result['model_1']['name']} [{result['model_1']['id'][:8]}]")
        console.print(f"  Model 2: {result['model_2']['name']} [{result['model_2']['id'][:8]}]")
        console.print()

        if result["differences"]:
            table = Table(title="Differences")
            table.add_column("Field", style="cyan")
            table.add_column("Model 1", style="green")
            table.add_column("Model 2", style="yellow")
            for field_name, vals in result["differences"].items():
                v1 = vals["model_1"]
                v2 = vals["model_2"]
                if isinstance(v1, float):
                    v1 = f"{v1:.4f}"
                if isinstance(v2, float):
                    v2 = f"{v2:.4f}"
                table.add_row(field_name, str(v1), str(v2))
            console.print(table)
        else:
            console.print("[yellow]Models are identical[/yellow]")
