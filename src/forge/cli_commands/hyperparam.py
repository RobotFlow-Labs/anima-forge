"""Hyperparameter search and optimization commands."""

from __future__ import annotations

import io
import json
from contextlib import nullcontext, redirect_stdout
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from forge.cli_commands.shared import emit_cli_error, emit_json, resolve_runtime_device

console = Console()
hyperparam_app = typer.Typer(name="hyperparam", help="Hyperparameter search & optimization (PRD-27)")


@hyperparam_app.command("status")
def hyperparam_status(
    results_dir: str = typer.Option("./outputs/hyperparam", help="Results directory"),
    output_json: bool = typer.Option(False, "--json", help="JSON output"),
):
    """Show hyperparameter search status and best results."""
    from forge.hyperparam import HyperparamSearch, SearchSpace

    search = HyperparamSearch(SearchSpace(), results_dir=results_dir)
    info = search.summary()

    if output_json:
        emit_json(info)
    else:
        table = Table(title="Hyperparameter Search Status")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        for k, v in info.items():
            if isinstance(v, float):
                table.add_row(k, f"{v:.6f}")
            elif isinstance(v, dict):
                table.add_row(k, json.dumps(v, default=str))
            else:
                table.add_row(k, str(v))
        console.print(table)


@hyperparam_app.command("recommend")
def hyperparam_recommend(
    results_dir: str = typer.Option("./benchmarks", help="Benchmark results directory"),
    objective: str = typer.Option("balanced", help="Objective: balanced, speed, quality, size"),
    top_n: int = typer.Option(3, "--top", help="Number of recommendations"),
    output_json: bool = typer.Option(False, "--json", help="JSON output"),
):
    """Rank configuration recommendations from benchmark data."""
    from forge.hyperparam import recommend_config

    recs = recommend_config(results_dir, objective=objective, top_n=top_n)

    if output_json:
        emit_json(recs)
    else:
        if not recs:
            console.print("[yellow]No benchmark data found. Run `forge benchmark suite 12` first.[/yellow]")
            return
        console.print(f"\n[bold]FORGE Hyperparameter Recommendations[/bold] (objective: {objective})\n")
        for rec in recs:
            style = "bold green" if rec["rank"] == 1 else "white"
            console.print(f"  [{style}]#{rec['rank']}[/{style}] {rec['name']} — score: {rec['score']}")
            m = rec["metrics"]
            console.print(f"     Config: {rec['config']}")
            console.print(
                f"     FP16: {m.get('fp16_fps')} fps | "
                f"Loss↓: {m.get('loss_reduction_pct')}% | "
                f"Compress: {m.get('compression_ratio')}x | "
                f"Mem: {m.get('gpu_mem_gb')} GB"
            )
            if rec.get("recommendation"):
                console.print(f"     → {rec['recommendation']}")
            console.print()

        if recs[0].get("training_insight"):
            ti = recs[0]["training_insight"]
            console.print(f"  [cyan]Training Insight ({ti['source']}):[/cyan] {ti['note']}")
            console.print()


@hyperparam_app.command("auto")
def hyperparam_auto(
    trials: int = typer.Option(30, "--trials", help="Number of trials"),
    objective: str = typer.Option("balanced", help="Objective: balanced, speed, quality, size"),
    steps: int = typer.Option(100, "--steps", help="Training steps per trial"),
    seed: int = typer.Option(42, "--seed", help="Reproducible TPE and trial RNG seed"),
    device: str | None = typer.Option(None, help="Device: auto|cuda|cpu"),
    model_dir: str = typer.Option(None, "--model-dir", help="Model directory"),
    data_dir: Path | None = typer.Option(None, "--data-dir", help="Required real LeRobot dataset directory"),
    allow_mock: bool = typer.Option(False, "--allow-mock", help="Explicit test-only random tensor input"),
    output_dir: str = typer.Option("./outputs/auto_hp", "--output-dir", help="Output directory"),
    pruner_type: str = typer.Option("median", "--pruner", help="Pruner: median, hyperband"),
    storage: str = typer.Option(None, "--storage", help="SQLite URL for persistence"),
    export_yaml: str = typer.Option(None, "--export-yaml", help="Export best config to YAML"),
    show_best: bool = typer.Option(False, "--show-best", help="Show best result from previous run"),
    output_json: bool = typer.Option(False, "--json", help="JSON output"),
    wandb: bool = typer.Option(False, "--wandb", help="Log to Weights & Biases (additive to JSON/console)"),
    wandb_project: str = typer.Option("forge", "--wandb-project", help="W&B project name"),
    wandb_entity: str = typer.Option(None, "--wandb-entity", help="W&B entity/team"),
):
    """Run automated hyperparameter search with Optuna."""
    from forge.auto_hyperparam import get_search_summary, run_auto_search

    if show_best:
        summary = get_search_summary(output_dir)
        if not summary:
            emit_cli_error(
                "No previous search found. Run a search first.",
                output_json=output_json,
                exit_code=1,
            )
        if output_json:
            emit_json(summary)
        else:
            best = summary.get("best_trial", {})
            console.print("\n[bold cyan]FORGE Auto-HP Best Result[/bold cyan]")
            console.print(f"  Objective: {summary.get('objective')}")
            console.print(
                f"  Trials: {summary.get('completed')} completed, "
                f"{summary.get('pruned')} pruned, {summary.get('failed')} failed"
            )
            console.print(f"  GPU time saved: {summary.get('gpu_time_saved_pct', 0)}%")
            console.print(f"  Total time: {summary.get('total_time_s', 0):.0f}s")
            if best:
                console.print(f"\n  [bold green]Best Trial #{best.get('number')}[/bold green]")
                console.print(f"    Score: {best.get('score')}")
                console.print(f"    Params: {best.get('params')}")
                m = best.get("metrics", {})
                console.print(
                    f"    FPS: {m.get('fps')} | Loss↓: {m.get('loss_reduction_pct')}% | "
                    f"Compress: {m.get('compression_ratio')}x"
                )
        return

    captured_stdout = io.StringIO()
    try:
        device = resolve_runtime_device(device=device, command="hyperparam", default="auto", strict=True)
        if not output_json:
            console.print("[bold cyan]FORGE Auto-HP Search[/bold cyan]")
            console.print(f"  Objective: {objective}")
            console.print(f"  Trials: {trials}")
            console.print(f"  Steps/trial: {steps}")
            console.print(f"  Seed: {seed}")
            console.print(f"  Device: {device}")
            console.print(f"  Pruner: {pruner_type}")
            console.print()

        output_context = redirect_stdout(captured_stdout) if output_json else nullcontext()
        with output_context:
            result = run_auto_search(
                objective=objective,
                n_trials=trials,
                train_steps=steps,
                device=device,
                model_dir=model_dir,
                output_dir=output_dir,
                pruner=pruner_type,
                storage=storage,
                wandb_project=wandb_project if wandb else None,
                wandb_entity=wandb_entity if wandb else None,
                data_dir=data_dir,
                allow_mock=allow_mock,
                random_seed=seed,
            )

        if export_yaml and result.get("best_trial"):
            from forge.auto_hyperparam import create_forge_study, export_best_yaml

            study_storage = storage or f"sqlite:///{Path(output_dir) / 'study.db'}"
            export_context = redirect_stdout(io.StringIO()) if output_json else nullcontext()
            with export_context:
                study = create_forge_study(
                    objective=objective,
                    pruner=pruner_type,
                    storage=study_storage,
                    random_seed=seed,
                )
                export_best_yaml(study, export_yaml)
            if not output_json:
                console.print(f"\n[green]Best config exported to {export_yaml}[/green]")
    except Exception as exc:
        emit_cli_error(str(exc), output_json=output_json, exit_code=2)
    if output_json and captured_stdout.getvalue():
        typer.echo(captured_stdout.getvalue(), err=True, nl=False)

    if not output_json and wandb and result.get("wandb_project"):
        entity = wandb_entity or result.get("wandb_entity", "")
        wandb_url = f"https://wandb.ai/{entity}/{wandb_project}" if entity else f"https://wandb.ai/me/{wandb_project}"
        console.print(f"\n[bold cyan]W&B Dashboard:[/bold cyan] {wandb_url}")

    if output_json:
        emit_json(result)
    else:
        best = result.get("best_trial", {})
        console.print("\n[bold]Search Complete[/bold]")
        console.print(f"  Completed: {result.get('completed')}")
        console.print(f"  Pruned: {result.get('pruned')} ({result.get('gpu_time_saved_pct', 0)}% GPU time saved)")
        console.print(f"  Total time: {result.get('total_time_s', 0):.0f}s")
        if best:
            console.print(f"\n  [bold green]Best:[/bold green] Trial #{best.get('number')}, score={best.get('score')}")
            m = best.get("metrics", {})
            console.print(f"    {best.get('params')}")
            console.print(
                f"    FPS: {m.get('fps')} | Loss↓: {m.get('loss_reduction_pct')}% | "
                f"Compress: {m.get('compression_ratio')}x"
            )


@hyperparam_app.command("top")
def hyperparam_top(
    n: int = typer.Option(5, help="Number of top trials"),
    results_dir: str = typer.Option("./outputs/hyperparam", help="Results directory"),
    output_json: bool = typer.Option(False, "--json", help="JSON output"),
):
    """Show top N trials by objective value."""
    from forge.hyperparam import HyperparamSearch, SearchSpace

    search = HyperparamSearch(SearchSpace(), results_dir=results_dir)
    top = search.top_trials(n)

    if output_json:
        emit_json([t.to_dict() for t in top])
    else:
        if not top:
            console.print("[yellow]No completed trials[/yellow]")
            return
        table = Table(title=f"Top {n} Trials")
        table.add_column("ID", style="cyan", max_width=12)
        table.add_column("Objective", style="green")
        table.add_column("Params")
        table.add_column("Duration")
        for t in top:
            obj_str = f"{t.objective_value:.6f}" if t.objective_value is not None else "—"
            params_str = ", ".join(f"{k}={v}" for k, v in t.params.items())
            dur_str = f"{t.duration_seconds:.1f}s" if t.duration_seconds > 0 else "—"
            table.add_row(t.trial_id, obj_str, params_str, dur_str)
        console.print(table)
