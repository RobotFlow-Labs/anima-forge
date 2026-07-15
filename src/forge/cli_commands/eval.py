"""VLA evaluation harness commands."""

from __future__ import annotations

import io
from contextlib import nullcontext, redirect_stdout
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from forge.cli_commands.shared import emit_cli_error, emit_json, resolve_runtime_device

eval_app = typer.Typer(name="eval", help="VLA evaluation harness (PRD-32)")
console = Console()


def _require_checkpoint_file(checkpoint: str, *, output_json: bool) -> str:
    path = Path(checkpoint).expanduser()
    if not path.is_file():
        emit_cli_error(
            f"Checkpoint not found: {checkpoint}",
            output_json=output_json,
            exit_code=2,
        )
    return str(path.resolve())


def _verify_eval_serve_checkpoint(
    checkpoint: str,
    *,
    allow_mock: bool,
) -> bool:
    """Verify provenance before starting the blocking eval server."""
    from forge.checkpoint_compat import load_checkpoint_payload
    from forge.config import StudentConfig

    effective_allow_mock = bool(allow_mock or StudentConfig().allow_mock)
    try:
        payload = load_checkpoint_payload(
            checkpoint,
            map_location="cpu",
            verify_provenance_for="eval",
            allow_mock=effective_allow_mock,
        )
    except Exception as exc:
        emit_cli_error(str(exc), output_json=False, exit_code=2)
    if payload is None:
        emit_cli_error(
            f"Checkpoint payload is unreadable: {checkpoint}",
            output_json=False,
            exit_code=2,
        )
    return effective_allow_mock


def _raise_failed_result(result: object, *, output_json: bool) -> None:
    if not isinstance(result, dict) or result.get("status", "completed") == "completed":
        return
    emit_cli_error(
        str(result.get("error") or "Evaluation failed"),
        output_json=output_json,
        exit_code=2,
    )


def _resolve_checkpoint(
    checkpoint: str | None,
    variant: str = "nano",
    model_dir: str | None = None,
) -> str:
    """Resolve a valid checkpoint when one is not explicitly provided."""
    if checkpoint:
        if Path(checkpoint).is_file():
            return checkpoint
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    search_roots = [Path(model_dir) if model_dir else Path("./outputs")]
    search_roots.extend(
        [
            Path("./outputs"),
            Path("./outputs/checkpoints"),
            Path("./outputs/real_training/checkpoints"),
            Path(f"./outputs/{variant}/checkpoints"),
        ]
    )

    preferred = ["final.pt", "best.pt"]
    candidates: list[tuple[float, Path]] = []

    for root in search_roots:
        for name in preferred:
            path = root / name
            if path.exists():
                candidates.append((path.stat().st_mtime, path))

    if not candidates:
        raise FileNotFoundError(
            "No checkpoint provided and no checkpoint found under ./outputs/**/checkpoints/{final.pt,best.pt}"
        )

    return str(max(candidates, key=lambda item: item[0])[1])


@eval_app.command("serve")
def eval_serve(
    checkpoint: str = typer.Option(..., help="Path to student checkpoint"),
    variant: str = typer.Option("nano", help="Student variant"),
    model_dir: str = typer.Option(None, help="Model directory"),
    port: int = typer.Option(8000, help="WebSocket port"),
    device: str | None = typer.Option(None, help="Device: auto|cuda|cpu"),
    allow_mock: bool = typer.Option(
        False,
        "--allow-mock",
        help="Explicitly allow a checkpoint whose provenance contains mock inputs",
    ),
):
    """Start FORGE model server for vla-eval benchmarks."""
    checkpoint = _require_checkpoint_file(checkpoint, output_json=False)
    allow_mock = _verify_eval_serve_checkpoint(checkpoint, allow_mock=allow_mock)
    try:
        device = resolve_runtime_device(device=device, command="eval", default="auto", strict=True)
        from forge.eval.model_server import ForgeModelServer

        server = ForgeModelServer(
            checkpoint_path=checkpoint,
            variant=variant,
            model_dir=model_dir,
            device=device,
            port=port,
            allow_mock=allow_mock,
        )
    except Exception as exc:
        emit_cli_error(str(exc), output_json=False, exit_code=2)

    typer.echo("Starting FORGE model server", err=True)
    typer.echo(f"  Checkpoint: {checkpoint}", err=True)
    typer.echo(f"  Variant: {variant}", err=True)
    typer.echo(f"  Port: {port}", err=True)
    try:
        server.start(blocking=True)
    except Exception as exc:
        emit_cli_error(str(exc), output_json=False, exit_code=2)


@eval_app.command("run")
def eval_run(
    benchmark: str = typer.Argument(help="Benchmark name: libero, simpler, vlabench"),
    checkpoint: str = typer.Option(..., help="Path to student checkpoint"),
    variant: str = typer.Option("nano", help="Student variant"),
    model_dir: str = typer.Option(None, help="Model directory"),
    config: str = typer.Option(None, help="Custom benchmark config YAML"),
    episodes: int = typer.Option(20, help="Episodes per task"),
    max_tasks: int = typer.Option(10, help="Max tasks to evaluate"),
    device: str | None = typer.Option(None, help="Device: auto|cuda|cpu"),
    output_dir: str = typer.Option("./outputs/eval", help="Output directory"),
    output_json: bool = typer.Option(False, "--json", help="JSON output"),
    allow_mock: bool = typer.Option(False, "--allow-mock", help="Explicitly allow mock provenance"),
):
    """Run a single benchmark evaluation."""
    checkpoint = _require_checkpoint_file(checkpoint, output_json=output_json)
    captured_stdout = io.StringIO()
    output_context = redirect_stdout(captured_stdout) if output_json else nullcontext()
    try:
        device = resolve_runtime_device(device=device, command="eval", default="auto", strict=True)
        with output_context:
            from forge.eval.runner import EvalRunner

            runner = EvalRunner(
                checkpoint_path=checkpoint,
                variant=variant,
                model_dir=model_dir,
                device=device,
                output_dir=output_dir,
                allow_mock=allow_mock,
            )
            result = runner.run_benchmark(
                benchmark=benchmark,
                config_path=config,
                episodes_per_task=episodes,
                max_tasks=max_tasks,
            )
    except Exception as exc:
        emit_cli_error(str(exc), output_json=output_json, exit_code=2)
    if output_json and captured_stdout.getvalue():
        typer.echo(captured_stdout.getvalue(), err=True, nl=False)

    _raise_failed_result(result, output_json=output_json)
    if output_json:
        emit_json(result)
    else:
        table = Table(title=f"Eval Result: {benchmark}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Benchmark", benchmark)
        table.add_row("Status", result.get("status", "completed"))
        success = result.get("success_rate", 0.0)
        table.add_row("Success Rate", f"{success:.1%}" if isinstance(success, float) else str(success))
        table.add_row("Tasks", str(result.get("tasks", "N/A")))
        table.add_row("Episodes/Task", str(result.get("episodes_per_task", episodes)))
        if result.get("error"):
            table.add_row("Error", str(result["error"])[:100])
        console.print(table)


@eval_app.command("run-all")
def eval_run_all(
    benchmarks: str = typer.Option(
        None,
        "--benchmarks",
        help="Optional comma-separated benchmark list (libero,simpler,vlabench).",
    ),
    checkpoint: str = typer.Option(..., help="Path to student checkpoint"),
    variant: str = typer.Option("nano", help="Student variant"),
    model_dir: str = typer.Option(None, help="Model directory"),
    episodes: int = typer.Option(20, help="Episodes per task"),
    max_tasks: int = typer.Option(10, help="Maximum number of tasks to evaluate"),
    device: str | None = typer.Option(None, help="Device: auto|cuda|cpu"),
    output_dir: str = typer.Option("./outputs/eval", help="Output directory"),
    output_json: bool = typer.Option(False, "--json", help="JSON output"),
    allow_mock: bool = typer.Option(False, "--allow-mock", help="Explicitly allow mock provenance"),
):
    """Run all 3 benchmarks (LIBERO, SimplerEnv, VLABench)."""
    checkpoint = _require_checkpoint_file(checkpoint, output_json=output_json)
    captured_stdout = io.StringIO()
    output_context = redirect_stdout(captured_stdout) if output_json else nullcontext()
    try:
        device = resolve_runtime_device(device=device, command="eval", default="auto", strict=True)
        bench_list = (
            [item.strip() for item in benchmarks.split(",") if item.strip()] if benchmarks is not None else None
        )
        with output_context:
            from forge.eval.runner import EvalRunner

            runner = EvalRunner(
                checkpoint_path=checkpoint,
                variant=variant,
                model_dir=model_dir,
                device=device,
                output_dir=output_dir,
                allow_mock=allow_mock,
            )
            results = runner.run_all(
                benchmarks=bench_list,
                episodes_per_task=episodes,
                max_tasks=max_tasks,
            )
    except Exception as exc:
        emit_cli_error(str(exc), output_json=output_json, exit_code=2)
    if output_json and captured_stdout.getvalue():
        typer.echo(captured_stdout.getvalue(), err=True, nl=False)

    failed = next((item for item in results if item.get("status", "completed") != "completed"), None)
    _raise_failed_result(failed, output_json=output_json)
    if output_json:
        emit_json(results)
    else:
        table = Table(title="VLA Evaluation Results")
        table.add_column("Benchmark", style="cyan")
        table.add_column("Success Rate", style="green")
        table.add_column("Status")
        for item in results:
            success = item.get("success_rate", 0.0)
            table.add_row(
                item.get("benchmark", "?"),
                f"{success:.1%}" if isinstance(success, float) else "N/A",
                item.get("status", "completed"),
            )
        console.print(table)


@eval_app.command("compare")
def eval_compare(
    checkpoint_a: str = typer.Option(..., "--a", help="First checkpoint"),
    checkpoint_b: str = typer.Option(..., "--b", help="Second checkpoint"),
    benchmark: str = typer.Option("libero", help="Benchmark to compare on"),
    variant: str = typer.Option("nano", help="Student variant"),
    device: str | None = typer.Option(None, help="Device: auto|cuda|cpu"),
    output_json: bool = typer.Option(False, "--json", help="JSON output"),
    allow_mock: bool = typer.Option(False, "--allow-mock", help="Explicitly allow mock provenance"),
):
    """Compare two student checkpoints on a benchmark."""
    checkpoint_a = _require_checkpoint_file(checkpoint_a, output_json=output_json)
    checkpoint_b = _require_checkpoint_file(checkpoint_b, output_json=output_json)
    captured_stdout = io.StringIO()
    output_context = redirect_stdout(captured_stdout) if output_json else nullcontext()
    try:
        device = resolve_runtime_device(device=device, command="eval", default="auto", strict=True)
        with output_context:
            from forge.eval.runner import EvalRunner

            runner = EvalRunner(
                checkpoint_path=checkpoint_a,
                variant=variant,
                device=device,
                allow_mock=allow_mock,
            )
            result = runner.compare(checkpoint_b, benchmark=benchmark)
    except Exception as exc:
        emit_cli_error(str(exc), output_json=output_json, exit_code=2)
    if output_json and captured_stdout.getvalue():
        typer.echo(captured_stdout.getvalue(), err=True, nl=False)

    _raise_failed_result(result, output_json=output_json)
    if output_json:
        emit_json(result)
    else:
        table = Table(title=f"Comparison: {benchmark}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Checkpoint A", checkpoint_a)
        table.add_row("Checkpoint B", checkpoint_b)
        table.add_row("Success Rate A", f"{result['success_rate_a']:.1%}")
        table.add_row("Success Rate B", f"{result['success_rate_b']:.1%}")
        delta = result["delta_success_rate"]
        color = "green" if delta > 0 else "red" if delta < 0 else "yellow"
        table.add_row("Delta", f"[{color}]{delta:+.1%}[/{color}]")
        console.print(table)


@eval_app.command("results")
def eval_results(
    output_dir: str = typer.Option("./outputs/eval", help="Eval output directory"),
    output_json: bool = typer.Option(False, "--json", help="JSON output"),
):
    """Show evaluation results."""
    from forge.eval.results import load_results, results_to_table

    results = load_results(output_dir)

    if output_json:
        emit_json([result.to_dict() for result in results])
    elif results:
        # Persisted container errors can contain Rich-style ``[tags]``. Render
        # this Markdown table literally so untrusted diagnostics cannot break
        # the human results command or disappear as console markup.
        console.print(results_to_table(results), markup=False, soft_wrap=True)
    else:
        console.print("[yellow]No evaluation results found.[/yellow]")
        console.print("Run: forge eval run libero --checkpoint <path>")


@eval_app.command("smoke")
def eval_smoke(
    checkpoint: str = typer.Option(None, help="Path to student checkpoint"),
    variant: str = typer.Option("nano", help="Student variant"),
    model_dir: str = typer.Option(None, help="Model directory"),
    device: str | None = typer.Option(None, help="Device: auto|cuda|cpu"),
    port: int = typer.Option(8000, help="WebSocket port"),
    output_json: bool = typer.Option(False, "--json", help="JSON output"),
    allow_mock: bool = typer.Option(False, "--allow-mock", help="Explicitly allow mock provenance"),
):
    """Quick smoke test: 1 task, 1 episode on LIBERO."""
    captured_stdout = io.StringIO()
    output_context = redirect_stdout(captured_stdout) if output_json else nullcontext()
    try:
        device = resolve_runtime_device(device=device, command="eval", default="auto", strict=True)
        checkpoint = _resolve_checkpoint(checkpoint, variant=variant, model_dir=model_dir)
        if not output_json:
            console.print("[bold]Running LIBERO smoke test (1 task, 1 episode)...[/bold]")
        with output_context:
            from forge.eval.runner import EvalRunner

            runner = EvalRunner(
                checkpoint_path=checkpoint,
                variant=variant,
                model_dir=model_dir,
                device=device,
                output_dir="./outputs/eval/smoke",
                port=port,
                allow_mock=allow_mock,
            )
            result = runner.run_benchmark(
                benchmark="libero",
                episodes_per_task=1,
                max_tasks=1,
            )
    except Exception as exc:
        emit_cli_error(str(exc), output_json=output_json, exit_code=2)
    if output_json and captured_stdout.getvalue():
        typer.echo(captured_stdout.getvalue(), err=True, nl=False)

    _raise_failed_result(result, output_json=output_json)
    if output_json:
        emit_json(result)
    else:
        status = result.get("status", "completed")
        success = result.get("success_rate", 0.0)
        icon = "[green]PASS[/green]" if status != "failed" else "[red]FAIL[/red]"
        console.print(f"  Status: {icon}")
        console.print(f"  Success Rate: {success:.0%}" if isinstance(success, float) else f"  Success Rate: {success}")
        if result.get("error"):
            console.print(f"  Error: {result['error'][:200]}")


@eval_app.command("setup")
def eval_setup(
    skip_pull: bool = typer.Option(
        False,
        "--skip-pull",
        help="Skip Docker image download and only validate executable CLI path",
    ),
):
    """Pull the Docker benchmark images."""
    from forge.eval.runner import EvalRunner

    has_docker = EvalRunner.check_docker()
    if not has_docker and not skip_pull:
        console.print("[red]Docker not found. Install Docker first.[/red]")
        raise typer.Exit(1)
    if not has_docker and skip_pull:
        console.print("[yellow]Docker not found; skipping image pulls by request.[/yellow]")
        console.print("[yellow]Set up Docker before running benchmark steps.[/yellow]")
        return

    console.print("[bold]Pulling VLA evaluation Docker images...[/bold]")
    if skip_pull:
        console.print("[yellow]Skipping Docker image pull by request.[/yellow]")
        return

    console.print("Download time and size depend on the current upstream images.")
    results = EvalRunner.pull_images()
    for name, status in results.items():
        icon = "[green]OK[/green]" if status == "ok" else f"[red]{status}[/red]"
        console.print(f"  {name}: {icon}")
