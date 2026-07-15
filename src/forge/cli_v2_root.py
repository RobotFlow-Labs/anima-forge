"""Root-level FORGE CLI commands for v2 entrypoint."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from forge.cli_commands import (
    benchmark_app,
    config_app,
    curriculum_app,
    demo_app,
    doctor_command,
    embodiment_app,
    eval_app,
    finetune_app,
    hyperparam_app,
    metrics_app,
    models_app,
    profile_app,
    quantize_app,
    quickstart_command,
    status_command,
    students_app,
    teacher_app,
    telemetry_app,
    top,
    top_agent,
    train_app,
    transfer_app,
)
from forge.cli_commands.logging_config import setup_cli_logging
from forge.cli_commands.shared import (
    DEFAULT_NANO_CONFIG,
    emit_cli_error,
    emit_json,
    load_forge_config,
    resolve_runtime_device,
)

console = Console()
_LOGGER = logging.getLogger("forge")
_DEFAULT_NANO_CONFIG = DEFAULT_NANO_CONFIG


def _load_cli_config(config_path: str):
    """Load an explicit config, resolving the CLI default from package data."""
    return load_forge_config(config_path, required=True)


def _version_callback(value: bool) -> None:
    if not value:
        return
    from forge import __version__

    typer.echo(__version__)
    raise typer.Exit


def setup_cli_logging_callback(
    ctx: typer.Context,
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the FORGE version and exit",
    ),
    log_file: str | None = typer.Option(
        None,
        "--log-file",
        envvar="FORGE_CLI_LOG_FILE",
        help="Rotate logs to this file (1MB max, 2 backups)",
    ),
    log_level: str = typer.Option("INFO", "--log-level", help="Log level"),
    log_to_stdout: bool = typer.Option(False, "--log-to-stdout", help="Also print logs to stdout"),
    log_json: bool = typer.Option(
        False,
        "--log-json/--no-log-json",
        envvar="FORGE_CLI_LOG_JSON",
        help="Format logs as JSON for automation",
    ),
) -> None:
    """Shared callback used by the root app."""
    del version
    if ctx.resilient_parsing:
        return
    setup_cli_logging(
        log_file=log_file,
        log_level=log_level,
        log_to_stdout=log_to_stdout,
        log_json=log_json,
    )
    if ctx.invoked_subcommand is None:
        console.print("[bold cyan]FORGE v3[/bold cyan] — distill VLA models for edge deployment")
        console.print("  forge quickstart --yes   First real distillation")
        console.print("  forge doctor             Check models, GPU, data, and disk")
        console.print("  forge pipeline --help    Run individual or complete stages")
        console.print("  forge models fetch --all-students   Install student backbones")
        console.print("  forge config init > forge.yaml      Create a starter config")
        console.print("  Docs: https://github.com/RobotFlow-Labs/anima-forge-distillation-pipeline")
        _print_completion_hint_once()


def _print_completion_hint_once() -> None:
    """Show shell completion setup once per user profile."""
    config_home = Path(os.environ.get("FORGE_CONFIG_HOME", Path.home() / ".config" / "forge")).expanduser()
    marker = config_home / "completion-hint-shown"
    if marker.exists():
        return
    try:
        config_home.mkdir(parents=True, exist_ok=True)
        marker.write_text("shown\n", encoding="utf-8")
    except OSError:
        return
    console.print("  Tip: run `forge --show-completion` to install shell completion.")


def info(
    config: str = typer.Option("configs/forge_nano.yaml", help="Config file path"),
    output_json: bool = typer.Option(False, "--json", help="Emit strict JSON"),
) -> None:
    """Show system info: backend, device, model paths."""
    try:
        from forge import __version__

        cfg = _load_cli_config(config)
        from forge.backend import detect_backend, get_backend

        backend_type = detect_backend()
        backend = get_backend()
        device_info = backend.get_device_info()
        from forge.cli_commands._doctor_core import _validate_model
        from forge.model_assets import CORE_MODEL_ASSETS

        model_dir = Path(cfg.paths.model_dir).expanduser()
        quickstart_assets = tuple(
            asset for asset in CORE_MODEL_ASSETS if asset.role.startswith("student:") or asset.role == "vision"
        )
        present = sum(_validate_model(asset, model_dir).status == "ok" for asset in quickstart_assets)
    except Exception as exc:
        emit_cli_error(str(exc), output_json=output_json, exit_code=2)

    if output_json:
        emit_json(
            {
                "version": __version__,
                "backend": backend_type.value,
                "device": device_info.device_name,
                "vram_gb": device_info.vram_gb,
                "compute_capability": device_info.compute_capability,
                "model_dir": str(model_dir),
                "teacher": str(cfg.paths.teacher_path),
                "vision_encoder": str(cfg.paths.vision_encoder_path),
                "language_model": str(cfg.paths.language_model_path),
                "model_readiness": {
                    "present": present,
                    "required": len(quickstart_assets),
                },
            }
        )
        return

    table = Table(title="FORGE System Info")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Backend", str(backend_type.value))
    table.add_row("Device", device_info.device_name)
    table.add_row("VRAM / RAM", f"{device_info.vram_gb:.1f} GB")
    if device_info.compute_capability:
        table.add_row("Compute Capability", device_info.compute_capability)
    table.add_row("Model Dir", cfg.paths.model_dir)
    table.add_row("Teacher", str(cfg.paths.teacher_path))
    table.add_row("Vision Encoder", str(cfg.paths.vision_encoder_path))
    table.add_row("Language Model", str(cfg.paths.language_model_path))
    table.add_row(
        "Model Readiness",
        f"{present}/{len(quickstart_assets)} v3 student/vision assets present — run `forge doctor`",
    )
    console.print(table)


def pipeline(
    config: str = typer.Option("configs/forge_nano.yaml", help="Config file path"),
    stage: str | None = typer.Option(None, help="Single stage: labels|distill|compress|export|validate"),
    skip_labels: bool = typer.Option(False, help="Skip teacher label generation"),
    device: str | None = typer.Option(None, help="Device: auto|cuda|cpu|mps"),
    max_steps: int | None = typer.Option(None, help="Override max distillation steps"),
    batch_size: int | None = typer.Option(
        None,
        "--batch-size",
        min=1,
        help="Override distillation batch size",
    ),
    gradient_accumulation_steps: int | None = typer.Option(
        None,
        "--gradient-accumulation-steps",
        min=1,
        help="Override distillation gradient accumulation",
    ),
    max_episodes: int | None = typer.Option(
        None,
        "--max-episodes",
        min=1,
        help="Limit teacher-label episodes for bounded real-data runs",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        help="Override paths.output_dir from the config",
    ),
    data_dir: Path | None = typer.Option(
        None,
        "--data-dir",
        help="Override paths.data_dir from the config",
    ),
    checkpoint: Path | None = typer.Option(
        None,
        "--checkpoint",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Trained checkpoint to use for compress, export, or validate",
    ),
    allow_mock: bool = typer.Option(
        False,
        "--allow-mock",
        help="Explicitly permit mock inputs (all artifacts are stamped MOCK)",
    ),
) -> None:
    """Run the FORGE distillation pipeline end-to-end."""
    from forge.pipeline import run_pipeline

    try:
        device = resolve_runtime_device(device=device, command="pipeline", default="auto", strict=True)
        cfg = _load_cli_config(config)
    except Exception as exc:
        emit_cli_error(str(exc), output_json=False, exit_code=2)
    if output_dir is not None:
        cfg.paths.output_dir = str(output_dir.expanduser().resolve())
    if data_dir is not None:
        cfg.paths.data_dir = str(data_dir.expanduser().resolve())
    if batch_size is not None:
        cfg.distill.batch_size = batch_size
    if gradient_accumulation_steps is not None:
        cfg.distill.gradient_accumulation_steps = gradient_accumulation_steps
    cfg.student.allow_mock = bool(cfg.student.allow_mock or allow_mock)

    console.print(f"[bold cyan]FORGE Pipeline — {cfg.student.variant}[/bold cyan]")
    console.print(f"Output: {Path(cfg.paths.output_dir).expanduser().resolve()}")

    def show_progress(event: dict[str, object]) -> None:
        stage_name = str(event["stage"])
        if event["status"] == "started":
            title = str(event.get("title", stage_name.replace("_", " ").title()))
            console.rule(f"[bold cyan]{title}[/bold cyan]")
            return
        if event["status"] == "progress":
            step = int(str(event.get("step", 0)))
            total_steps = int(str(event.get("total_steps", 0)))
            loss = float(str(event.get("loss", 0.0)))
            eta = float(str(event.get("eta_seconds", 0.0)))
            speed = float(str(event.get("steps_per_second", 0.0)))
            vram = event.get("vram_gib")
            vram_text = f" · VRAM {float(str(vram)):.2f} GiB" if vram is not None else ""
            console.print(
                f"  step {step}/{total_steps} · loss {loss:.4f} · {speed:.2f} steps/s · ETA {eta:.0f}s{vram_text}"
            )
            return
        elapsed = float(str(event.get("elapsed_seconds", 0.0)))
        if event["status"] == "failed":
            console.print(f"[red]FAILED[/red] {stage_name} ({elapsed:.2f}s)")
        else:
            console.print(f"[green]DONE[/green] {stage_name} ({elapsed:.2f}s)")

    results = run_pipeline(
        cfg,
        device=device,
        skip_labels=skip_labels,
        stage=stage,
        checkpoint_path=checkpoint,
        max_label_episodes=max_episodes,
        max_distill_steps=max_steps,
        progress_callback=show_progress,
    )

    table = Table(title="Pipeline Results")
    table.add_column("Stage", style="cyan")
    table.add_column("Status", style="green")
    for key, value in results.items():
        if isinstance(value, dict) and "status" in value:
            status = "[green]OK[/green]" if value["status"] != "failed" else "[red]FAILED[/red]"
            table.add_row(key, status)
        elif key == "total_time_seconds":
            table.add_row("Total Time", f"{value:.1f}s")
    table.add_row("Overall", str(results.get("status", "unknown")))
    console.print(table)
    summary_path = Path(str(results["pipeline_summary_path"]))
    if summary_path.is_file():
        console.print(f"Pipeline summary: {summary_path}")

    if results.get("status") == "failed":
        failure = next(
            (str(value["error"]) for value in results.values() if isinstance(value, dict) and value.get("error")),
            "Pipeline failed",
        )
        typer.echo(f"Error: {failure}", err=True)
        raise typer.Exit(2)


def report(
    results_file: Path = typer.Option(
        ...,
        "--results-file",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Measured JSON results artifact",
    ),
    output: Path = typer.Option(Path("./outputs/FORGE_REPORT.md"), help="Output markdown path"),
) -> None:
    """Generate a truthful Markdown report from a measured JSON artifact."""
    from forge.report import generate_report

    try:
        md = generate_report(results_path=str(results_file), output_path=str(output))
    except (OSError, ValueError, TypeError) as exc:
        raise typer.BadParameter(f"Could not read measured results artifact: {exc}") from exc

    console.print(f"[green]Report generated: {output}[/green]")
    console.print(md[:500] + "...")


def serve(
    host: str = typer.Option("0.0.0.0", help="Server host"),
    port: int = typer.Option(8000, help="Server port"),
    device: str | None = typer.Option(None, help="Device: auto|cuda|cpu"),
    checkpoint: str = typer.Option(..., "--checkpoint", help="Verified trained-model checkpoint"),
    allow_mock: bool = typer.Option(
        False,
        "--allow-mock",
        help="Explicitly allow a checkpoint whose provenance contains mock inputs",
    ),
) -> None:
    """Start FastAPI inference endpoint."""
    from forge.serve import start_server

    try:
        device = resolve_runtime_device(device=device, command="serve", default="auto", strict=True)
        model_dir = _load_cli_config(_DEFAULT_NANO_CONFIG).paths.model_dir
        console.print(f"[bold cyan]FORGE Serve — starting on {host}:{port}[/bold cyan]")
        console.print(f"  Device: {device}")
        console.print(f"  Model Dir: {model_dir}")
        start_server(
            host=host,
            port=port,
            model_dir=model_dir,
            checkpoint=checkpoint,
            device=device,
            allow_mock=allow_mock,
        )
    except Exception as exc:
        emit_cli_error(str(exc), output_json=False, exit_code=2)


def web_command(
    port: int = typer.Option(3000, help="Port number"),
    host: str = typer.Option("0.0.0.0", help="Host address"),
    no_browser: bool = typer.Option(False, help="Don't open browser"),
) -> None:
    """Launch FORGE Command Center web UI."""
    import threading
    import webbrowser

    import uvicorn

    from forge.config import ForgeConfig
    from forge.web.api import create_app

    config = ForgeConfig.default()
    config.web.port = port
    config.web.host = host
    app_ = create_app(config)

    if not no_browser:

        def _open() -> None:
            time.sleep(1.5)
            webbrowser.open(f"http://localhost:{port}")

        threading.Thread(target=_open, daemon=True).start()

    console.print(f"[bold cyan]FORGE Command Center[/bold cyan] → http://{host}:{port}")
    uvicorn.run(app_, host=host, port=port, log_level="info")


def autosense_command(
    model_dir: str | None = typer.Option(
        None,
        "--model-dir",
        help="Model directory (defaults to FORGE_MODEL_DIR)",
    ),
    vision: str | None = typer.Option(None, "--vision", help="Vision encoder dir name"),
    lm: str | None = typer.Option(None, "--lm", help="Language model dir name"),
    output_json: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Auto-detect model dimensions from config.json files."""
    import os

    from forge.autosense import autosense_config, sense_language_model, sense_vision_encoder
    from forge.config import ForgeConfig

    config = ForgeConfig.default()
    model_dir_path = Path(model_dir or os.environ.get("FORGE_MODEL_DIR", config.paths.model_dir))

    if not model_dir_path.exists():
        emit_cli_error(
            f"Model dir not found: {model_dir_path}",
            output_json=output_json,
            exit_code=2,
        )

    vision_name = vision or config.paths.vision_encoder
    lm_name = lm or config.paths.language_model

    result: dict[str, Any] = {"model_dir": str(model_dir_path)}

    vision_path = model_dir_path / vision_name
    vision_info = sense_vision_encoder(vision_path)
    if vision_info:
        result["vision"] = {"name": vision_name, **vision_info}

    lm_path = model_dir_path / lm_name
    lm_info = sense_language_model(lm_path)
    if lm_info:
        result["language"] = {"name": lm_name, **lm_info}

    overrides = autosense_config(model_dir_path, vision_name, lm_name)
    result["overrides"] = overrides

    if output_json:
        emit_json(result)
        return

    console.print("[bold cyan]AutoSense — Model Config Detection[/bold cyan]")
    console.print(f"  Model dir: {model_dir_path}")
    console.print()

    if "vision" in result:
        v = result["vision"]
        assert isinstance(v, dict)
        console.print(f"  [green]Vision:[/green] {v['name']}")
        console.print(f"    d_output: {v.get('d_output', '?')}")
        if "n_tokens" in v:
            console.print(f"    n_tokens: {v['n_tokens']}")
        if "patch_size" in v:
            console.print(f"    patch_size: {v['patch_size']}")
        if "image_size" in v:
            console.print(f"    image_size: {v['image_size']}")
    else:
        console.print(f"  [yellow]Vision:[/yellow] {vision_name} — no config.json found")

    console.print()
    if "language" in result:
        lm = result["language"]
        assert isinstance(lm, dict)
        console.print(f"  [green]Language:[/green] {lm['name']}")
        console.print(f"    d_model: {lm.get('d_model', '?')}")
        if "vocab_size" in lm:
            console.print(f"    vocab_size: {lm['vocab_size']}")
        if "n_layers" in lm:
            console.print(f"    n_layers: {lm['n_layers']}")
        if "n_heads" in lm:
            console.print(f"    n_heads: {lm['n_heads']}")
    else:
        console.print(f"  [yellow]Language:[/yellow] {lm_name} — no config.json found")

    if overrides:
        console.print()
        parts = [f"{k}={v}" for k, v in overrides.items()]
        console.print(f"  [bold]Config overrides:[/bold] {', '.join(parts)}")


def register_v2_commands(app_: typer.Typer) -> None:
    """Register all v2 command groups and root commands."""
    app_.add_typer(teacher_app)
    app_.add_typer(config_app)
    app_.add_typer(students_app)
    app_.add_typer(benchmark_app)
    app_.add_typer(embodiment_app)
    app_.add_typer(embodiment_app, name="embodyments")
    app_.add_typer(embodiment_app, name="embodiments")
    app_.add_typer(demo_app)
    app_.add_typer(quantize_app)
    app_.add_typer(curriculum_app)
    app_.add_typer(profile_app)
    app_.add_typer(train_app)
    app_.add_typer(metrics_app)
    app_.add_typer(models_app)
    app_.add_typer(hyperparam_app)
    app_.add_typer(finetune_app)
    app_.add_typer(telemetry_app)
    app_.add_typer(transfer_app)
    app_.add_typer(eval_app)
    app_.command("top")(top)
    app_.command("agent")(top_agent)
    app_.command("agent-top")(top_agent)
    app_.command("top-agent")(top_agent)
    app_.command("report")(report)
    app_.command("doctor")(doctor_command)
    app_.command("quickstart")(quickstart_command)
    app_.command("web")(web_command)
    app_.command("status")(status_command)
    app_.command("autosense")(autosense_command)
    app_.command("info")(info)
    app_.command("pipeline")(pipeline)
    app_.command("serve")(serve)
