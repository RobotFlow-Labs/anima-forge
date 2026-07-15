"""Demo generation commands."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from typer import Context

from forge.cli_commands.shared import resolve_runtime_device

console = Console()
demo_app = typer.Typer(name="demo", help="VC demo & report generation")


def _execute_demo(
    *,
    config_path: Path,
    checkpoint: Path | None,
    data_dir: Path | None,
    instruction: str | None,
    device: str | None,
    output: Path,
    samples: int,
    duration: float,
    allow_mock: bool,
) -> None:
    from forge.cli_commands.benchmark import _load_real_benchmark_input
    from forge.cli_commands.quantize import load_student_for_quant
    from forge.demo.runner import DemoRunner

    resolved_device = resolve_runtime_device(device=device, command="demo", default="auto", strict=True)
    config, model, provenance = load_student_for_quant(
        str(config_path),
        checkpoint=str(checkpoint) if checkpoint is not None else None,
        allow_mock=allow_mock,
        require_trained_checkpoint=True,
        protected_action="demo",
    )
    resolved_data_dir = (
        data_dir.expanduser()
        if data_dir is not None
        else Path(config.paths.model_dir).expanduser() / "datasets" / "lerobot--pusht"
    )
    images, language_text, input_provenance = _load_real_benchmark_input(
        resolved_data_dir,
        instruction=instruction,
    )
    runner = DemoRunner(
        config,
        device=resolved_device,
        model=model,
        provenance=provenance,
        source_checkpoint=str(checkpoint.expanduser().resolve()) if checkpoint is not None else None,
    )
    runner.run(
        output_path=str(output),
        images=images,
        language_text=language_text,
        input_provenance=input_provenance,
        samples=samples,
        duration=duration,
    )
    console.print(f"[green]Report generated: {output}[/green]")


@demo_app.callback(invoke_without_command=True)
def demo_entrypoint(
    ctx: Context,
    config: Path = typer.Option(Path("configs/forge_nano.yaml"), help="Student config"),
    checkpoint: Path | None = typer.Option(None, "--checkpoint", help="Trained checkpoint"),
    data_dir: Path | None = typer.Option(None, "--data-dir", help="Real LeRobot dataset"),
    instruction: str | None = typer.Option(None, "--instruction", help="Real task instruction"),
    device: str = typer.Option(None, help="Device (auto|cuda|cpu)"),
    output: Path = typer.Option(Path("forge_v3_report.html"), help="Output HTML path"),
    samples: int = typer.Option(30, "--samples", min=1, help="Latency samples"),
    duration: float = typer.Option(2.0, "--duration", min=0.01, help="Throughput seconds"),
    allow_mock: bool = typer.Option(
        False,
        "--allow-mock",
        help="Explicitly permit mock backbones (report is stamped MOCK)",
    ),
) -> None:
    """Compatibility root entrypoint for `forge demo --device ...`."""
    if ctx.invoked_subcommand:
        return

    try:
        _execute_demo(
            config_path=config,
            checkpoint=checkpoint,
            data_dir=data_dir,
            instruction=instruction,
            device=device,
            output=output,
            samples=samples,
            duration=duration,
            allow_mock=allow_mock,
        )
    except Exception as exc:
        raise typer.BadParameter(str(exc)) from exc


@demo_app.command("run")
def demo_run(
    config: Path = typer.Option(Path("configs/forge_nano.yaml"), help="Student config"),
    checkpoint: Path | None = typer.Option(None, "--checkpoint", help="Trained checkpoint"),
    data_dir: Path | None = typer.Option(None, "--data-dir", help="Real LeRobot dataset"),
    instruction: str | None = typer.Option(None, "--instruction", help="Real task instruction"),
    device: str = typer.Option(None, help="Device (auto|cuda|cpu)"),
    output: Path = typer.Option(Path("forge_v3_report.html"), help="Output HTML path"),
    samples: int = typer.Option(30, "--samples", min=1, help="Latency samples"),
    duration: float = typer.Option(2.0, "--duration", min=0.01, help="Throughput seconds"),
    allow_mock: bool = typer.Option(
        False,
        "--allow-mock",
        help="Explicitly permit mock backbones (report is stamped MOCK)",
    ),
):
    """Generate VC-ready demo report."""
    try:
        _execute_demo(
            config_path=config,
            checkpoint=checkpoint,
            data_dir=data_dir,
            instruction=instruction,
            device=device,
            output=output,
            samples=samples,
            duration=duration,
            allow_mock=allow_mock,
        )
    except Exception as exc:
        raise typer.BadParameter(str(exc)) from exc
