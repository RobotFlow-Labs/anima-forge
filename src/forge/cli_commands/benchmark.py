"""Benchmarking commands for FORGE."""

from __future__ import annotations

import sys
from contextlib import nullcontext, redirect_stdout
from pathlib import Path

import typer
from rich.console import Console

from forge.cli_commands.shared import emit_cli_error, emit_json, resolve_runtime_device

console = Console()
benchmark_app = typer.Typer(name="benchmark", help="Benchmark suite")


def _load_real_benchmark_input(
    data_dir: Path,
    *,
    instruction: str | None,
) -> tuple[object, str, dict[str, object]]:
    """Load one genuine LeRobot frame and its dataset/user-supplied instruction."""
    import json

    from forge.data.lerobot_video_dataset import LeRobotVideoActionDataset

    dataset = LeRobotVideoActionDataset(data_dir, max_samples=1)
    resolved_instruction = instruction.strip() if instruction is not None else ""
    instruction_source = "cli"
    if not resolved_instruction:
        tasks_jsonl = data_dir / "meta" / "tasks.jsonl"
        if tasks_jsonl.is_file():
            for line in tasks_jsonl.read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                candidate = record.get("task", record.get("instruction", ""))
                if isinstance(candidate, str) and candidate.strip():
                    resolved_instruction = candidate.strip()
                    instruction_source = "dataset-metadata"
                    break
    if not resolved_instruction:
        raise ValueError(
            "Benchmark data has no language task metadata. Pass the real task instruction with --instruction; "
            "FORGE will not invent language input."
        )

    sample = dataset[0]
    provenance = dict(dataset.provenance)
    provenance["instruction_source"] = instruction_source
    return sample["image"].unsqueeze(0), resolved_instruction, provenance


@benchmark_app.command("list")
def benchmark_list(
    output_json: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """List every packaged real-world benchmark suite."""
    from forge.benchmark.suite_runner import suite_catalog

    catalog = suite_catalog()
    if output_json:
        emit_json(catalog)
        return
    for suite in catalog:
        console.print(f"{suite['number']}  {suite['slug']:<20} {suite['description']}")


@benchmark_app.command("suite")
def benchmark_suite(
    name: str = typer.Argument(..., help="Suite number or name from `forge benchmark list`"),
    results_dir: Path = typer.Option(Path("benchmarks"), help="JSON artifact directory"),
    model_dir: Path = typer.Option(Path("models"), help="Local model directory"),
    export_dir: Path = typer.Option(Path("outputs/export"), help="Export scratch directory"),
    data_dir: Path | None = typer.Option(None, help="Required real-data directory override"),
    device: str = typer.Option("auto", help="Device visibility (auto|cuda|cpu)"),
    output_json: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Run one packaged real-world benchmark suite."""
    from forge.benchmark.suite_runner import run_suite

    try:
        result = run_suite(
            name,
            results_dir=results_dir,
            model_dir=model_dir,
            export_dir=export_dir,
            data_dir=data_dir,
            device=device,
            progress=sys.stderr,
        )
    except (OSError, ValueError) as exc:
        emit_cli_error(str(exc), output_json=output_json, exit_code=2)
    if output_json:
        emit_json(result)
    else:
        console.print(
            f"Suite {result['number']} ({result['suite']}): {result['status']}"
            + (f" — {result['artifact']}" if result.get("artifact") else "")
        )
    if result["status"] != "completed":
        raise typer.Exit(2)


@benchmark_app.command("all")
def benchmark_all(
    results_dir: Path = typer.Option(Path("benchmarks"), help="JSON artifact directory"),
    model_dir: Path = typer.Option(Path("models"), help="Local model directory"),
    export_dir: Path = typer.Option(Path("outputs/export"), help="Export scratch directory"),
    data_dir: Path | None = typer.Option(None, help="Required real-data directory override"),
    device: str = typer.Option("auto", help="Device visibility (auto|cuda|cpu)"),
    output_json: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Run all packaged real-world benchmark suites."""
    from forge.benchmark.suite_runner import run_all_suites

    try:
        summary = run_all_suites(
            results_dir=results_dir,
            model_dir=model_dir,
            export_dir=export_dir,
            data_dir=data_dir,
            device=device,
            progress=sys.stderr,
        )
    except (OSError, ValueError) as exc:
        emit_cli_error(str(exc), output_json=output_json, exit_code=2)
    if output_json:
        emit_json(summary)
    else:
        console.print(
            f"Benchmark suites: {summary['status']} "
            f"({summary['completed']} completed, {summary['skipped']} skipped, "
            f"{summary['failed']} failed)"
        )
        console.print(f"[green]Summary saved to {summary['artifact']}[/green]")
    if summary["status"] != "completed":
        raise typer.Exit(2)


@benchmark_app.command("aggregate")
def benchmark_aggregate(
    results_dir: Path = typer.Option(Path("benchmarks"), help="Existing JSON artifact directory"),
    output_json: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Summarize and verify already completed packaged benchmark suite artifacts."""
    from forge.benchmark.suite_runner import summarize_existing_suites

    try:
        summary = summarize_existing_suites(results_dir=results_dir)
    except (OSError, ValueError) as exc:
        emit_cli_error(str(exc), output_json=output_json, exit_code=2)
    if output_json:
        emit_json(summary)
    else:
        console.print(
            f"Benchmark artifact aggregate: {summary['status']} "
            f"({summary['completed']} completed, {summary['skipped']} skipped, "
            f"{summary['failed']} failed)"
        )
        console.print(f"[green]Summary saved to {summary['artifact']}[/green]")
    if summary["status"] != "completed":
        raise typer.Exit(2)


@benchmark_app.command("matrix")
def benchmark_matrix(
    manifest: Path = typer.Argument(..., help="Artifact validation manifest JSON"),
    results_dir: Path = typer.Option(Path("benchmarks"), help="JSON artifact directory"),
    device: str = typer.Option("cuda", help="Device (cuda|cpu)"),
    samples: int = typer.Option(20, min=1, help="PyTorch latency samples"),
    duration: float = typer.Option(2.0, min=0.01, help="PyTorch throughput seconds"),
    onnx_warmup: int = typer.Option(5, min=0, help="ONNX/TensorRT warmup runs"),
    onnx_runs: int = typer.Option(50, min=1, help="ONNX/TensorRT timed runs"),
    output_json: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Run the artifact-backed checkpoint/export validation matrix."""
    from forge.benchmark.matrix import run_validation_matrix

    if not manifest.is_file():
        emit_cli_error(
            f"Validation manifest not found: {manifest}",
            output_json=output_json,
            exit_code=2,
        )
    try:
        resolved_device = resolve_runtime_device(
            device=device,
            command="benchmark matrix",
            default="cuda",
            strict=True,
        )
        summary = run_validation_matrix(
            manifest,
            results_dir=results_dir,
            device=resolved_device,
            samples=samples,
            duration=duration,
            onnx_warmup=onnx_warmup,
            onnx_runs=onnx_runs,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        emit_cli_error(str(exc), output_json=output_json, exit_code=2)
    if output_json:
        emit_json(summary)
    else:
        console.print(f"Validation matrix: {summary['status']} ({len(summary['variants'])} variants)")
        console.print(f"[green]Results saved to {results_dir.resolve()}[/green]")
    if summary["status"] != "completed":
        raise typer.Exit(2)


@benchmark_app.command("run")
def benchmark_run(
    config: str = typer.Option("configs/forge_nano.yaml", help="Config file path"),
    checkpoint: Path | None = typer.Option(
        None,
        "--checkpoint",
        help="Trained checkpoint to benchmark",
    ),
    device: str = typer.Option(None, help="Device (auto|cuda|cpu|mps)"),
    output: str = typer.Option(None, help="Output JSON path"),
    samples: int = typer.Option(100, "--samples", min=1, help="Latency samples"),
    duration: float = typer.Option(2.0, "--duration", min=0.01, help="Throughput seconds"),
    data_dir: Path | None = typer.Option(
        None,
        "--data-dir",
        help="Real LeRobot dataset (defaults to <model-dir>/datasets/lerobot--pusht)",
    ),
    instruction: str | None = typer.Option(
        None,
        "--instruction",
        help="Real task instruction when the dataset has no task text metadata",
    ),
    allow_mock: bool = typer.Option(
        False,
        "--allow-mock",
        help="Allow an untrained or mock-provenance model for explicit test workflows",
    ),
    output_json: bool = typer.Option(False, "--json", help="JSON output"),
):
    """Benchmark a trained FORGE checkpoint."""
    from forge.benchmark.execution import benchmark_execution
    from forge.benchmark.runner import BenchmarkRunner
    from forge.cli_commands.quantize import load_student_for_quant

    requested_device = device or "auto"
    output_context = redirect_stdout(sys.stderr) if output_json else nullcontext()
    try:
        device = resolve_runtime_device(device=device, command="benchmark", default="auto", strict=True)
        if checkpoint is not None and not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        with output_context:
            cfg, model, provenance = load_student_for_quant(
                config,
                checkpoint=str(checkpoint) if checkpoint is not None else None,
                allow_mock=allow_mock,
                require_trained_checkpoint=True,
                protected_action="benchmark",
            )
            resolved_data_dir = (
                data_dir.expanduser()
                if data_dir is not None
                else Path(cfg.paths.model_dir).expanduser() / "datasets" / "lerobot--pusht"
            )
            images, language_text, input_provenance = _load_real_benchmark_input(
                resolved_data_dir,
                instruction=instruction,
            )
            runner = BenchmarkRunner(model, cfg, device=device)
            report = runner.run(
                n_latency_samples=samples,
                throughput_duration=duration,
                images=images,
                language_text=language_text,
                input_provenance=input_provenance,
            )
    except Exception as exc:
        emit_cli_error(str(exc), output_json=output_json, exit_code=2)

    report.provenance = provenance
    report.execution = benchmark_execution(
        command="run",
        requested_device=requested_device,
        resolved_device=device,
    )
    if checkpoint is not None:
        resolved_checkpoint = checkpoint.expanduser().resolve()
        report.source_checkpoint = str(resolved_checkpoint)
        report.artifact_size_mb = resolved_checkpoint.stat().st_size / 1e6

    if output:
        runner.export(report, output)
        if not output_json:
            console.print(f"[green]Report saved to {output}[/green]")

    if output_json:
        emit_json(report.to_dict())
    else:
        runner.display(report)


@benchmark_app.command("compare")
def benchmark_compare(
    report1: str = typer.Argument(..., help="First report JSON"),
    report2: str = typer.Argument(..., help="Second report JSON"),
    output_json: bool = typer.Option(False, "--json", help="JSON output"),
):
    """Compare two benchmark reports."""
    from forge.benchmark.comparison import build_comparison, compare_reports, load_report

    missing = [path for path in (report1, report2) if not Path(path).is_file()]
    if missing:
        emit_cli_error(
            f"Benchmark report not found: {missing[0]}",
            output_json=output_json,
            exit_code=2,
        )

    try:
        reports = [load_report(report1), load_report(report2)]
    except (OSError, ValueError) as exc:
        emit_cli_error(
            f"Could not read benchmark report: {exc}",
            output_json=output_json,
            exit_code=2,
        )

    comparison = build_comparison(reports)
    if output_json:
        emit_json(comparison)
    else:
        compare_reports(reports, comparison=comparison)
