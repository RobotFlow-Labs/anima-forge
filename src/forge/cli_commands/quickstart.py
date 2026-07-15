"""A bounded, real-label first distillation workflow."""

from __future__ import annotations

import os
import shlex
import sys
import time
from collections.abc import Callable
from contextlib import nullcontext, redirect_stdout
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from forge.cli_commands.shared import emit_cli_error, emit_json, resolve_runtime_device

console = Console()
SAMPLE_LABELS_REPO = "robotflowlabs/forge-sample-labels"


def _quickstart_next_steps(
    *,
    checkpoint: Path,
    output_dir: Path,
    data_dir: Path,
    device: str,
) -> list[str]:
    compressed_output = output_dir.parent / f"{output_dir.name}-compressed"
    compressed_checkpoint = compressed_output / "compressed" / "qvla_4bit.pt"
    return [
        " ".join(
            (
                "forge pipeline --config configs/forge_nano.yaml --stage compress",
                f"--checkpoint {shlex.quote(str(checkpoint))}",
                f"--data-dir {shlex.quote(str(data_dir))}",
                f"--output-dir {shlex.quote(str(compressed_output))}",
            )
        ),
        " ".join(
            (
                "forge benchmark run",
                f"--checkpoint {shlex.quote(str(compressed_checkpoint))}",
                f"--device {device}",
                "--data-dir /path/to/real-lerobot-dataset",
                "--instruction 'describe the real task'",
            )
        ),
    ]


def _quickstart_assets():
    from forge.model_assets import find_model_asset

    assets = (
        find_model_asset("Qwen/Qwen3-0.6B"),
        find_model_asset("google/siglip2-so400m-patch14-384"),
    )
    if any(asset is None for asset in assets):
        raise RuntimeError("The built-in nano/vision asset manifest is incomplete")
    return tuple(asset for asset in assets if asset is not None)


def _ensure_real_sample_labels(
    data_dir: Path,
    *,
    repo_id: str,
    token: str | None,
) -> Path:
    """Use a local real label pack or download the published sample pack."""
    from forge.data.teacher_dataset import TeacherLabelDataset

    label_dir = data_dir / "teacher_labels"
    if not (label_dir / "metadata.json").is_file():
        try:
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=str(data_dir),
                token=token,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not download real sample labels from {repo_id} into {data_dir}: {exc}. "
                "Set HF_TOKEN and retry `forge quickstart --yes`, or generate labels with "
                "`forge pipeline --stage labels --output-dir ./outputs/labels`."
            ) from exc

    try:
        dataset = TeacherLabelDataset(label_dir)
        provenance = dataset.labels_provenance
        episode_count = len(dataset)
        dataset.close()
    except Exception as exc:
        raise RuntimeError(
            f"Sample label pack at {label_dir} is unreadable: {exc}. Remove that directory "
            "and retry `forge quickstart --yes`."
        ) from exc
    if provenance != "real" or episode_count < 1:
        raise RuntimeError(
            f"Sample label pack at {label_dir} is not trusted real-teacher data. Remove it and "
            "retry `forge quickstart --yes`."
        )
    return label_dir


def run_quickstart(
    *,
    device: str,
    model_dir: Path,
    data_dir: Path,
    output_dir: Path,
    max_steps: int,
    batch_size: int,
    sample_labels_repo: str,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, Any]:
    """Run doctor, idempotent asset setup, real labels, and bounded distillation."""
    from huggingface_hub import get_token

    from forge.cli_commands._doctor_core import _default_hf_cache_dir
    from forge.cli_commands.doctor import run_doctor
    from forge.cli_commands.fetch import fetch_assets
    from forge.config import ForgeConfig, apply_student_variant
    from forge.pipeline import run_pipeline

    started = time.perf_counter()
    assets = _quickstart_assets()
    doctor = run_doctor(
        model_dir=model_dir,
        output_dir=output_dir,
        expected_assets=assets,
    )
    doctor_failed = int(doctor.get("exit_code", 0)) >= 2
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "doctor",
                "status": "failed" if doctor_failed else "completed",
                "summary": doctor["summary"],
            }
        )
    if doctor_failed:
        failed_checks = [
            str(check.get("name", "unknown"))
            for check in doctor.get("checks", [])
            if isinstance(check, dict) and check.get("status") == "error"
        ]
        detail = f" Failing checks: {', '.join(failed_checks)}." if failed_checks else ""
        raise RuntimeError(
            "FORGE doctor found blocking environment errors; quickstart stopped before downloads or training."
            f"{detail} Run `forge doctor`, fix the reported errors, and retry."
        )

    fetch_report = fetch_assets(
        assets,
        model_dir=model_dir,
        cache_dir=_default_hf_cache_dir(),
        token=get_token(),
    )
    if fetch_report["exit_code"]:
        failed = next(item for item in fetch_report["results"] if item["status"] == "error")
        raise RuntimeError(
            f"Model setup failed for {failed['repo_id']} at {failed['path']}: {failed['error']}. "
            f"Retry `forge models fetch {failed['repo_id']}` and then run `forge doctor`."
        )
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "models",
                "status": "completed",
                "summary": fetch_report["summary"],
            }
        )

    label_dir = _ensure_real_sample_labels(data_dir, repo_id=sample_labels_repo, token=get_token())
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "labels",
                "status": "completed",
                "path": str(label_dir),
            }
        )

    config = ForgeConfig.default()
    apply_student_variant(config.student, "nano")
    config.student.allow_mock = False
    config.paths.model_dir = str(model_dir)
    config.paths.data_dir = str(data_dir)
    config.paths.output_dir = str(output_dir)
    config.paths.language_model = "Qwen--Qwen3-0.6B"
    config.paths.vision_encoder = "google--siglip2-so400m-patch14-384"
    config.distill.batch_size = batch_size
    config.distill.gradient_accumulation_steps = 4

    pipeline = run_pipeline(
        config,
        device=device,
        skip_labels=True,
        stage="distill",
        max_distill_steps=max_steps,
        progress_callback=progress_callback,
    )
    if pipeline.get("status") != "completed":
        error = pipeline.get("distill", {}).get("error", "distillation failed")
        raise RuntimeError(
            f"Quickstart distillation failed for output {output_dir}: {error}. Run `forge doctor`, "
            "then retry `forge quickstart --yes`."
        )

    checkpoint = output_dir / "checkpoints" / "final.pt"
    if not checkpoint.is_file():
        raise RuntimeError(
            f"Quickstart completed without the expected checkpoint at {checkpoint}. "
            "Inspect the pipeline summary and retry `forge quickstart --yes`."
        )
    return {
        "status": "completed",
        "device": device,
        "variant": "nano",
        "steps": max_steps,
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint": str(checkpoint.resolve()),
        "pipeline_summary": pipeline["pipeline_summary_path"],
        "labels": {"path": str(label_dir.resolve()), "provenance": "real"},
        "doctor": {"status": doctor["status"], "summary": doctor["summary"]},
        "models": fetch_report["summary"],
        "next_steps": _quickstart_next_steps(
            checkpoint=checkpoint,
            output_dir=output_dir,
            data_dir=data_dir,
            device=device,
        ),
    }


def quickstart_command(
    yes: bool = typer.Option(False, "--yes", "-y", help="Accept required model/label downloads"),
    device: str | None = typer.Option(None, help="Device: auto|cuda|cpu"),
    model_dir: Path = typer.Option(Path("./models"), help="Model asset directory"),
    data_dir: Path = typer.Option(Path("~/.cache/forge/quickstart/data"), help="Real sample label directory"),
    output_dir: Path = typer.Option(Path("./outputs/quickstart"), help="Checkpoint/output directory"),
    max_steps: int = typer.Option(200, min=1, help="Bounded distillation steps"),
    batch_size: int = typer.Option(16, min=1, help="Distillation batch size"),
    sample_labels_repo: str = typer.Option(SAMPLE_LABELS_REPO, help="Hugging Face sample-label dataset"),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress progress; print only the final result"),
    output_json: bool = typer.Option(False, "--json", help="Emit one final JSON document"),
) -> None:
    """Download the nano stack and run a real 200-step first distillation."""
    try:
        model_dir = Path(os.environ.get("FORGE_MODEL_DIR", str(model_dir))).expanduser().resolve()
        data_dir = data_dir.expanduser().resolve()
        output_dir = output_dir.expanduser().resolve()
        device = resolve_runtime_device(device=device, command="quickstart", default="auto", strict=True)
        assets = _quickstart_assets()
        needs_download = any(not (model_dir / asset.local_name).exists() for asset in assets)
        needs_download = needs_download or not (data_dir / "teacher_labels" / "metadata.json").is_file()
        if needs_download and not yes:
            if output_json:
                emit_cli_error(
                    "quickstart requires downloads; rerun `forge quickstart --yes --json`",
                    output_json=True,
                    exit_code=2,
                )
            if not typer.confirm("Download the nano backbone, SigLIP2, and real sample labels?"):
                raise typer.Abort()
    except (typer.Abort, typer.Exit):
        raise
    except Exception as exc:
        emit_cli_error(str(exc), output_json=output_json, exit_code=2)

    def show_progress(event: dict[str, object]) -> None:
        if quiet or output_json:
            return
        status = event.get("status")
        stage = str(event.get("stage", "quickstart"))
        if status == "progress":
            step = int(str(event.get("step", 0)))
            total = int(str(event.get("total_steps", 0)))
            loss = float(str(event.get("loss", 0.0)))
            eta = float(str(event.get("eta_seconds", 0.0)))
            console.print(f"  train {step}/{total} · loss {loss:.4f} · ETA {eta:.0f}s")
        elif status == "started":
            console.rule(str(event.get("title", stage.title())))
        else:
            console.print(f"[green]DONE[/green] {stage}")

    output_context = redirect_stdout(sys.stderr) if output_json else nullcontext()
    try:
        with output_context:
            result = run_quickstart(
                device=device,
                model_dir=model_dir,
                data_dir=data_dir,
                output_dir=output_dir,
                max_steps=max_steps,
                batch_size=batch_size,
                sample_labels_repo=sample_labels_repo,
                progress_callback=show_progress,
            )
    except Exception as exc:
        emit_cli_error(str(exc), output_json=output_json, exit_code=2)

    if output_json:
        emit_json(result)
        return
    table = Table(title="FORGE Quickstart Complete")
    table.add_column("Result", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Checkpoint", result["checkpoint"])
    table.add_row("Steps", str(result["steps"]))
    table.add_row("Elapsed", f"{result['elapsed_seconds']:.1f}s")
    table.add_row("Labels", "real")
    console.print(table)
    console.print("[bold]Next steps[/bold]")
    for command in result["next_steps"]:
        console.print(f"  {command}")


__all__ = ["_quickstart_next_steps", "quickstart_command", "run_quickstart"]
