"""Model profile and recommendation commands."""

from __future__ import annotations

import sys
import time
from contextlib import nullcontext, redirect_stdout
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.live import Live
from rich.table import Table

from forge.cli_commands.shared import emit_json
from forge.gpu_utils import get_gpu_samples

console = Console()
profile_app = typer.Typer(name="profile", help="Model profiling & card generation")


@profile_app.command("card")
def profile_card(
    variant: str = typer.Option("nano", help="Student variant: micro/nano/small/medium"),
    model_dir: str = typer.Option(None, help="Model directory for autosense"),
    gpu_vram: float = typer.Option(24.0, help="Target GPU VRAM in GB"),
    dataset_size: int = typer.Option(50000, help="Dataset size for recommendations"),
    output: str = typer.Option(None, help="Save profile card to JSON file"),
    markdown: str = typer.Option(None, help="Save HuggingFace model card to file"),
    output_json: bool = typer.Option(False, "--json", help="JSON output"),
):
    """Generate a complete model profile card."""
    from forge.profiler import FORGEProfiler

    output_context = redirect_stdout(sys.stderr) if output_json else nullcontext()
    with output_context:
        profiler = FORGEProfiler(variant=variant, model_dir=model_dir)
        card = profiler.generate_card(dataset_size=dataset_size, gpu_vram_gb=gpu_vram)
        if output:
            card.save_json(output)
        if markdown:
            md = profiler.generate_markdown(card)
            Path(markdown).write_text(md, encoding="utf-8")

    if output and not output_json:
        console.print(f"Profile card saved to {output}")
    if markdown and not output_json:
        console.print(f"Model card saved to {markdown}")

    if output_json:
        emit_json(card.to_dict())
        return

    table = Table(title=f"FORGE Profile Card — {card.model_name}")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Variant", card.variant)
    table.add_row("Total Params", f"{card.total_params / 1e6:.1f}M")
    table.add_row("Trainable Params", f"{card.trainable_params / 1e6:.1f}M")
    table.add_row("Frozen Params", f"{card.frozen_params / 1e6:.1f}M")
    table.add_row("FP16 Size", f"{card.fp16_size_mb:.0f} MB")
    table.add_row("INT4 Size", f"{card.int4_size_mb:.0f} MB")
    if card.flops:
        table.add_row("FLOPs", f"{card.flops.total_gflops:.1f} GFLOPs")
    if card.vram:
        table.add_row("Training VRAM (FP16)", f"{card.vram.training_fp16_mb:.0f} MB")
        table.add_row("Inference VRAM (FP16)", f"{card.vram.inference_fp16_mb:.0f} MB")
        table.add_row("Recommended Batch", str(card.vram.recommended_batch_size))
    console.print(table)

    comp_table = Table(title="Component Breakdown")
    comp_table.add_column("Component", style="cyan")
    comp_table.add_column("Params", style="green")
    comp_table.add_column("Trainable", style="yellow")
    comp_table.add_column("FLOPs", style="magenta")
    for c in card.components:
        comp_table.add_row(
            c.name,
            f"{c.param_count / 1e6:.1f}M",
            f"{c.trainable_params / 1e6:.1f}M",
            f"{c.estimated_flops / 1e9:.1f}G",
        )
    console.print(comp_table)


@profile_app.command("vram")
def profile_vram(
    variant: str = typer.Option("nano", help="Student variant"),
    model_dir: str = typer.Option(None, help="Model directory for autosense"),
    gpu_vram: float = typer.Option(24.0, help="Target GPU VRAM in GB"),
    output_json: bool = typer.Option(False, "--json", help="JSON output"),
):
    """Estimate VRAM requirements."""
    from forge.profiler import FORGEProfiler

    output_context = redirect_stdout(sys.stderr) if output_json else nullcontext()
    with output_context:
        profiler = FORGEProfiler(variant=variant, model_dir=model_dir)
        vram = profiler.estimate_vram(gpu_vram_gb=gpu_vram)

    if output_json:
        emit_json(asdict(vram))
        return

    table = Table(title=f"VRAM Estimates — FORGE-{variant}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Inference (FP32)", f"{vram.inference_mb:.0f} MB")
    table.add_row("Inference (FP16)", f"{vram.inference_fp16_mb:.0f} MB")
    table.add_row("Training (FP32)", f"{vram.training_mb:.0f} MB")
    table.add_row("Training (FP16)", f"{vram.training_fp16_mb:.0f} MB")
    table.add_row("Per-sample Activations", f"{vram.per_sample_activation_mb:.1f} MB")
    table.add_row("Recommended Batch Size", str(vram.recommended_batch_size))
    console.print(table)

    gpu_table = Table(title="GPU Compatibility")
    gpu_table.add_column("GPU", style="cyan")
    gpu_table.add_column("Fits?", style="green")
    for gpu, fits in vram.fits_gpu.items():
        gpu_table.add_row(gpu, "Yes" if fits else "[red]No[/red]")
    console.print(gpu_table)


@profile_app.command("recommend")
def profile_recommend(
    variant: str = typer.Option("nano", help="Student variant"),
    model_dir: str = typer.Option(None, help="Model directory for autosense"),
    dataset_size: int = typer.Option(50000, help="Dataset size"),
    gpu_vram: float = typer.Option(24.0, help="Target GPU VRAM in GB"),
    objective: str = typer.Option("balanced", help="Objective: balanced/quality/speed"),
    output_json: bool = typer.Option(False, "--json", help="JSON output"),
):
    """Recommend training hyperparameters."""
    from forge.profiler import FORGEProfiler

    output_context = redirect_stdout(sys.stderr) if output_json else nullcontext()
    with output_context:
        profiler = FORGEProfiler(variant=variant, model_dir=model_dir)
        hp = profiler.recommend_hyperparams(
            dataset_size=dataset_size,
            gpu_vram_gb=gpu_vram,
            objective=objective,
        )

    if output_json:
        emit_json(asdict(hp))
        return

    table = Table(title=f"Recommended Hyperparams — FORGE-{variant} ({objective})")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Rationale", style="dim")
    table.add_row("Learning Rate", str(hp.learning_rate), hp.rationale.get("learning_rate", ""))
    table.add_row("Batch Size", str(hp.batch_size), hp.rationale.get("batch_size", ""))
    table.add_row("Grad Accum Steps", str(hp.gradient_accumulation_steps), "")
    table.add_row("Effective Batch", str(hp.effective_batch_size), "")
    table.add_row("Warmup Steps", str(hp.warmup_steps), hp.rationale.get("warmup_steps", ""))
    table.add_row("Max Steps", str(hp.max_steps), hp.rationale.get("max_steps", ""))
    table.add_row("Weight Decay", str(hp.weight_decay), "")
    table.add_row("LoRA Rank", str(hp.lora_rank), hp.rationale.get("lora_rank", ""))
    table.add_row("Action Head", hp.action_head_type, hp.rationale.get("action_head_type", ""))
    table.add_row("Bridge Queries", str(hp.bridge_n_queries), "")
    table.add_row("Bridge Layers", str(hp.bridge_n_layers), "")
    table.add_row("Flow Steps", str(hp.flow_inference_steps), hp.rationale.get("flow_inference_steps", ""))
    console.print(table)


def _format_gib(value_mib: int) -> str:
    """Render MiB as GiB for table readability."""
    return f"{value_mib / 1024:.2f}"


def _build_monitor_rows(
    samples: list[dict],
    training_fp16_mb: float,
    inference_fp16_mb: float,
    per_sample_activation_mb: float,
) -> tuple[Table, int, int]:
    """Build a Rich table for one GPU monitoring snapshot."""
    table = Table(title="FORGE GPU Live Fit Monitor")
    table.add_column("GPU", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("GPU Util", style="magenta")
    table.add_column("Mem Used", style="yellow")
    table.add_column("Mem Free", style="yellow")
    table.add_column("Fits Train", style="green")
    table.add_column("Fits Infer", style="green")
    table.add_column("Batch @Current", style="blue")

    training_payload = training_fp16_mb + per_sample_activation_mb
    train_ready = 0
    infer_ready = 0

    for sample in samples:
        idx: int = sample["index"]
        free_mib: int = sample["memory_free_mib"]
        used_mib: int = sample["memory_used_mib"]
        total_mib: int = sample["memory_total_mib"]
        gpu_util: int = sample["utilization_gpu"]
        gpu_util_text = "n/a" if gpu_util < 0 else f"{gpu_util}%"

        if per_sample_activation_mb > 0:
            remaining_for_act = free_mib - training_fp16_mb
            if remaining_for_act >= 0:
                fit_batch = int(remaining_for_act // per_sample_activation_mb)
                if fit_batch < 1:
                    fit_batch = 1 if remaining_for_act >= per_sample_activation_mb else 0
            else:
                fit_batch = 0
        else:
            fit_batch = 1 if free_mib > 0 else 0

        train_ok = free_mib >= training_payload
        infer_ok = free_mib >= inference_fp16_mb

        if train_ok:
            train_ready += 1
        if infer_ok:
            infer_ready += 1

        table.add_row(
            str(idx),
            sample["name"],
            gpu_util_text,
            f"{_format_gib(used_mib)} / {_format_gib(total_mib)} GiB",
            f"{_format_gib(free_mib)} GiB",
            "[green]Yes[/green]" if train_ok else "[red]No[/red]",
            "[green]Yes[/green]" if infer_ok else "[red]No[/red]",
            str(fit_batch),
        )

    return table, train_ready, infer_ready


@profile_app.command("monitor")
def profile_monitor(
    variant: str = typer.Option("nano", help="Student variant: micro/nano/small/medium"),
    model_dir: str = typer.Option(None, help="Model directory for autosense"),
    gpu_vram: float = typer.Option(24.0, help="Target GPU VRAM in GB"),
    refresh: float = typer.Option(2.0, help="Refresh interval in seconds"),
    iterations: int = typer.Option(0, help="Number of snapshots; 0 = endless"),
) -> None:
    """Stream per-GPU usage and model fit checks in near real-time."""
    from forge.profiler import FORGEProfiler

    if refresh <= 0:
        console.print("[red]--refresh must be greater than zero[/red]")
        raise typer.Exit(2)

    profiler = FORGEProfiler(variant=variant, model_dir=model_dir)
    vram = profiler.estimate_vram(gpu_vram_gb=gpu_vram)

    console.print(
        "[bold cyan]FORGE Live GPU Profiler[/bold cyan]"
        f" | model=FORGE-{variant}"
        f" | train_req={vram.training_fp16_mb:.0f} MB + act "
        f"{vram.per_sample_activation_mb:.1f} MB/sample"
    )

    count = 0
    try:
        with Live(auto_refresh=True, refresh_per_second=4, vertical_overflow="ellipsis") as live:
            while True:
                samples = get_gpu_samples()
                if not samples:
                    raise RuntimeError("nvidia-smi is unavailable or no GPUs detected.")

                timestamp = datetime.now(UTC).isoformat()
                table, train_ready, infer_ready = _build_monitor_rows(
                    samples=samples,
                    training_fp16_mb=vram.training_fp16_mb,
                    inference_fp16_mb=vram.inference_fp16_mb,
                    per_sample_activation_mb=vram.per_sample_activation_mb,
                )

                can_train = "[green]YES[/green]" if train_ready else "[red]NO[/red]"
                can_infer = "[green]YES[/green]" if infer_ready else "[red]NO[/red]"

                status_line = f"Timestamp: {timestamp} | Iteration: {count + 1}"
                if iterations:
                    status_line += f"/{iterations}"

                if iterations:
                    status_line += " | remaining: " + str(max(iterations - count - 1, 0))

                status_line += f" | Can train now: {can_train} ({train_ready}/{len(samples)} GPUs)"
                status_line += f" | Can infer now: {can_infer} ({infer_ready}/{len(samples)} GPUs)"

                table.title = f"{table.title} — {status_line}"
                live.update(table)

                count += 1
                if iterations and count >= iterations:
                    break

                time.sleep(refresh)
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
