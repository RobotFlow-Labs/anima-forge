"""Fine-tuning workflow commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from forge.cli_commands.shared import emit_json

console = Console()
finetune_app = typer.Typer(name="finetune", help="Domain adaptation & fine-tuning (PRD-28)")


@finetune_app.command("status")
def finetune_status(
    output_dir: str = typer.Option("./outputs/finetune", help="Fine-tune output directory"),
    output_json: bool = typer.Option(False, "--json", help="JSON output"),
):
    """Show fine-tuning status and recent checkpoints."""
    output_path = Path(output_dir)

    checkpoints = sorted(output_path.glob("finetune_*.pt")) if output_path.exists() else []
    info = {
        "output_dir": str(output_path),
        "checkpoint_count": len(checkpoints),
        "checkpoints": [str(c.name) for c in checkpoints[-5:]],
    }

    if checkpoints:
        import torch

        try:
            ckpt = torch.load(checkpoints[-1], map_location="cpu", weights_only=True)
            info["latest_step"] = ckpt.get("global_step", 0)
            info["strategy"] = ckpt.get("strategy", "unknown")
        except Exception:
            pass

    if output_json:
        emit_json(info)
    else:
        table = Table(title="Fine-Tuning Status")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        for k, v in info.items():
            if isinstance(v, list):
                table.add_row(k, ", ".join(v) if v else "none")
            else:
                table.add_row(k, str(v))
        console.print(table)


@finetune_app.command("list")
def finetune_list(
    output_dir: str = typer.Option("./outputs/finetune", help="Fine-tune output directory"),
    output_json: bool = typer.Option(False, "--json", help="JSON output"),
):
    """List fine-tuning checkpoints."""
    output_path = Path(output_dir)
    checkpoints = sorted(output_path.glob("finetune_*.pt")) if output_path.exists() else []

    entries: list[dict[str, Any]] = []
    for ckpt_path in checkpoints:
        entry = {
            "name": ckpt_path.name,
            "size_mb": round(ckpt_path.stat().st_size / (1024 * 1024), 1),
        }
        try:
            import torch

            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            entry["step"] = ckpt.get("global_step", 0)
            entry["strategy"] = ckpt.get("strategy", "unknown")
        except Exception:
            pass
        entries.append(entry)

    if output_json:
        emit_json(entries)
    else:
        if not entries:
            console.print("[yellow]No checkpoints found[/yellow]")
            return
        table = Table(title="Fine-Tuning Checkpoints")
        table.add_column("Name", style="cyan")
        table.add_column("Step", style="green")
        table.add_column("Strategy")
        table.add_column("Size (MB)")
        for e in entries:
            table.add_row(
                e["name"],
                str(e.get("step", "—")),
                e.get("strategy", "—"),
                str(e.get("size_mb", "—")),
            )
        console.print(table)
