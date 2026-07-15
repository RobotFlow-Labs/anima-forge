"""Agent-facing top/status CLI commands."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from .shared import emit_json, json_payload

console = Console()
MAX_TOP_OUTPUT_BYTES = 64_000


def top(
    show_jobs: bool = typer.Option(True, "--jobs/--no-jobs", help="Include active job snapshots"),
    output_json: bool = typer.Option(False, "--json", help="JSON output"),
    max_output_bytes: int = typer.Option(
        MAX_TOP_OUTPUT_BYTES,
        "--max-output-bytes",
        help="Warn when JSON output exceeds this many bytes (0 to disable)",
    ),
) -> None:
    """Agent-oriented status snapshot similar to a lightweight top command."""
    from forge.web.state import ServerState

    state = ServerState()
    status = state.get_system_status()

    payload: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "status": status,
        "train_state": state.train_state,
    }

    if show_jobs:
        payload["jobs"] = [
            {
                "id": job.job_id,
                "name": job.name,
                "status": job.status,
                "started_at": job.started_at,
                "finished_at": job.finished_at,
                "error": job.error,
            }
            for job in state.active_jobs.values()
        ]

    if output_json:
        payload_size = len(json_payload(payload).encode("utf-8"))
        if max_output_bytes > 0 and payload_size > max_output_bytes:
            logging.getLogger("forge").warning(
                "top JSON output exceeds the requested size cap (%s > %s bytes); "
                "emitting the full document to preserve valid JSON.",
                payload_size,
                max_output_bytes,
            )
        emit_json(payload)
        return

    table = Table(title="FORGE Agent Top")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Timestamp", payload["timestamp"])
    table.add_row("Version", str(status.get("version", "unknown")))
    table.add_row("GPU", str(status.get("gpu", "N/A")))
    table.add_row(
        "VRAM",
        f"{status.get('vram_used_gb', 0):.2f} / {status.get('vram_total_gb', 0):.2f} GB",
    )
    table.add_row("Disk Free", f"{status.get('disk_free_gb', 0):.2f} GB")
    table.add_row("Uptime", f"{status.get('uptime_s', 0):.1f}s")
    table.add_row("Active Jobs", str(status.get("active_jobs", 0)))
    table.add_row("Train Running", str(payload["train_state"].get("running", False)))

    if show_jobs:
        for entry in payload["jobs"]:
            table.add_row(f"job:{entry['id']}", f"{entry['name']} [{entry['status']}]")

    console.print(table)


def top_agent(
    show_jobs: bool = typer.Option(True, "--jobs/--no-jobs", help="Include active job snapshots"),
    output_json: bool = typer.Option(False, "--json", help="JSON output"),
    max_output_bytes: int = typer.Option(
        MAX_TOP_OUTPUT_BYTES,
        "--max-output-bytes",
        help="Warn when JSON output exceeds this many bytes (0 to disable)",
    ),
) -> None:
    """Alias for `forge top` for agent and automation workflows."""
    top(show_jobs=show_jobs, output_json=output_json, max_output_bytes=max_output_bytes)


def status_command(
    output_json: bool = typer.Option(False, "--json", help="JSON output"),
    max_output_bytes: int = typer.Option(
        MAX_TOP_OUTPUT_BYTES,
        "--max-output-bytes",
        help="Warn when JSON output exceeds this many bytes (0 to disable)",
    ),
) -> None:
    """Show FORGE system status."""
    from forge.web.state import ServerState

    state = ServerState()
    info = state.get_system_status()

    if output_json:
        payload_size = len(json_payload(info).encode("utf-8"))
        if max_output_bytes > 0 and payload_size > max_output_bytes:
            logging.getLogger("forge").warning(
                "status JSON output exceeds the requested size cap (%s > %s bytes); "
                "emitting the full document to preserve valid JSON.",
                payload_size,
                max_output_bytes,
            )
        emit_json(info)
        return

    table = Table(title="FORGE Status")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    for key, value in info.items():
        table.add_row(key, str(value))
    console.print(table)
