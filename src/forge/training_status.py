"""Shared readers for persisted production-training process state."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from forge.training_runtime import (
    latest_run_dir,
    process_identity_matches,
    process_is_running,
    read_heartbeat,
)


def resolve_training_run_dir(
    run_dir: str | Path | None,
    output_dir: str | Path,
) -> Path:
    """Resolve an explicit run or the newest run below an output root."""
    if run_dir is not None:
        return Path(run_dir).expanduser().resolve()
    return latest_run_dir(Path(output_dir).expanduser().resolve())


def read_training_process_record(
    run_dir: str | Path,
    state: dict[str, Any],
) -> tuple[int | None, int | None]:
    """Read a PID and process identity from the heartbeat or atomic pidfile."""
    value = state.get("pid")
    if isinstance(value, int) and value > 0:
        start_time = state.get("process_start_time_ticks")
        return value, start_time if isinstance(start_time, int) and start_time > 0 else None
    pidfile = Path(run_dir) / "train.pid"
    try:
        raw = pidfile.read_text(encoding="utf-8").strip()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            legacy_pid = int(raw)
            return (legacy_pid, None) if legacy_pid > 0 else (None, None)
        if not isinstance(payload, dict):
            return None, None
        pid = payload.get("pid")
        start_time = payload.get("process_start_time_ticks")
        if not isinstance(pid, int) or pid <= 0:
            return None, None
        return pid, start_time if isinstance(start_time, int) and start_time > 0 else None
    except (OSError, ValueError):
        return None, None


def read_training_pid(run_dir: str | Path, state: dict[str, Any]) -> int | None:
    """Read a heartbeat PID, falling back to the detached-run pidfile."""
    pid, _start_time = read_training_process_record(run_dir, state)
    return pid


def read_training_run_status(
    *,
    run_dir: str | Path | None = None,
    output_dir: str | Path = "./outputs",
) -> tuple[Path, dict[str, Any]]:
    """Return persisted state annotated with real process liveness."""
    selected = resolve_training_run_dir(run_dir, output_dir)
    state = read_heartbeat(selected)
    pid, start_time = read_training_process_record(selected, state)
    running = process_identity_matches(pid, start_time) if start_time is not None else process_is_running(pid)
    result = {
        **state,
        "pid": pid,
        "process_start_time_ticks": start_time,
        "process_running": running,
    }
    if state.get("status") in {"launching", "starting", "running", "stopping"} and not running:
        result["status"] = "stale"
        result["note"] = "Heartbeat is non-terminal but its process is not running."
    return selected, result


__all__ = [
    "read_training_pid",
    "read_training_process_record",
    "read_training_run_status",
    "resolve_training_run_dir",
]
