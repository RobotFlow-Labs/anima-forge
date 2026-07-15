"""Truthful production training commands with persistent run state."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from forge.cli_commands.shared import emit_cli_error, emit_json, resolve_runtime_device
from forge.training_runtime import (
    TrainingRuntimeError,
    atomic_write_json,
    build_production_trainer,
    create_run_dir,
    cuda_memory_snapshot,
    process_identity_matches,
    process_is_running,
    process_start_time_ticks,
    read_heartbeat,
)
from forge.training_status import (
    read_training_process_record,
    read_training_run_status,
    resolve_training_run_dir,
)

console = Console()
error_console = Console(stderr=True)
train_app = typer.Typer(name="train", help="Production training pipeline (PRD-23)")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _load_config(
    config_path: Path | None,
    *,
    output_dir: Path | None,
    data_dir: Path | None,
    max_steps: int,
    curriculum: bool,
    plateau: bool,
    hard_mining: bool,
    allow_mock: bool,
):
    from forge.config import ForgeConfig

    config = ForgeConfig.from_yaml(config_path) if config_path else ForgeConfig.default()
    if output_dir is not None:
        config.paths.output_dir = str(output_dir.expanduser())
    if data_dir is not None:
        config.paths.data_dir = str(data_dir.expanduser())
    config.distill.max_steps = max_steps
    config.curriculum.enabled = curriculum
    config.curriculum.hard_example_mining = hard_mining
    config.student.allow_mock = bool(config.student.allow_mock or allow_mock)
    if not plateau:
        config.curriculum.plateau_window = 0
    return config


def _write_heartbeat(run_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    state = {
        **payload,
        "schema_version": 1,
        "run_dir": str(run_dir),
        "updated_at": _utc_now(),
    }
    pid = state.get("pid")
    if isinstance(pid, int) and "process_start_time_ticks" not in state:
        state["process_start_time_ticks"] = process_start_time_ticks(pid)
    atomic_write_json(run_dir / "train_state.json", state)
    return state


def _write_pidfile(run_dir: Path, pid: int) -> Path:
    """Atomically bind a detached run to an exact process instance."""
    start_time = process_start_time_ticks(pid)
    if start_time is None:
        raise TrainingRuntimeError(f"Cannot record process identity for detached training PID {pid}")
    return atomic_write_json(
        run_dir / "train.pid",
        {"pid": pid, "process_start_time_ticks": start_time},
    )


def _remove_owned_pidfile(run_dir: Path, pid: int) -> None:
    """Remove the pidfile only when it still belongs to this worker."""
    record_pid, _start_time = read_training_process_record(run_dir, {})
    if record_pid == pid:
        (run_dir / "train.pid").unlink(missing_ok=True)


def _send_training_sigint(pid: int, expected_start_time_ticks: object) -> None:
    """Signal only the process instance bound to the recorded training PID."""
    pidfd_open = getattr(os, "pidfd_open", None)
    pidfd_send_signal = getattr(signal, "pidfd_send_signal", None)
    if callable(pidfd_open) and callable(pidfd_send_signal):
        descriptor = pidfd_open(pid)
        try:
            if not process_identity_matches(pid, expected_start_time_ticks):
                raise TrainingRuntimeError(
                    f"Refusing to signal PID {pid}: its process identity does not match the training heartbeat"
                )
            pidfd_send_signal(descriptor, signal.SIGINT)
        finally:
            os.close(descriptor)
        return

    if not process_identity_matches(pid, expected_start_time_ticks):
        raise TrainingRuntimeError(
            f"Refusing to signal PID {pid}: its process identity does not match the training heartbeat"
        )
    os.kill(pid, signal.SIGINT)


def _progress_line(progress: dict[str, Any]) -> str:
    return (
        f"step {progress['step']}/{progress['max_steps']} "
        f"loss={progress['loss']:.6f} phase={progress['phase']} "
        f"eta={progress['eta_seconds']:.1f}s"
    )


def _batch_log_detail(details: dict[str, Any]) -> str:
    if details.get("source") != "vram_estimate":
        return str(details.get("source", "unknown"))
    limiter = f", limited_by={details['limiting_factor']}" if details.get("limiting_factor") else ""
    return (
        f"vram_estimate, estimated={details['estimated_utilization']:.1%}, "
        f"target={details['target_utilization']:.0%}{limiter}"
    )


def _curriculum_snapshot(config, trainer: Any | None = None) -> dict[str, Any]:
    """Build a truthful curriculum snapshot from this run config and trainer."""
    trainer_status = trainer.get_status() if trainer is not None else {}
    enabled = bool(config.curriculum.enabled)
    difficulty = trainer_status.get("curriculum_difficulty")
    if enabled and difficulty is None and trainer is None:
        difficulty = config.curriculum.initial_difficulty
    return {
        "enabled": enabled,
        "difficulty": difficulty if enabled else None,
        "difficulty_metric": config.curriculum.difficulty_metric,
        "initial_difficulty": config.curriculum.initial_difficulty,
        "final_difficulty": config.curriculum.final_difficulty,
        "ramp_schedule": config.curriculum.ramp_schedule,
        "ramp_steps": config.curriculum.ramp_steps,
        "hard_example_mining": bool(config.curriculum.hard_example_mining),
        "hard_examples_seen": trainer_status.get("hard_examples_seen", 0),
        "plateau_detection": config.curriculum.plateau_window > 0,
        "plateaus": trainer_status.get("plateaus", 0),
        "teacher_dropout": bool(config.curriculum.teacher_dropout),
        "teacher_dropout_rate": trainer_status.get("teacher_dropout_rate"),
    }


def _run_training(
    *,
    config,
    config_path: Path | None,
    device: str,
    run_dir: Path,
    max_steps: int,
    batch_size: int | None,
    heartbeat_every: int,
    log_every: int,
    checkpoint_every: int | None,
    output_json: bool,
) -> dict[str, Any]:
    stop_event = threading.Event()
    trainer: Any | None = None
    detached_worker = os.environ.get("FORGE_TRAIN_WORKER") == "1"
    install_signal_handler = threading.current_thread() is threading.main_thread()
    previous_sigint = signal.getsignal(signal.SIGINT) if install_signal_handler else None

    def request_stop(_signum, _frame) -> None:
        stop_event.set()

    if install_signal_handler:
        signal.signal(signal.SIGINT, request_stop)
    _write_heartbeat(
        run_dir,
        {
            "status": "starting",
            "pid": os.getpid(),
            "device": device,
            "config_path": str(config_path.resolve()) if config_path else None,
            "max_steps": max_steps,
            "step": 0,
            "loss": None,
            "phase": None,
            "eta_seconds": None,
            "curriculum": _curriculum_snapshot(config),
        },
    )

    try:
        trainer, label_dir, batch_details = build_production_trainer(
            config,
            device=device,
            run_dir=run_dir,
            batch_size=batch_size,
        )
        starting_state = _write_heartbeat(
            run_dir,
            {
                "status": "running",
                "pid": os.getpid(),
                "device": device,
                "config_path": str(config_path.resolve()) if config_path else None,
                "label_dir": str(label_dir),
                "max_steps": max_steps,
                "step": 0,
                "loss": None,
                "phase": 1,
                "eta_seconds": None,
                "batch_size": config.distill.batch_size,
                "batch_selection": batch_details,
                "curriculum": _curriculum_snapshot(config, trainer),
                "vram": cuda_memory_snapshot(device),
            },
        )
        batch_log_detail = _batch_log_detail(batch_details)
        if output_json:
            error_console.print(
                f"training run={run_dir} batch={config.distill.batch_size} selection={batch_log_detail}"
            )
        else:
            console.print(f"[bold cyan]Production training[/bold cyan] — {run_dir}")
            console.print(
                f"batch={config.distill.batch_size} ({batch_log_detail}), device={device}, labels={label_dir}"
            )

        last_state = starting_state

        def on_progress(progress: dict[str, Any]) -> None:
            nonlocal last_state
            step = int(progress["step"])
            if step == 1 or step % heartbeat_every == 0 or step == max_steps:
                last_state = _write_heartbeat(
                    run_dir,
                    {
                        "status": "stopping" if stop_event.is_set() else "running",
                        "pid": os.getpid(),
                        "device": device,
                        "config_path": str(config_path.resolve()) if config_path else None,
                        "label_dir": str(label_dir),
                        "batch_size": config.distill.batch_size,
                        "batch_selection": batch_details,
                        **progress,
                        "curriculum": _curriculum_snapshot(config, trainer),
                        "vram": cuda_memory_snapshot(device),
                    },
                )
            if step == 1 or step % log_every == 0 or step == max_steps:
                line = _progress_line(progress)
                if output_json:
                    error_console.print(line)
                else:
                    console.print(line)

        report = trainer.train(
            max_steps=max_steps,
            log_every=log_every,
            checkpoint_every=checkpoint_every,
            progress_callback=on_progress,
            stop_requested=stop_event.is_set,
        )
        final = {
            **report.to_dict(),
            "run_dir": str(run_dir),
            "pid": os.getpid(),
            "device": device,
            "max_steps": max_steps,
            "step": report.total_steps,
            "loss": report.final_loss,
            "phase": trainer.state.phase,
            "batch_size": config.distill.batch_size,
            "batch_selection": batch_details,
            "label_dir": str(label_dir),
            "curriculum": _curriculum_snapshot(config, trainer),
            "eta_seconds": 0.0 if report.status == "completed" else last_state.get("eta_seconds"),
            "vram": cuda_memory_snapshot(device),
            "completed_at": _utc_now(),
        }
        _write_heartbeat(run_dir, final)
        return final
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        _write_heartbeat(
            run_dir,
            {
                "status": "failed",
                "pid": os.getpid(),
                "device": device,
                "max_steps": max_steps,
                "curriculum": _curriculum_snapshot(config),
                "error": str(exc),
                "completed_at": _utc_now(),
            },
        )
        raise
    finally:
        if trainer is not None:
            close_dataset = getattr(trainer.dataset, "close", None)
            if callable(close_dataset):
                close_dataset()
        if install_signal_handler:
            signal.signal(signal.SIGINT, previous_sigint)
        if detached_worker:
            _remove_owned_pidfile(run_dir, os.getpid())


def _detached_command(
    *,
    config_path: Path | None,
    device: str,
    max_steps: int,
    batch_size: int | None,
    output_dir: Path,
    data_dir: Path | None,
    run_dir: Path,
    heartbeat_every: int,
    log_every: int,
    checkpoint_every: int | None,
    curriculum: bool,
    plateau: bool,
    hard_mining: bool,
    allow_mock: bool,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "forge.cli_v2",
        "train",
        "start",
        "--device",
        device,
        "--max-steps",
        str(max_steps),
        "--output-dir",
        str(output_dir),
        "--run-dir",
        str(run_dir),
        "--heartbeat-every",
        str(heartbeat_every),
        "--log-every",
        str(log_every),
        "--curriculum" if curriculum else "--no-curriculum",
        "--plateau" if plateau else "--no-plateau",
        "--hard-mining" if hard_mining else "--no-hard-mining",
        "--json",
    ]
    if config_path is not None:
        command.extend(("--config", str(config_path.resolve())))
    if data_dir is not None:
        command.extend(("--data-dir", str(data_dir.resolve())))
    if batch_size is not None:
        command.extend(("--batch-size", str(batch_size)))
    if checkpoint_every is not None:
        command.extend(("--checkpoint-every", str(checkpoint_every)))
    if allow_mock:
        command.append("--allow-mock")
    return command


def _launch_detached(command: list[str], run_dir: Path) -> int:
    stdout_path = run_dir / "train.stdout.log"
    stderr_path = run_dir / "train.stderr.log"
    environment = os.environ.copy()
    environment["FORGE_TRAIN_WORKER"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    with stdout_path.open("ab", buffering=0) as stdout, stderr_path.open("ab", buffering=0) as stderr:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            env=environment,
            start_new_session=True,
            close_fds=True,
        )
    try:
        _write_pidfile(run_dir, process.pid)
    except BaseException:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise
    return process.pid


@train_app.command("start")
def train_start(
    config_path: Path | None = typer.Option(None, "--config", help="YAML config path"),
    device: str | None = typer.Option(None, "--device", help="Device (auto|cuda|cpu)"),
    max_steps: int = typer.Option(50000, "--max-steps", min=1, help="Max training steps"),
    batch_size: int | None = typer.Option(None, "--batch-size", min=1, help="Explicit batch size"),
    output_dir: Path | None = typer.Option(None, "--output-dir", help="Training output root"),
    data_dir: Path | None = typer.Option(None, "--data-dir", help="Teacher-label data root"),
    run_dir: Path | None = typer.Option(None, "--run-dir", hidden=True),
    detach: bool = typer.Option(False, "--detach", help="Run in a detached process"),
    heartbeat_every: int = typer.Option(10, "--heartbeat-every", min=1),
    log_every: int = typer.Option(10, "--log-every", min=1),
    checkpoint_every: int | None = typer.Option(None, "--checkpoint-every", min=1),
    curriculum: bool = typer.Option(True, "--curriculum/--no-curriculum"),
    plateau: bool = typer.Option(True, "--plateau/--no-plateau"),
    hard_mining: bool = typer.Option(True, "--hard-mining/--no-hard-mining"),
    allow_mock: bool = typer.Option(
        False,
        "--allow-mock",
        help="Explicitly permit mock backbones or labels (artifacts are stamped MOCK)",
    ),
    output_json: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Start real ProductionTrainer training and persist run state."""
    try:
        resolved_device = resolve_runtime_device(
            device=device,
            command="train",
            default="auto",
            strict=True,
        )
        config = _load_config(
            config_path,
            output_dir=output_dir,
            data_dir=data_dir,
            max_steps=max_steps,
            curriculum=curriculum,
            plateau=plateau,
            hard_mining=hard_mining,
            allow_mock=allow_mock,
        )
        output_root = Path(config.paths.output_dir).expanduser().resolve()
        selected_run_dir = run_dir.expanduser().resolve() if run_dir else create_run_dir(output_root)
        selected_run_dir.mkdir(parents=True, exist_ok=True)

        is_worker = os.environ.get("FORGE_TRAIN_WORKER") == "1"
        if detach and not is_worker:
            _write_heartbeat(
                selected_run_dir,
                {
                    "status": "launching",
                    "pid": None,
                    "device": resolved_device,
                    "max_steps": max_steps,
                    "step": 0,
                    "loss": None,
                    "phase": None,
                    "eta_seconds": None,
                    "curriculum": _curriculum_snapshot(config),
                },
            )
            command = _detached_command(
                config_path=config_path,
                device=resolved_device,
                max_steps=max_steps,
                batch_size=batch_size,
                output_dir=output_root,
                data_dir=data_dir,
                run_dir=selected_run_dir,
                heartbeat_every=heartbeat_every,
                log_every=log_every,
                checkpoint_every=checkpoint_every,
                curriculum=curriculum,
                plateau=plateau,
                hard_mining=hard_mining,
                allow_mock=bool(config.student.allow_mock),
            )
            pid = _launch_detached(command, selected_run_dir)
            result = {
                "status": "launched",
                "pid": pid,
                "run_dir": str(selected_run_dir),
                "heartbeat": str(selected_run_dir / "train_state.json"),
                "stdout_log": str(selected_run_dir / "train.stdout.log"),
                "stderr_log": str(selected_run_dir / "train.stderr.log"),
            }
        else:
            result = _run_training(
                config=config,
                config_path=config_path,
                device=resolved_device,
                run_dir=selected_run_dir,
                max_steps=max_steps,
                batch_size=batch_size,
                heartbeat_every=heartbeat_every,
                log_every=log_every,
                checkpoint_every=checkpoint_every,
                output_json=output_json,
            )

        if output_json:
            emit_json(result)
        elif detach and not is_worker:
            console.print(f"[green]Training launched[/green] pid={result['pid']}")
            console.print(f"Run directory: {result['run_dir']}")
        else:
            console.print(
                f"[green]Training {result['status']}[/green]: step={result['step']} loss={result['loss']:.6f}"
            )
            console.print(f"Run directory: {result['run_dir']}")
    except Exception as exc:
        emit_cli_error(str(exc), output_json=output_json, exit_code=2)


@train_app.command("status")
def train_status(
    run_dir: Path | None = typer.Option(None, "--run-dir", help="Specific training run"),
    output_dir: Path = typer.Option(Path("./outputs"), "--output-dir", help="Training output root"),
    output_json: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Show persisted status for the latest or selected training run."""
    try:
        selected, result = read_training_run_status(
            run_dir=run_dir,
            output_dir=output_dir,
        )
    except TrainingRuntimeError as exc:
        if run_dir is None and str(exc).startswith("No training runs found under:"):
            result = {
                "status": "no_runs",
                "runs": [],
                "output_dir": str(output_dir.expanduser().resolve()),
            }
            if output_json:
                emit_json(result)
            else:
                console.print(f"[yellow]No training runs found under {result['output_dir']}[/yellow]")
            return
        emit_cli_error(str(exc), output_json=output_json, exit_code=2)
    except OSError as exc:
        emit_cli_error(str(exc), output_json=output_json, exit_code=2)

    if output_json:
        emit_json(result)
        return

    table = Table(title="Production Training Status")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    for key in ("status", "step", "max_steps", "loss", "phase", "eta_seconds", "pid", "process_running", "run_dir"):
        table.add_row(key, str(result.get(key)))
    console.print(table)


@train_app.command("stop")
def train_stop(
    run_dir: Path | None = typer.Option(None, "--run-dir", help="Specific training run"),
    output_dir: Path = typer.Option(Path("./outputs"), "--output-dir", help="Training output root"),
    timeout: float = typer.Option(30.0, "--timeout", min=0.1, help="Seconds to await checkpoint exit"),
    output_json: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Request a detached run to checkpoint and stop via SIGINT."""
    try:
        selected = resolve_training_run_dir(run_dir, output_dir)
        state = read_heartbeat(selected)
        pid, start_time = read_training_process_record(selected, state)
        if not process_is_running(pid):
            message = f"Training process is not running for {selected}"
            emit_cli_error(message, output_json=output_json, exit_code=1)

        assert pid is not None
        latest_state = read_heartbeat(selected)
        latest_pid, latest_start_time = read_training_process_record(selected, latest_state)
        if latest_pid != pid or latest_state.get("status") not in {"launching", "starting", "running", "stopping"}:
            emit_cli_error(
                f"Training process is already stopped or changed for {selected}",
                output_json=output_json,
                exit_code=1,
            )
        try:
            _send_training_sigint(pid, latest_start_time or start_time)
        except ProcessLookupError:
            final_state = read_heartbeat(selected)
            emit_cli_error(
                f"Training process is already stopped for {selected} "
                f"(last status: {final_state.get('status', 'unknown')})",
                output_json=output_json,
                exit_code=1,
            )
        except TrainingRuntimeError as exc:
            emit_cli_error(str(exc), output_json=output_json, exit_code=1)
        deadline = time.monotonic() + timeout
        while process_is_running(pid) and time.monotonic() < deadline:
            time.sleep(0.1)
        if process_is_running(pid):
            emit_cli_error(
                f"Training process {pid} did not stop within {timeout:.1f}s",
                output_json=output_json,
                exit_code=1,
            )

        final_state = read_heartbeat(selected)
        result = {
            "status": final_state.get("status", "stopped"),
            "pid": pid,
            "run_dir": str(selected),
            "step": final_state.get("step"),
            "checkpoint_dir": final_state.get("checkpoint_dir"),
        }
        if output_json:
            emit_json(result)
        else:
            console.print(f"[green]Training stopped[/green] at step {result['step']}")
            console.print(f"Run directory: {selected}")
    except typer.Exit:
        raise
    except (OSError, TrainingRuntimeError) as exc:
        emit_cli_error(str(exc), output_json=output_json, exit_code=2)


__all__ = ["train_app", "train_start", "train_status", "train_stop"]
