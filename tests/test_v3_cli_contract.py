"""Behavioral contracts for the truthful v3 CLI runtime."""

from __future__ import annotations

import json
import math
import threading
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typer.testing import CliRunner

from forge import __version__
from forge.cli_commands.json_output import SANITIZED_NOTE, json_payload
from forge.cli_v2 import app
from forge.config import ForgeConfig
from forge.trainer import ProductionTrainer
from forge.training_runtime import atomic_write_json, choose_batch_size


class _TinyStudent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bridge = nn.Linear(12, 8)
        self.action_head = nn.Linear(8, 3)
        self.lora_layer = nn.Linear(8, 8)

    def forward(self, images, gt_actions=None):
        features = self.bridge(images.flatten(1))
        return {"actions": self.action_head(features), "vision_features": features}


class _TinyDataset(Dataset):
    def __init__(self, size: int = 8) -> None:
        self.images = torch.randn(size, 3, 2, 2)
        self.actions = torch.randn(size, 3)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        action = self.actions[index]
        return {
            "image": self.images[index],
            "ground_truth_actions": action,
            "teacher_action_logits": action,
            "teacher_action_mean": action,
            "teacher_action_std": torch.ones_like(action),
            "confidence": torch.ones_like(action),
        }


class _TinyLoss(nn.Module):
    def forward(self, student_actions, ground_truth_actions, **_kwargs):
        loss = nn.functional.mse_loss(student_actions, ground_truth_actions)
        return {"total": loss, "kd": loss, "task": loss}


def test_runtime_device_preserves_explicit_cuda_index(monkeypatch: pytest.MonkeyPatch) -> None:
    from forge.cli_commands.shared import resolve_runtime_device
    from forge.eval.model_server import ForgeModelServer
    from forge.serve import _resolve_runtime_device as resolve_serve_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)

    assert resolve_runtime_device("cuda:3", strict=True) == "cuda:3"
    assert ForgeModelServer("missing.pt", device="cuda:2").config.device == "cuda:2"
    assert resolve_serve_device("cuda:1") == "cuda:1"


@pytest.mark.parametrize("requested", ["cuda:4", "cuda:999", "cuda:-1", "cuda-not-a-device"])
def test_runtime_device_rejects_invalid_cuda_selection(
    requested: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from forge.cli_commands.shared import resolve_runtime_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)

    with pytest.raises(ValueError):
        resolve_runtime_device(requested, strict=True)


def _tiny_factory(config, *, device, run_dir, batch_size):
    config.distill.batch_size = batch_size or 2
    config.distill.gradient_accumulation_steps = 1
    config.distill.warmup_steps = 0
    config.distill.save_every = 1
    config.distill.learning_rate = 1e-3
    config.curriculum.enabled = False
    config.curriculum.plateau_window = 0
    config.curriculum.hard_example_mining = False
    trainer = ProductionTrainer(
        student=_TinyStudent(),
        dataset=_TinyDataset(),
        loss_fn=_TinyLoss(),
        config=config,
        device=device,
        checkpoint_dir=str(run_dir),
    )
    return (
        trainer,
        Path(run_dir) / "labels",
        {
            "source": "explicit",
            "batch_size": config.distill.batch_size,
        },
    )


def test_json_payload_replaces_non_finite_values() -> None:
    payload = json.loads(json_payload({"best": math.inf, "losses": [math.nan, -math.inf, 1.5]}))
    assert payload["best"] is None
    assert payload["losses"] == [None, None, 1.5]
    assert payload["note"] == SANITIZED_NOTE


def test_root_version_option() -> None:
    result = CliRunner().invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == __version__
    assert result.stderr == ""


def test_atomic_heartbeat_replaces_non_finite_values(tmp_path: Path) -> None:
    destination = atomic_write_json(
        tmp_path / "train_state.json",
        {"status": "running", "loss": math.nan, "best_loss": math.inf},
    )
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["loss"] is None
    assert payload["best_loss"] is None
    assert payload["note"] == SANITIZED_NOTE
    assert not list(tmp_path.glob("*.tmp"))


def test_training_pidfile_round_trip_and_identity(tmp_path: Path, monkeypatch) -> None:
    from forge.cli_commands import train_v3
    from forge.training_status import read_training_process_record

    monkeypatch.setattr(train_v3, "process_start_time_ticks", lambda pid: 456 if pid == 123 else None)
    train_v3._write_pidfile(tmp_path, 123)

    payload = json.loads((tmp_path / "train.pid").read_text(encoding="utf-8"))
    assert payload == {"pid": 123, "process_start_time_ticks": 456}
    assert read_training_process_record(tmp_path, {}) == (123, 456)


def test_detached_parent_does_not_overwrite_child_heartbeat(tmp_path: Path, monkeypatch) -> None:
    from forge.cli_commands import train_v3
    from forge.cli_v2 import train_app

    child_state = {
        "status": "running",
        "pid": 321,
        "process_start_time_ticks": 654,
        "step": 7,
        "loss": 1.25,
    }

    def launch(_command, run_dir):
        atomic_write_json(run_dir / "train_state.json", child_state)
        atomic_write_json(
            run_dir / "train.pid",
            {"pid": 321, "process_start_time_ticks": 654},
        )
        return 321

    monkeypatch.setattr(train_v3, "_launch_detached", launch)
    output_root = tmp_path / "outputs"
    result = CliRunner().invoke(
        train_app,
        [
            "start",
            "--detach",
            "--json",
            "--device",
            "cpu",
            "--max-steps",
            "1",
            "--output-dir",
            str(output_root),
        ],
    )

    assert result.exit_code == 0, result.output
    response = json.loads(result.stdout)
    persisted = json.loads((Path(response["run_dir"]) / "train_state.json").read_text(encoding="utf-8"))
    assert persisted == child_state


def test_detached_launch_terminates_child_when_pidfile_write_fails(tmp_path: Path, monkeypatch) -> None:
    from forge.cli_commands import train_v3

    class Process:
        pid = 123
        terminated = False

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            assert timeout == 5
            return 0

    process = Process()
    monkeypatch.setattr(train_v3.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(
        train_v3,
        "_write_pidfile",
        lambda *_args: (_ for _ in ()).throw(OSError("disk full")),
    )

    with pytest.raises(OSError, match="disk full"):
        train_v3._launch_detached(["forge", "train"], tmp_path)

    assert process.terminated


def test_cpu_batch_selection_uses_config_and_dataset_limit() -> None:
    config = ForgeConfig.default()
    config.distill.batch_size = 16
    batch_size, details = choose_batch_size(
        config,
        device="cpu",
        requested=None,
        dataset_size=5,
    )
    assert batch_size == 5
    assert details == {"source": "config_cpu", "batch_size": 5}


def test_production_trainer_progress_and_clean_stop(tmp_path: Path) -> None:
    config = ForgeConfig.default()
    config.distill.batch_size = 2
    config.distill.gradient_accumulation_steps = 1
    config.distill.warmup_steps = 0
    config.curriculum.enabled = False
    config.curriculum.plateau_window = 0
    trainer = ProductionTrainer(
        student=_TinyStudent(),
        dataset=_TinyDataset(),
        loss_fn=_TinyLoss(),
        config=config,
        checkpoint_dir=str(tmp_path),
    )
    progress: list[dict] = []
    report = trainer.train(
        max_steps=10,
        checkpoint_every=20,
        progress_callback=progress.append,
        stop_requested=lambda: len(progress) >= 2,
    )
    assert report.status == "stopped"
    assert report.total_steps == 2
    assert progress[-1]["step"] == 2
    assert math.isfinite(progress[-1]["loss"])
    assert (tmp_path / "checkpoints" / "production" / "stopped.pt").is_file()
    assert not (tmp_path / "checkpoints" / "production" / "final.pt").exists()


def test_train_start_runs_trainer_writes_checkpoint_and_status(tmp_path: Path, monkeypatch) -> None:
    from forge.cli_commands import train_v3
    from forge.cli_v2 import train_app

    monkeypatch.setattr(train_v3, "build_production_trainer", _tiny_factory)
    runner = CliRunner()
    output_root = tmp_path / "outputs"
    result = runner.invoke(
        train_app,
        [
            "start",
            "--json",
            "--device",
            "cpu",
            "--max-steps",
            "3",
            "--batch-size",
            "2",
            "--output-dir",
            str(output_root),
            "--heartbeat-every",
            "1",
            "--log-every",
            "1",
            "--checkpoint-every",
            "2",
            "--no-curriculum",
            "--no-plateau",
            "--no-hard-mining",
        ],
    )
    assert result.exit_code == 0, result.output
    response = json.loads(result.stdout)
    assert response["status"] == "completed"
    assert response["step"] == 3
    assert math.isfinite(response["loss"])

    run_dir = Path(response["run_dir"])
    checkpoint_dir = run_dir / "checkpoints" / "production"
    assert (checkpoint_dir / "step_2.pt").is_file()
    assert (checkpoint_dir / "final.pt").is_file()
    heartbeat = json.loads((run_dir / "train_state.json").read_text(encoding="utf-8"))
    assert heartbeat["status"] == "completed"
    assert heartbeat["step"] == 3
    assert heartbeat["loss"] == response["loss"]

    status_result = runner.invoke(
        train_app,
        ["status", "--json", "--run-dir", str(run_dir)],
    )
    assert status_result.exit_code == 0, status_result.output
    status = json.loads(status_result.stdout)
    assert status["status"] == "completed"
    assert status["step"] == 3
    assert status["run_dir"] == str(run_dir)


def test_training_runner_works_off_main_thread_without_signal_registration(tmp_path: Path, monkeypatch) -> None:
    from forge.cli_commands import train_v3

    monkeypatch.setattr(train_v3, "build_production_trainer", _tiny_factory)
    monkeypatch.setattr(
        train_v3.signal,
        "signal",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("signal handler registered")),
    )
    config = ForgeConfig.default()
    config.curriculum.enabled = False
    config.curriculum.plateau_window = 0
    result: dict[str, object] = {}
    errors: list[BaseException] = []

    def run() -> None:
        try:
            result.update(
                train_v3._run_training(
                    config=config,
                    config_path=None,
                    device="cpu",
                    run_dir=tmp_path,
                    max_steps=1,
                    batch_size=1,
                    heartbeat_every=1,
                    log_every=1,
                    checkpoint_every=None,
                    output_json=False,
                )
            )
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=run)
    worker.start()
    worker.join(timeout=10)

    assert not worker.is_alive()
    assert errors == []
    assert result["status"] == "completed"


def test_training_runner_closes_dataset_after_failure(tmp_path: Path, monkeypatch) -> None:
    from forge.cli_commands import train_v3

    class Dataset:
        closed = False

        def close(self):
            self.closed = True

    dataset = Dataset()

    class Trainer:
        def __init__(self):
            self.dataset = dataset

        def get_status(self):
            return {}

        def train(self, **_kwargs):
            raise RuntimeError("training failed")

    def factory(*_args, **_kwargs):
        return Trainer(), tmp_path / "labels", {"source": "explicit", "batch_size": 1}

    monkeypatch.setattr(train_v3, "build_production_trainer", factory)
    config = ForgeConfig.default()
    config.curriculum.enabled = False
    config.curriculum.plateau_window = 0

    with pytest.raises(RuntimeError, match="training failed"):
        train_v3._run_training(
            config=config,
            config_path=None,
            device="cpu",
            run_dir=tmp_path,
            max_steps=1,
            batch_size=1,
            heartbeat_every=1,
            log_every=1,
            checkpoint_every=None,
            output_json=False,
        )

    assert dataset.closed


def test_train_stop_handles_process_exit_between_check_and_signal(tmp_path: Path, monkeypatch) -> None:
    from forge.cli_commands import train_v3
    from forge.cli_v2 import train_app

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    atomic_write_json(
        run_dir / "train_state.json",
        {
            "status": "running",
            "pid": 1234,
            "process_start_time_ticks": 99,
            "step": 5,
        },
    )
    monkeypatch.setattr(train_v3, "process_is_running", lambda _pid: True)
    monkeypatch.setattr(
        train_v3,
        "_send_training_sigint",
        lambda *_args: (_ for _ in ()).throw(ProcessLookupError()),
    )

    invocation = CliRunner().invoke(
        train_app,
        ["stop", "--run-dir", str(run_dir), "--json"],
    )

    assert invocation.exit_code == 1
    assert invocation.stdout == ""
    assert "already stopped" in json.loads(invocation.stderr)["error"]


def test_train_stop_refuses_reused_pid_identity(tmp_path: Path, monkeypatch) -> None:
    from forge.cli_commands import train_v3
    from forge.cli_v2 import train_app

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    atomic_write_json(
        run_dir / "train_state.json",
        {
            "status": "running",
            "pid": 1234,
            "process_start_time_ticks": 99,
            "step": 5,
        },
    )
    monkeypatch.setattr(train_v3, "process_is_running", lambda _pid: True)
    monkeypatch.setattr(
        train_v3,
        "_send_training_sigint",
        lambda *_args: (_ for _ in ()).throw(
            train_v3.TrainingRuntimeError(
                "Refusing to signal PID 1234: its process identity does not match the training heartbeat"
            )
        ),
    )

    invocation = CliRunner().invoke(
        train_app,
        ["stop", "--run-dir", str(run_dir), "--json"],
    )

    assert invocation.exit_code == 1
    assert invocation.stdout == ""
    assert "identity does not match" in json.loads(invocation.stderr)["error"]
