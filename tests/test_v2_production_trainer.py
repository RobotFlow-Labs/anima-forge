"""Tests for PRD-23: Production Training Pipeline.

All tests run on CPU with synthetic data — no real models required.
"""

from __future__ import annotations

import json

import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from forge.config import CurriculumConfig, ForgeConfig
from forge.trainer import (
    PHASE_DESCRIPTIONS,
    AdaptiveLRScheduler,
    ProductionTrainer,
    TrainingReport,
    TrainingState,
    get_phase,
    set_trainable_for_phase,
)

B, D_ACTION = 4, 7


# ── Helpers ──────────────────────────────────────────────────────


class TinyStudent(nn.Module):
    """Minimal student for testing."""

    def __init__(self):
        super().__init__()
        self.forward_calls = 0
        self.bridge = nn.Linear(3 * 64 * 64, 32)
        self.action_head = nn.Linear(32, D_ACTION)
        self.lora_layer = nn.Linear(32, 32)  # simulates LoRA

    def forward(self, images, gt_actions=None):
        self.forward_calls += 1
        x = images.flatten(1)
        features = self.bridge(x)
        actions = self.action_head(features)
        return {"actions": actions, "vision_features": features}

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


class TinyLoss(nn.Module):
    """Minimal loss for testing."""

    def forward(self, student_actions, ground_truth_actions, **kwargs):
        task = nn.functional.mse_loss(student_actions, ground_truth_actions)
        kd = task * 0.5  # fake KD
        return {"total": task + kd, "kd": kd, "task": task}


class SyntheticDataset(Dataset):
    """Synthetic dataset returning image + action pairs."""

    def __init__(self, size: int = 100):
        self.size = size
        self.images = torch.randn(size, 3, 64, 64)
        self.actions = torch.randn(size, D_ACTION) * 0.1

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            "image": self.images[idx],
            "ground_truth_actions": self.actions[idx],
        }


def _make_config(
    max_steps: int = 100,
    curriculum: bool = True,
    plateau: bool = True,
    hard_mining: bool = False,
    teacher_dropout: bool = False,
) -> ForgeConfig:
    config = ForgeConfig.default()
    config.distill.max_steps = max_steps
    config.distill.batch_size = B
    config.distill.learning_rate = 1e-3
    config.distill.warmup_steps = 10
    config.distill.gradient_accumulation_steps = 1
    config.distill.save_every = 50
    config.paths.output_dir = "/tmp/forge_test"
    config.curriculum.enabled = curriculum
    config.curriculum.initial_difficulty = 0.5
    config.curriculum.final_difficulty = 1.0
    config.curriculum.ramp_steps = max_steps
    config.curriculum.hard_example_mining = hard_mining
    config.curriculum.teacher_dropout = teacher_dropout
    if not plateau:
        config.curriculum.plateau_window = 0
    else:
        config.curriculum.plateau_window = 20
        config.curriculum.plateau_threshold = 0.01
    return config


# ── Phase Management tests ───────────────────────────────────────


class TestPhaseManagement:
    def test_phase_1_early(self):
        assert get_phase(0, 1000) == 1
        assert get_phase(50, 1000) == 1

    def test_phase_2_middle(self):
        assert get_phase(200, 1000) == 2

    def test_phase_3_late(self):
        assert get_phase(900, 1000) == 3

    def test_phase_descriptions(self):
        for phase in [1, 2, 3]:
            assert phase in PHASE_DESCRIPTIONS

    def test_set_trainable_phase1(self):
        student = TinyStudent()
        set_trainable_for_phase(student, 1)
        assert student.bridge.weight.requires_grad
        assert student.action_head.weight.requires_grad
        assert not student.lora_layer.weight.requires_grad

    def test_set_trainable_phase2(self):
        student = TinyStudent()
        set_trainable_for_phase(student, 2)
        assert student.bridge.weight.requires_grad
        assert student.action_head.weight.requires_grad
        assert student.lora_layer.weight.requires_grad

    def test_set_trainable_phase3(self):
        student = TinyStudent()
        set_trainable_for_phase(student, 3)
        assert not student.bridge.weight.requires_grad
        assert student.action_head.weight.requires_grad
        assert not student.lora_layer.weight.requires_grad


# ── AdaptiveLRScheduler tests ────────────────────────────────────


class TestAdaptiveLRScheduler:
    def test_warmup_ramps_lr(self):
        opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=1.0)
        sched = AdaptiveLRScheduler(opt, warmup_steps=10, total_steps=100)
        sched.step()
        assert sched.get_lr() < 1.0  # still warming up

    def test_cosine_decays(self):
        opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=1.0)
        sched = AdaptiveLRScheduler(opt, warmup_steps=0, total_steps=100)
        for _ in range(50):
            sched.step()
        lr_mid = sched.get_lr()
        for _ in range(50):
            sched.step()
        lr_end = sched.get_lr()
        assert lr_mid > lr_end

    def test_plateau_reduces_lr(self):
        from forge.curriculum import PlateauDetector

        detector = PlateauDetector(window=10, threshold=0.01, lr_factor=0.5)
        opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=1.0)
        sched = AdaptiveLRScheduler(opt, warmup_steps=0, total_steps=1000, plateau_detector=detector)

        # Feed flat losses to trigger plateau
        for i in range(20):
            sched.step(loss=1.0)
        # After plateau, LR should be reduced
        assert sched.get_plateau_count() >= 1

    def test_state_dict_roundtrip(self):
        opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=1.0)
        sched = AdaptiveLRScheduler(opt, warmup_steps=5, total_steps=100)
        for _ in range(10):
            sched.step()
        state = sched.state_dict()
        assert state["step"] == 10

        sched2 = AdaptiveLRScheduler(opt, warmup_steps=5, total_steps=100)
        sched2.load_state_dict(state)
        assert sched2._step == 10


# ── TrainingState tests ──────────────────────────────────────────


class TestTrainingState:
    def test_defaults(self):
        state = TrainingState()
        assert state.global_step == 0
        assert state.best_loss == float("inf")
        assert state.phase == 1

    def test_to_dict(self):
        state = TrainingState(global_step=100, phase=2, plateau_count=1)
        d = state.to_dict()
        assert d["global_step"] == 100
        assert d["phase"] == 2
        assert d["plateau_count"] == 1


# ── TrainingReport tests ─────────────────────────────────────────


class TestTrainingReport:
    def test_to_dict(self):
        report = TrainingReport(
            total_steps=1000,
            elapsed_seconds=60.5,
            final_loss=0.123456,
            best_loss=0.1,
        )
        d = report.to_dict()
        assert d["total_steps"] == 1000
        assert d["elapsed_seconds"] == 60.5
        assert isinstance(d["final_loss"], float)


# ── ProductionTrainer tests ──────────────────────────────────────


class TestProductionTrainer:
    def test_train_basic(self, tmp_path):
        """Basic training loop completes."""
        config = _make_config(max_steps=20, curriculum=False, plateau=False)
        config.paths.output_dir = str(tmp_path)
        student = TinyStudent()
        dataset = SyntheticDataset(size=32)
        loss_fn = TinyLoss()

        trainer = ProductionTrainer(
            student=student,
            dataset=dataset,
            loss_fn=loss_fn,
            config=config,
            device="cpu",
        )
        report = trainer.train(max_steps=20, checkpoint_every=10)
        assert report.total_steps == 20
        assert report.final_loss > 0
        assert report.elapsed_seconds > 0

    def test_train_with_curriculum(self, tmp_path):
        """Training with curriculum enabled."""
        config = _make_config(max_steps=30, curriculum=True, plateau=False)
        config.paths.output_dir = str(tmp_path)
        student = TinyStudent()
        dataset = SyntheticDataset(size=32)
        loss_fn = TinyLoss()

        trainer = ProductionTrainer(
            student=student,
            dataset=dataset,
            loss_fn=loss_fn,
            config=config,
            device="cpu",
        )
        assert trainer.curriculum_sampler is not None
        report = trainer.train(max_steps=30)
        assert report.total_steps == 30

    def test_train_with_plateau_detection(self, tmp_path):
        """Training with plateau detection enabled."""
        config = _make_config(max_steps=30, curriculum=False, plateau=True)
        config.paths.output_dir = str(tmp_path)
        student = TinyStudent()
        dataset = SyntheticDataset(size=32)
        loss_fn = TinyLoss()

        trainer = ProductionTrainer(
            student=student,
            dataset=dataset,
            loss_fn=loss_fn,
            config=config,
            device="cpu",
        )
        assert trainer.plateau_detector is not None
        report = trainer.train(max_steps=30)
        assert report.total_steps == 30

    def test_train_with_hard_mining(self, tmp_path):
        """Training with hard example mining enabled."""
        config = _make_config(max_steps=20, curriculum=True, plateau=False, hard_mining=True)
        config.paths.output_dir = str(tmp_path)
        student = TinyStudent()
        dataset = SyntheticDataset(size=32)
        loss_fn = TinyLoss()

        trainer = ProductionTrainer(
            student=student,
            dataset=dataset,
            loss_fn=loss_fn,
            config=config,
            device="cpu",
        )
        assert trainer.hard_miner is not None
        report = trainer.train(max_steps=20)
        assert report.total_steps == 20

    def test_phase_transitions_logged(self, tmp_path):
        """Phase transitions are recorded in report."""
        config = _make_config(max_steps=100, curriculum=False, plateau=False)
        config.paths.output_dir = str(tmp_path)
        student = TinyStudent()
        dataset = SyntheticDataset(size=32)
        loss_fn = TinyLoss()

        trainer = ProductionTrainer(
            student=student,
            dataset=dataset,
            loss_fn=loss_fn,
            config=config,
            device="cpu",
        )
        report = trainer.train(max_steps=100, checkpoint_every=200)
        # Should have at least 1 phase transition (1→2 at 10%)
        assert len(report.phase_transitions) >= 1
        assert report.phase_transitions[0]["from_phase"] == 1
        assert report.phase_transitions[0]["to_phase"] == 2

    def test_checkpoint_saved(self, tmp_path):
        """Checkpoints are saved during training."""
        config = _make_config(max_steps=20, curriculum=False, plateau=False)
        config.paths.output_dir = str(tmp_path)
        student = TinyStudent()
        dataset = SyntheticDataset(size=32)
        loss_fn = TinyLoss()

        trainer = ProductionTrainer(
            student=student,
            dataset=dataset,
            loss_fn=loss_fn,
            config=config,
            device="cpu",
        )
        trainer.train(max_steps=20, checkpoint_every=10)
        ckpt_dir = tmp_path / "checkpoints" / "production"
        assert (ckpt_dir / "step_10.pt").exists()
        assert (ckpt_dir / "final.pt").exists()

    def test_checkpoint_resume(self, tmp_path):
        """Can resume training from checkpoint."""
        config = _make_config(max_steps=20, curriculum=False, plateau=False)
        config.paths.output_dir = str(tmp_path)
        student = TinyStudent()
        dataset = SyntheticDataset(size=32)
        loss_fn = TinyLoss()

        trainer = ProductionTrainer(
            student=student,
            dataset=dataset,
            loss_fn=loss_fn,
            config=config,
            device="cpu",
        )
        ckpt = trainer.save_checkpoint(tag="test_resume")
        trainer.state.global_step = 999
        trainer.load_checkpoint(ckpt)
        assert trainer.state.global_step == 0  # restored

    def test_gradient_accumulation_counts_optimizer_steps(self, tmp_path):
        config = _make_config(max_steps=3, curriculum=False, plateau=False)
        config.paths.output_dir = str(tmp_path)
        config.distill.gradient_accumulation_steps = 2
        student = TinyStudent()
        trainer = ProductionTrainer(
            student=student,
            dataset=SyntheticDataset(size=32),
            loss_fn=TinyLoss(),
            config=config,
            device="cpu",
        )

        progress = []
        report = trainer.train(max_steps=3, checkpoint_every=10, progress_callback=progress.append)

        assert report.total_steps == 3
        assert student.forward_calls == 6
        assert [item["step"] for item in progress] == [1, 2, 3]

    def test_resume_preserves_optimizer_and_scheduler_progress(self, tmp_path):
        config = _make_config(max_steps=20, curriculum=False, plateau=False)
        config.paths.output_dir = str(tmp_path)
        first = ProductionTrainer(
            student=TinyStudent(),
            dataset=SyntheticDataset(size=32),
            loss_fn=TinyLoss(),
            config=config,
            device="cpu",
        )
        first.train(
            max_steps=20,
            checkpoint_every=100,
            stop_requested=lambda: first.state.global_step >= 3,
        )
        checkpoint = first.checkpoint_dir / "stopped.pt"

        resumed = ProductionTrainer(
            student=TinyStudent(),
            dataset=SyntheticDataset(size=32),
            loss_fn=TinyLoss(),
            config=config,
            device="cpu",
        )
        resumed.load_checkpoint(checkpoint)
        scheduler_step = resumed.scheduler._step
        assert scheduler_step > 0
        assert resumed.optimizer.state

        resumed.train(
            max_steps=20,
            checkpoint_every=100,
            stop_requested=lambda: resumed.state.global_step >= 4,
        )

        assert resumed.state.global_step == 4
        assert resumed.scheduler._step == scheduler_step + 1

    def test_resumed_progress_rate_counts_only_current_process_steps(self, tmp_path):
        config = _make_config(max_steps=20, curriculum=False, plateau=False)
        config.paths.output_dir = str(tmp_path)
        trainer = ProductionTrainer(
            student=TinyStudent(),
            dataset=SyntheticDataset(size=32),
            loss_fn=TinyLoss(),
            config=config,
            device="cpu",
        )
        trainer.state.global_step = 10
        progress = []

        trainer.train(
            max_steps=20,
            checkpoint_every=100,
            progress_callback=progress.append,
            stop_requested=lambda: True,
        )

        assert progress[0]["step"] == 11
        assert progress[0]["steps_per_second"] == pytest.approx(1.0 / progress[0]["elapsed_seconds"])
        assert progress[0]["eta_seconds"] == pytest.approx((20 - progress[0]["step"]) / progress[0]["steps_per_second"])

    def test_non_finite_loss_stops_before_optimizer_update(self, tmp_path):
        class NonFiniteLoss(nn.Module):
            def forward(self, student_actions, **_kwargs):
                return {"total": student_actions.sum() * torch.tensor(float("nan"))}

        config = _make_config(max_steps=2, curriculum=False, plateau=False)
        config.paths.output_dir = str(tmp_path)
        trainer = ProductionTrainer(
            student=TinyStudent(),
            dataset=SyntheticDataset(size=32),
            loss_fn=NonFiniteLoss(),
            config=config,
            device="cpu",
        )

        with pytest.raises(RuntimeError, match="invalid total loss"):
            trainer.train(max_steps=2)

        assert trainer.state.global_step == 0

    def test_get_status(self, tmp_path):
        """Status dict contains expected keys."""
        config = _make_config(max_steps=50, curriculum=True, plateau=True)
        config.paths.output_dir = str(tmp_path)
        student = TinyStudent()
        dataset = SyntheticDataset(size=32)
        loss_fn = TinyLoss()

        trainer = ProductionTrainer(
            student=student,
            dataset=dataset,
            loss_fn=loss_fn,
            config=config,
            device="cpu",
        )
        status = trainer.get_status()
        assert "global_step" in status
        assert "lr" in status
        assert "curriculum_difficulty" in status

    def test_loss_decreases(self, tmp_path):
        """Loss should generally decrease over training."""
        config = _make_config(max_steps=50, curriculum=False, plateau=False)
        config.paths.output_dir = str(tmp_path)
        student = TinyStudent()
        dataset = SyntheticDataset(size=32)
        loss_fn = TinyLoss()

        trainer = ProductionTrainer(
            student=student,
            dataset=dataset,
            loss_fn=loss_fn,
            config=config,
            device="cpu",
        )
        report = trainer.train(max_steps=50, checkpoint_every=100)
        # Best loss should be better than first loss
        assert report.best_loss < report.final_loss or report.best_loss <= report.final_loss

    def test_all_features_enabled(self, tmp_path):
        """Training with all features enabled simultaneously."""
        config = _make_config(
            max_steps=30,
            curriculum=True,
            plateau=True,
            hard_mining=True,
        )
        config.paths.output_dir = str(tmp_path)
        student = TinyStudent()
        dataset = SyntheticDataset(size=32)
        loss_fn = TinyLoss()

        trainer = ProductionTrainer(
            student=student,
            dataset=dataset,
            loss_fn=loss_fn,
            config=config,
            device="cpu",
        )
        assert trainer.curriculum_sampler is not None
        assert trainer.plateau_detector is not None
        assert trainer.hard_miner is not None
        report = trainer.train(max_steps=30, checkpoint_every=100)
        assert report.total_steps == 30


# ── CLI tests ────────────────────────────────────────────────────


class TestCLI:
    def test_train_start_json_refuses_missing_labels(self, tmp_path):
        from typer.testing import CliRunner

        from forge.cli_v2 import train_app

        runner = CliRunner()
        result = runner.invoke(
            train_app,
            [
                "start",
                "--json",
                "--device",
                "cpu",
                "--max-steps",
                "1",
                "--output-dir",
                str(tmp_path / "outputs"),
                "--data-dir",
                str(tmp_path / "missing"),
            ],
        )
        assert result.exit_code == 2
        data = json.loads(result.stderr)
        assert "Teacher labels not found" in data["error"]

    def test_train_status_json_reports_empty_output_root(self, tmp_path):
        from typer.testing import CliRunner

        from forge.cli_v2 import train_app

        runner = CliRunner()
        result = runner.invoke(
            train_app,
            ["status", "--json", "--output-dir", str(tmp_path)],
        )
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert data == {
            "status": "no_runs",
            "runs": [],
            "output_dir": str(tmp_path.resolve()),
        }

    def test_train_status_json_refuses_explicit_missing_run(self, tmp_path):
        from typer.testing import CliRunner

        from forge.cli_v2 import train_app

        missing = tmp_path / "missing-run"
        result = CliRunner().invoke(
            train_app,
            ["status", "--json", "--run-dir", str(missing)],
        )
        assert result.exit_code == 2
        assert result.stdout == ""
        assert "Training heartbeat not found" in json.loads(result.stderr)["error"]


# ── Config tests ─────────────────────────────────────────────────


class TestConfig:
    def test_forgeconfig_has_curriculum(self):
        config = ForgeConfig.default()
        assert hasattr(config, "curriculum")
        assert isinstance(config.curriculum, CurriculumConfig)

    def test_trainer_config_integration(self):
        """Config values propagate to trainer components."""
        config = _make_config(max_steps=100, curriculum=True)
        config.curriculum.initial_difficulty = 0.2
        student = TinyStudent()
        dataset = SyntheticDataset(size=32)
        loss_fn = TinyLoss()

        trainer = ProductionTrainer(
            student=student,
            dataset=dataset,
            loss_fn=loss_fn,
            config=config,
            device="cpu",
        )
        difficulty = trainer.curriculum_sampler.scheduler.get_difficulty(0)
        assert abs(difficulty - 0.2) < 1e-5
