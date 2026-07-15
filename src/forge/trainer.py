"""PRD-23: Production Training Pipeline.

Unified training orchestrator that integrates all FORGE v2 components:
1. CurriculumSampler — progressive difficulty ramping
2. PlateauDetector — auto LR reduction on loss stalls
3. TeacherDropout — progressive teacher dropping for robustness
4. HardExampleMiner — re-sampling high-loss examples
5. Phase management — bridge warmup → full KD → action fine-tune

Works with both single-teacher (ForgeDistillationLoss) and multi-teacher
(UniversalDistillationLoss) setups.

Usage:
    trainer = ProductionTrainer(config, device="cuda")
    result = trainer.train(max_steps=50000)
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Sized
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from forge.config import CurriculumConfig, ForgeConfig
from forge.curriculum import (
    CurriculumSampler,
    DifficultyScorer,
    HardExampleMiner,
    PlateauDetector,
    TeacherDropout,
)
from forge.provenance import build_provenance

logger = logging.getLogger(__name__)


# ── Training State ───────────────────────────────────────────────


@dataclass
class TrainingState:
    """Serializable snapshot of full training state."""

    global_step: int = 0
    best_loss: float = float("inf")
    phase: int = 1
    plateau_count: int = 0
    lr_multiplier: float = 1.0
    curriculum_step: int = 0
    epoch: int = 0

    # Tracked metrics (last N steps)
    loss_history: list[float] = field(default_factory=list)
    kd_loss_history: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "global_step": self.global_step,
            "best_loss": self.best_loss,
            "phase": self.phase,
            "plateau_count": self.plateau_count,
            "lr_multiplier": self.lr_multiplier,
            "curriculum_step": self.curriculum_step,
            "epoch": self.epoch,
        }


@dataclass
class TrainingReport:
    """Final training report with all metrics."""

    total_steps: int = 0
    elapsed_seconds: float = 0.0
    final_loss: float = 0.0
    best_loss: float = float("inf")
    final_lr: float = 0.0
    plateaus_detected: int = 0
    phase_transitions: list[dict[str, Any]] = field(default_factory=list)
    curriculum_stats: dict[str, Any] = field(default_factory=dict)
    checkpoint_dir: str = ""
    device: str = "cpu"
    status: str = "completed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_steps": self.total_steps,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "final_loss": round(self.final_loss, 6),
            "best_loss": round(self.best_loss, 6),
            "final_lr": self.final_lr,
            "plateaus_detected": self.plateaus_detected,
            "phase_transitions": self.phase_transitions,
            "curriculum_stats": self.curriculum_stats,
            "checkpoint_dir": self.checkpoint_dir,
            "device": self.device,
            "status": self.status,
        }


# ── Phase Management ─────────────────────────────────────────────


def get_phase(step: int, max_steps: int) -> int:
    """Determine training phase.

    Phase 1 (0-10%):  Bridge warmup — only bridge + action head
    Phase 2 (10-83%): Full distillation — bridge + LoRA + action head
    Phase 3 (83-100%): Action fine-tune — action head only
    """
    if step < max_steps * 0.1:
        return 1
    elif step < max_steps * 0.83:
        return 2
    else:
        return 3


def set_trainable_for_phase(student: nn.Module, phase: int) -> None:
    """Freeze/unfreeze parameters by phase."""
    for param in student.parameters():
        param.requires_grad = False

    if phase == 1:
        for name, param in student.named_parameters():
            if "bridge" in name or "action_head" in name:
                param.requires_grad = True
    elif phase == 2:
        for name, param in student.named_parameters():
            if "bridge" in name or "action_head" in name or "lora" in name.lower():
                param.requires_grad = True
    elif phase == 3:
        for name, param in student.named_parameters():
            if "action_head" in name:
                param.requires_grad = True


PHASE_DESCRIPTIONS = {
    1: "Bridge warmup (bridge + action head)",
    2: "Full distillation (bridge + LoRA + action head)",
    3: "Action fine-tune (action head only)",
}


# ── LR Scheduler with Plateau Support ────────────────────────────


class AdaptiveLRScheduler:
    """Cosine schedule with warmup + plateau-based reduction.

    Combines standard cosine annealing with PlateauDetector to auto-reduce
    LR when training stalls.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        plateau_detector: PlateauDetector | None = None,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.plateau_detector = plateau_detector
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self._step = 0

    def _cosine_factor(self, step: int) -> float:
        if step < self.warmup_steps:
            return step / max(1, self.warmup_steps)
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    def step(self, loss: float | None = None) -> None:
        """Advance scheduler. Pass loss for plateau detection."""
        self._step += 1
        cosine = self._cosine_factor(self._step)

        # Plateau multiplier
        plateau_mult = 1.0
        if self.plateau_detector is not None and loss is not None:
            self.plateau_detector.update(loss)
            self.plateau_detector.check_plateau(self._step)
            plateau_mult = self.plateau_detector.get_lr_multiplier()

        for base_lr, pg in zip(self.base_lrs, self.optimizer.param_groups):
            pg["lr"] = base_lr * cosine * plateau_mult

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def get_plateau_count(self) -> int:
        if self.plateau_detector is not None:
            return self.plateau_detector.plateau_count
        return 0

    def state_dict(self) -> dict:
        state = {
            "step": self._step,
            "base_lrs": self.base_lrs,
        }
        if self.plateau_detector is not None:
            state["plateau_count"] = self.plateau_detector.plateau_count
            state["plateau_last_check"] = self.plateau_detector._last_check_step
        return state

    def load_state_dict(self, state: dict) -> None:
        self._step = state["step"]
        self.base_lrs = state["base_lrs"]
        if self.plateau_detector is not None and "plateau_count" in state:
            self.plateau_detector.plateau_count = state["plateau_count"]
            self.plateau_detector._last_check_step = state["plateau_last_check"]


# ── Production Trainer ───────────────────────────────────────────


class ProductionTrainer:
    """Unified training orchestrator integrating all FORGE v2 components.

    Supports:
    - Single-teacher or multi-teacher (universal) distillation
    - Curriculum learning with difficulty ramping
    - Plateau detection with auto LR reduction
    - Teacher dropout for multi-teacher robustness
    - Hard example mining
    - Phase management (bridge warmup → full KD → action fine-tune)
    - Checkpointing with full state resume
    """

    def __init__(
        self,
        student: nn.Module,
        dataset: Dataset,
        loss_fn: nn.Module,
        config: ForgeConfig,
        *,
        device: str = "cpu",
        n_teachers: int = 1,
        checkpoint_dir: str | None = None,
    ):
        self.student = student.to(device)
        self.dataset = dataset
        self.loss_fn = loss_fn
        self.config = config
        self.device = device
        self.n_teachers = n_teachers
        self.checkpoint_dir = Path(checkpoint_dir or config.paths.output_dir) / "checkpoints" / "production"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Curriculum components
        self._init_curriculum(config.curriculum, len(cast(Sized, dataset)))

        # Training state and phase-specific optimizer. Initial checkpoints must
        # have the same parameter-group structure that a resume will rebuild.
        self.state = TrainingState()
        set_trainable_for_phase(self.student, self.state.phase)
        self.optimizer = AdamW(
            [p for p in self.student.parameters() if p.requires_grad],
            lr=config.distill.learning_rate,
            weight_decay=config.distill.weight_decay,
        )

        # LR Scheduler with plateau detection
        max_steps = config.distill.max_steps
        self.scheduler = AdaptiveLRScheduler(
            self.optimizer,
            warmup_steps=config.distill.warmup_steps,
            total_steps=max_steps,
            plateau_detector=self.plateau_detector,
        )
        self._resume_pending = False

    def _init_curriculum(self, cfg: CurriculumConfig, dataset_size: int) -> None:
        """Initialize curriculum learning components."""
        # Difficulty scorer
        self.difficulty_scorer = DifficultyScorer(metric=cfg.difficulty_metric)

        # Hard example miner
        self.hard_miner: HardExampleMiner | None = None
        if cfg.hard_example_mining:
            self.hard_miner = HardExampleMiner(
                dataset_size=dataset_size,
                hard_ratio=cfg.hard_example_ratio,
                history_size=cfg.loss_history_size,
            )

        # Curriculum sampler
        self.curriculum_sampler: CurriculumSampler | None = None
        if cfg.enabled:
            self.curriculum_sampler = CurriculumSampler(
                dataset_size=dataset_size,
                config=cfg,
            )

        # Plateau detector
        self.plateau_detector: PlateauDetector | None = None
        if cfg.plateau_window > 0:
            self.plateau_detector = PlateauDetector(
                window=cfg.plateau_window,
                threshold=cfg.plateau_threshold,
                lr_factor=cfg.plateau_lr_factor,
                max_plateaus=cfg.max_plateaus,
            )

        # Teacher dropout
        self.teacher_dropout: TeacherDropout | None = None
        if cfg.teacher_dropout and self.n_teachers > 1:
            self.teacher_dropout = TeacherDropout(
                n_teachers=self.n_teachers,
                dropout_start=cfg.teacher_dropout_start,
                dropout_end=cfg.teacher_dropout_end,
                ramp_steps=cfg.teacher_dropout_ramp_steps,
            )

    def _create_dataloader(self, batch_size: int) -> DataLoader:
        """Create DataLoader with optional curriculum sampler."""
        if self.curriculum_sampler is not None:
            self.curriculum_sampler.set_step(self.state.global_step)
            return DataLoader(
                self.dataset,
                batch_size=batch_size,
                sampler=self.curriculum_sampler,
                num_workers=0,
                drop_last=True,
            )
        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )

    def _get_teacher_mask(self) -> list[bool] | None:
        """Get teacher dropout mask for current step."""
        if self.teacher_dropout is None:
            return None
        return self.teacher_dropout.get_active_mask(self.state.global_step)

    def _update_hard_miner(
        self,
        indices: list[int] | torch.Tensor | None,
        losses: torch.Tensor,
    ) -> None:
        """Update hard example miner with per-sample losses."""
        if self.hard_miner is None or indices is None:
            return
        self.hard_miner.update_losses(indices, losses)

    def _update_curriculum_scores(self) -> None:
        """Refresh curriculum sampler with latest difficulty scores."""
        if (
            self.curriculum_sampler is not None
            and self.hard_miner is not None
            and self.hard_miner.update_count.sum() > 0
        ):
            scores = self.hard_miner.get_difficulty_scores()
            self.curriculum_sampler.update_difficulty_scores(scores)

    def train(
        self,
        max_steps: int | None = None,
        log_every: int = 100,
        checkpoint_every: int | None = None,
        eval_fn: Any | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        stop_requested: Callable[[], bool] | None = None,
    ) -> TrainingReport:
        """Run the full production training loop.

        Args:
            max_steps: Override config max_steps
            log_every: Log metrics every N steps
            checkpoint_every: Save checkpoint every N steps
            eval_fn: Optional callable(student, step) → eval_metrics dict
            progress_callback: Called after every completed optimizer step
            stop_requested: Called after each step; true requests a clean checkpoint stop

        Returns:
            TrainingReport with full metrics
        """
        max_steps = max_steps if max_steps is not None else self.config.distill.max_steps
        checkpoint_every = checkpoint_every if checkpoint_every is not None else self.config.distill.save_every
        batch_size = self.config.distill.batch_size
        grad_accum = self.config.distill.gradient_accumulation_steps
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if grad_accum < 1:
            raise ValueError("gradient_accumulation_steps must be positive")
        if log_every < 1:
            raise ValueError("log_every must be positive")
        if checkpoint_every < 1:
            raise ValueError("checkpoint_every must be positive")

        logger.info(
            f"ProductionTrainer: device={self.device}, "
            f"max_steps={max_steps}, batch_size={batch_size}, "
            f"curriculum={'ON' if self.curriculum_sampler else 'OFF'}, "
            f"plateau={'ON' if self.plateau_detector else 'OFF'}, "
            f"teacher_dropout={'ON' if self.teacher_dropout else 'OFF'}, "
            f"hard_mining={'ON' if self.hard_miner else 'OFF'}"
        )

        # Phase init
        expected_phase = get_phase(self.state.global_step, max_steps)
        if self._resume_pending and expected_phase != self.state.phase:
            raise ValueError(
                "Checkpoint phase is incompatible with the requested max_steps: "
                f"saved phase {self.state.phase}, expected phase {expected_phase} at step {self.state.global_step}"
            )
        self.state.phase = expected_phase
        set_trainable_for_phase(self.student, self.state.phase)
        logger.info(f"Phase {self.state.phase}: {PHASE_DESCRIPTIONS[self.state.phase]}")

        if self._resume_pending:
            self._resume_pending = False
        else:
            # Rebuild optimizer with the parameters trainable in the initial phase.
            self.optimizer = AdamW(
                [p for p in self.student.parameters() if p.requires_grad],
                lr=self.config.distill.learning_rate,
                weight_decay=self.config.distill.weight_decay,
            )
            self.scheduler = AdaptiveLRScheduler(
                self.optimizer,
                warmup_steps=self.config.distill.warmup_steps,
                total_steps=max_steps,
                plateau_detector=self.plateau_detector,
            )

        report = TrainingReport(device=self.device, checkpoint_dir=str(self.checkpoint_dir))
        initial_global_step = self.state.global_step
        t_start = time.time()
        stopped = False

        # DataLoader (rebuilt each epoch for curriculum updates)
        dataloader = self._create_dataloader(batch_size)
        data_iter = iter(dataloader)

        while self.state.global_step < max_steps:
            accumulated_loss = 0.0
            self.optimizer.zero_grad()
            for _micro_step in range(grad_accum):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    self.state.epoch += 1
                    # Update curriculum scores from miner at epoch boundary.
                    self._update_curriculum_scores()
                    dataloader = self._create_dataloader(batch_size)
                    data_iter = iter(dataloader)
                    try:
                        batch = next(data_iter)
                    except StopIteration as exc:
                        raise RuntimeError(
                            "Training DataLoader produced no batches; ensure dataset size is at least batch_size"
                        ) from exc

                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                teacher_mask = self._get_teacher_mask()
                loss_dict = self._forward_step(batch, teacher_mask)
                total_loss = loss_dict["total"]
                finite_loss = bool(torch.isfinite(total_loss).all())
                if total_loss.numel() != 1 or not finite_loss:
                    self.optimizer.zero_grad()
                    raise RuntimeError(
                        f"Training produced an invalid total loss at step {self.state.global_step}: "
                        f"shape={tuple(total_loss.shape)}, finite={finite_loss}"
                    )

                (total_loss / grad_accum).backward()
                accumulated_loss += total_loss.item()

                sample_indices = batch.get("sample_indices")
                if sample_indices is not None and "per_sample_loss" in loss_dict:
                    self._update_hard_miner(sample_indices, loss_dict["per_sample_loss"])

            torch.nn.utils.clip_grad_norm_(
                [p for p in self.student.parameters() if p.requires_grad],
                max_norm=1.0,
            )
            self.optimizer.step()
            loss_val = accumulated_loss / grad_accum
            self.scheduler.step(loss=loss_val)
            self.optimizer.zero_grad()
            self.state.global_step += 1

            # Track loss history
            self.state.loss_history.append(loss_val)
            if len(self.state.loss_history) > 1000:
                self.state.loss_history = self.state.loss_history[-500:]

            # Best loss tracking
            if loss_val < self.state.best_loss:
                self.state.best_loss = loss_val

            # Phase transition
            new_phase = get_phase(self.state.global_step, max_steps)
            if new_phase != self.state.phase:
                old_phase = self.state.phase
                self.state.phase = new_phase
                set_trainable_for_phase(self.student, new_phase)
                # Rebuild optimizer for new trainable params
                self.optimizer = AdamW(
                    [p for p in self.student.parameters() if p.requires_grad],
                    lr=self.config.distill.learning_rate,
                    weight_decay=self.config.distill.weight_decay,
                )
                self.scheduler = AdaptiveLRScheduler(
                    self.optimizer,
                    warmup_steps=0,
                    total_steps=max_steps - self.state.global_step,
                    plateau_detector=self.plateau_detector,
                )
                report.phase_transitions.append(
                    {
                        "step": self.state.global_step,
                        "from_phase": old_phase,
                        "to_phase": new_phase,
                        "description": PHASE_DESCRIPTIONS[new_phase],
                    }
                )
                logger.info(f"Phase {old_phase} → {new_phase}: {PHASE_DESCRIPTIONS[new_phase]}")

            # Logging
            if self.state.global_step % log_every == 0:
                lr = self.scheduler.get_lr()
                elapsed = time.time() - t_start
                steps_this_run = self.state.global_step - initial_global_step
                steps_per_sec = steps_this_run / max(elapsed, 1e-6)

                curriculum_diff = None
                if self.curriculum_sampler is not None:
                    curriculum_diff = self.curriculum_sampler.scheduler.get_difficulty(self.state.global_step)

                logger.info(
                    f"Step {self.state.global_step}/{max_steps} | "
                    f"Loss: {loss_val:.4f} | "
                    f"LR: {lr:.2e} | "
                    f"Phase: {self.state.phase} | "
                    f"Plateaus: {self.scheduler.get_plateau_count()} | "
                    + (f"Curriculum: {curriculum_diff:.2f} | " if curriculum_diff else "")
                    + f"{steps_per_sec:.1f} steps/s"
                )

            # Checkpoint
            if self.state.global_step % checkpoint_every == 0:
                self.save_checkpoint()

            # Curriculum sampler step update
            if self.curriculum_sampler is not None:
                self.curriculum_sampler.set_step(self.state.global_step)

            if progress_callback is not None:
                elapsed = time.time() - t_start
                completed_steps = self.state.global_step
                steps_this_run = completed_steps - initial_global_step
                steps_per_second = steps_this_run / max(elapsed, 1e-6)
                remaining_steps = max(0, max_steps - completed_steps)
                progress_callback(
                    {
                        "step": completed_steps,
                        "max_steps": max_steps,
                        "loss": loss_val,
                        "best_loss": self.state.best_loss,
                        "phase": self.state.phase,
                        "epoch": self.state.epoch,
                        "learning_rate": self.scheduler.get_lr(),
                        "elapsed_seconds": elapsed,
                        "eta_seconds": remaining_steps / max(steps_per_second, 1e-6),
                        "steps_per_second": steps_per_second,
                    }
                )

            if stop_requested is not None and stop_requested():
                stopped = True
                self.save_checkpoint(tag="stopped")
                logger.info("Training stop requested after step %s", self.state.global_step)
                break

        # Final checkpoint for completed runs; interrupted runs save ``stopped.pt`` above.
        if not stopped:
            self.save_checkpoint(tag="final")

        elapsed = time.time() - t_start
        report.total_steps = self.state.global_step
        report.elapsed_seconds = elapsed
        report.final_loss = self.state.loss_history[-1] if self.state.loss_history else 0.0
        report.best_loss = self.state.best_loss
        report.final_lr = self.scheduler.get_lr()
        report.plateaus_detected = self.scheduler.get_plateau_count()
        report.status = "stopped" if stopped else "completed"

        if self.curriculum_sampler is not None:
            report.curriculum_stats = {
                "final_difficulty": self.curriculum_sampler.scheduler.get_difficulty(self.state.global_step),
                "hard_examples_tracked": (int(self.hard_miner.update_count.sum().item()) if self.hard_miner else 0),
                "teacher_dropout_active": self.teacher_dropout is not None,
            }

        logger.info(
            f"Training complete: {report.total_steps} steps, "
            f"{elapsed:.0f}s, final_loss={report.final_loss:.4f}, "
            f"best_loss={report.best_loss:.4f}"
        )
        return report

    def _forward_step(
        self,
        batch: dict[str, Any],
        teacher_mask: list[bool] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run forward pass through student + loss.

        Handles both single-teacher and multi-teacher cases.
        """
        self.student.train()

        images = batch.get("image")
        gt_actions = batch.get("ground_truth_actions")

        # Student forward
        student_out = self.student(images, gt_actions=gt_actions)
        student_actions = student_out["actions"]

        # Check if multi-teacher loss (has teacher_actions_list)
        if hasattr(self.loss_fn, "router"):
            # UniversalDistillationLoss path
            teacher_actions = batch.get("teacher_actions_list", [])
            teacher_confidences = batch.get("teacher_confidences")
            student_features = student_out.get("vision_features", student_actions)

            # Apply teacher dropout mask
            if teacher_mask is not None and teacher_actions:
                teacher_actions = [t for t, active in zip(teacher_actions, teacher_mask) if active]
                if teacher_confidences is not None:
                    active_indices = [i for i, active in enumerate(teacher_mask) if active]
                    teacher_confidences = teacher_confidences[:, active_indices]

            loss_dict = self.loss_fn(
                student_actions=student_actions,
                teacher_actions_list=teacher_actions,
                ground_truth_actions=gt_actions,
                student_features=student_features,
                teacher_confidences=teacher_confidences,
            )
        else:
            # Single-teacher ForgeDistillationLoss path
            loss_dict = self.loss_fn(
                student_actions=student_actions,
                teacher_action_logits=batch.get("teacher_action_logits"),
                ground_truth_actions=gt_actions,
                student_vision_features=student_out.get("vision_features"),
                teacher_vision_features=batch.get("teacher_vision_features"),
                teacher_action_mean=batch.get("teacher_action_mean"),
                teacher_action_std=batch.get("teacher_action_std"),
                teacher_confidence=batch.get("confidence"),
            )

        # Add action head loss (diffusion/flow head internal loss)
        if "loss" in student_out:
            loss_dict["total"] = loss_dict["total"] + student_out["loss"]

        return loss_dict

    def save_checkpoint(self, tag: str | None = None) -> Path:
        """Save full training state."""
        name = tag or f"step_{self.state.global_step}"
        ckpt_path = self.checkpoint_dir / f"{name}.pt"
        state = {
            "global_step": self.state.global_step,
            "best_loss": self.state.best_loss,
            "phase": self.state.phase,
            "epoch": self.state.epoch,
            "student_state_dict": self.student.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "provenance": build_provenance(
                student=self.student,
                config=self.config,
                dataset=self.dataset,
            ),
            "student_config": asdict(self.config.student),
        }
        if self.hard_miner is not None:
            state["hard_miner_loss_table"] = self.hard_miner.loss_table
            state["hard_miner_update_count"] = self.hard_miner.update_count
        torch.save(state, ckpt_path)
        logger.info(f"Checkpoint saved: {ckpt_path}")

        # Also save best
        loss_val = self.state.loss_history[-1] if self.state.loss_history else float("inf")
        if loss_val <= self.state.best_loss:
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(state, best_path)

        return ckpt_path

    def load_checkpoint(self, path: str | Path) -> None:
        """Resume training from checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.state.global_step = ckpt["global_step"]
        self.state.best_loss = ckpt["best_loss"]
        self.state.phase = ckpt["phase"]
        self.state.epoch = ckpt.get("epoch", 0)
        self.student.load_state_dict(ckpt["student_state_dict"])
        set_trainable_for_phase(self.student, self.state.phase)
        self.optimizer = AdamW(
            [p for p in self.student.parameters() if p.requires_grad],
            lr=self.config.distill.learning_rate,
            weight_decay=self.config.distill.weight_decay,
        )
        self.scheduler = AdaptiveLRScheduler(
            self.optimizer,
            warmup_steps=self.config.distill.warmup_steps,
            total_steps=self.config.distill.max_steps,
            plateau_detector=self.plateau_detector,
        )
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if self.hard_miner is not None and "hard_miner_loss_table" in ckpt:
            self.hard_miner.loss_table = ckpt["hard_miner_loss_table"]
            self.hard_miner.update_count = ckpt["hard_miner_update_count"]
        self._resume_pending = True
        logger.info(f"Resumed from step {self.state.global_step}")

    def get_status(self) -> dict[str, Any]:
        """Get current training status for CLI/API."""
        status = self.state.to_dict()
        status["lr"] = self.scheduler.get_lr()
        status["plateaus"] = self.scheduler.get_plateau_count()
        if self.curriculum_sampler is not None:
            status["curriculum_difficulty"] = self.curriculum_sampler.scheduler.get_difficulty(self.state.global_step)
        if self.teacher_dropout is not None:
            status["teacher_dropout_rate"] = self.teacher_dropout.get_dropout_rate(self.state.global_step)
        if self.hard_miner is not None:
            status["hard_examples_seen"] = int(self.hard_miner.update_count.sum().item())
        return status
