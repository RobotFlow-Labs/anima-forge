"""PRD-21: Universal Teacher Ensemble Distillation.

One student learns from ALL teachers simultaneously via confidence-based
routing, diversity regularization, and consistency loss. Extends PRD-12's
TeacherRouter and MultiTeacherDistillationLoss.

Key components:
1. ConfidenceRouter — extends TeacherRouter with confidence-augmented input + Gumbel softmax
2. DiversityLoss — encourages uniform teacher utilisation
3. ConsistencyLoss — inverse-variance weighted student-teacher agreement
4. UniversalDistillationLoss — combines all losses into a single forward pass
5. UniversalRunner — multi-GPU parallel teacher inference + training loop
6. plan_gpu_placement() — greedy bin-packing for teacher→GPU assignment
"""

from __future__ import annotations

import logging
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as functional

from forge.multi_teacher import MultiTeacherDistillationLoss, TeacherRouter
from forge.provenance import build_provenance, provenance_contains_mock

logger = logging.getLogger(__name__)


def _runner_allows_mock(student: nn.Module, config: Any) -> bool:
    """Return explicit mock permission from runtime config or environment."""
    unwrapped = getattr(student, "module", student)
    candidates = (config, getattr(config, "student", None), getattr(unwrapped, "config", None))
    for candidate in candidates:
        if candidate is not None and hasattr(candidate, "allow_mock"):
            return bool(candidate.allow_mock)
    return os.environ.get("FORGE_ALLOW_MOCK", "").strip().casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }


# ── Teacher VRAM sizes (GB) ──────────────────────────────────────

TEACHER_SIZES: dict[str, float] = {
    "openvla-7b": 15.2,
    "rdt2-fm": 2.5,
    "smolvla-base": 1.0,
    "molmoact2-libero": 12.1,
    "vla-jepa-3b": 7.0,
    "rt2-x": 12.0,
    "octo-base": 1.5,
    "octo-small": 0.6,
    "pi0-base": 3.2,
    "pi0-small": 1.1,
    "gr1-base": 4.0,
    "cogact-base": 2.8,
    "spatialvla-4b": 8.5,
    "bitvla-base": 0.8,
}


# ── TeacherSlot ───────────────────────────────────────────────────


@dataclass
class TeacherSlot:
    """Metadata for a single teacher in the ensemble."""

    name: str
    adapter: Any = None  # TeacherAdapter instance (lazy)
    device: str = "cpu"
    vram_mb: float = 0.0
    confidence_dim: int = 7  # default = action_dim


# ── GPU placement ─────────────────────────────────────────────────


def plan_gpu_placement(
    teacher_names: list[str],
    gpu_memory_mb: list[float] | None = None,
) -> dict[str, str]:
    """Greedy bin-packing: assign teachers to GPUs, large-first.

    Args:
        teacher_names: list of teacher model names
        gpu_memory_mb: available VRAM per GPU in MB.  If None or empty,
                       all teachers go to CPU.

    Returns:
        dict mapping teacher_name → device string ("cuda:0", "cpu", etc.)
    """
    if not gpu_memory_mb:
        return {name: "cpu" for name in teacher_names}

    # Sort teachers by size descending (greedy large-first)
    sized = []
    for name in teacher_names:
        gb = TEACHER_SIZES.get(name, 2.0)  # default 2 GB if unknown
        sized.append((name, gb * 1024))  # convert to MB

    sized.sort(key=lambda x: x[1], reverse=True)

    remaining = list(gpu_memory_mb)  # mutable copy
    assignment: dict[str, str] = {}

    for name, need_mb in sized:
        placed = False
        for i, avail in enumerate(remaining):
            if avail >= need_mb:
                assignment[name] = f"cuda:{i}"
                remaining[i] -= need_mb
                placed = True
                break
        if not placed:
            # Doesn't fit on any GPU → CPU fallback
            assignment[name] = "cpu"

    return assignment


# ── ConfidenceRouter ──────────────────────────────────────────────


class ConfidenceRouter(TeacherRouter):
    """Router that concatenates teacher confidence scores to student features.

    Input:  student features (B, D) + teacher confidences (B, N, C)
    Output: teacher weights (B, N) via temperature-scaled softmax or Gumbel

    When use_gumbel=True, training uses Gumbel-softmax for differentiable
    discrete routing; eval always uses standard softmax.
    """

    def __init__(
        self,
        d_input: int,
        n_teachers: int,
        confidence_dim: int = 7,
        temperature: float = 1.0,
        use_gumbel: bool = True,
    ):
        # Router MLP takes concatenated [features; flattened_confidences]
        total_dim = d_input + n_teachers * confidence_dim
        super().__init__(total_dim, n_teachers, temperature)
        self.confidence_dim = confidence_dim
        self.use_gumbel = use_gumbel
        self.d_input = d_input  # original feature dim (before concat)

    def forward(
        self,
        features: torch.Tensor,
        teacher_confidences: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Route with optional confidence augmentation.

        Args:
            features: (B, D) student hidden features
            teacher_confidences: (B, N, C) per-teacher confidence vectors.
                                 If None, zero-pad (fallback to base router).

        Returns:
            (B, N) routing weights summing to 1
        """
        batch_size = features.shape[0]

        if teacher_confidences is not None:
            # NaN guard — replace NaN with 0
            teacher_confidences = torch.where(
                torch.isnan(teacher_confidences),
                torch.zeros_like(teacher_confidences),
                teacher_confidences,
            )
            flat_conf = teacher_confidences.reshape(batch_size, -1)  # (B, N*C)
            x = torch.cat([features, flat_conf], dim=-1)
        else:
            # Zero-pad confidence slots so dimensions match
            pad = torch.zeros(
                batch_size,
                self.n_teachers * self.confidence_dim,
                device=features.device,
                dtype=features.dtype,
            )
            x = torch.cat([features, pad], dim=-1)

        logits = self.router(x) / self.temperature

        if self.use_gumbel and self.training:
            return functional.gumbel_softmax(logits, tau=self.temperature, hard=False)
        return functional.softmax(logits, dim=-1)


# ── Diversity Loss ────────────────────────────────────────────────


class DiversityLoss(nn.Module):
    """Encourages uniform teacher utilisation across the batch.

    L_diversity = max_entropy - entropy(mean_weights)

    When all teachers are used equally, mean_weights is uniform and
    entropy is maximal → loss ≈ 0.  When collapsed to one teacher,
    entropy is 0 → loss = max_entropy = log(N).
    """

    def forward(self, weights: torch.Tensor) -> torch.Tensor:
        """Compute diversity loss.

        Args:
            weights: (B, N) routing weights per sample

        Returns:
            scalar diversity loss
        """
        n_teachers = weights.shape[-1]
        avg_weights = weights.mean(dim=0)  # (N,)

        # Clamp to avoid log(0)
        avg_weights = avg_weights.clamp(min=1e-8)

        entropy = -(avg_weights * avg_weights.log()).sum()
        max_entropy = math.log(n_teachers)
        return max_entropy - entropy


# ── Consistency Loss ──────────────────────────────────────────────


class ConsistencyLoss(nn.Module):
    """Inverse-variance weighted agreement between student and teacher mean.

    Where teachers agree (low variance), the student is penalised more
    for deviating.  Where they disagree (high variance), the penalty
    is lighter — the signal is unreliable.

    L_consistency = mean( (1/var) * (student - teacher_mean)^2 )
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        student_actions: torch.Tensor,
        teacher_actions_list: list[torch.Tensor],
    ) -> torch.Tensor:
        """Compute consistency loss.

        Args:
            student_actions: (B, H, D) or (B, D) student predictions
            teacher_actions_list: list of N tensors, same shape as student

        Returns:
            scalar consistency loss
        """
        stacked = torch.stack(teacher_actions_list, dim=0)  # (N, B, ...)
        teacher_mean = stacked.mean(dim=0)  # (B, ...)
        teacher_var = stacked.var(dim=0, unbiased=False)  # (B, ...)

        inv_var = 1.0 / (teacher_var + self.eps)
        sq_diff = (student_actions - teacher_mean) ** 2

        return (inv_var * sq_diff).mean()


# ── UniversalDistillationLoss ─────────────────────────────────────


class UniversalDistillationLoss(MultiTeacherDistillationLoss):
    """Extends MultiTeacherDistillationLoss with confidence routing,
    diversity regularisation, and consistency loss.

    Total = α_task * L_task
          + (1 - α_task - α_div - α_con) * L_kd
          + α_div * L_diversity
          + α_con * L_consistency
    """

    def __init__(
        self,
        n_teachers: int,
        d_student: int,
        confidence_dim: int = 7,
        temperature: float = 4.0,
        alpha_task: float = 0.3,
        alpha_diversity: float = 0.05,
        alpha_consistency: float = 0.1,
        use_gumbel: bool = True,
    ):
        super().__init__(
            n_teachers=n_teachers,
            d_student=d_student,
            temperature=temperature,
            alpha_task=alpha_task,
        )
        # Replace the base router with ConfidenceRouter
        self.router = ConfidenceRouter(
            d_input=d_student,
            n_teachers=n_teachers,
            confidence_dim=confidence_dim,
            temperature=temperature,
            use_gumbel=use_gumbel,
        )
        self.diversity_loss = DiversityLoss()
        self.consistency_loss = ConsistencyLoss()
        self.alpha_diversity = alpha_diversity
        self.alpha_consistency = alpha_consistency

        # Validate alpha sum
        alpha_sum = alpha_task + alpha_diversity + alpha_consistency
        if alpha_sum > 1.0:
            raise ValueError(
                f"Alpha weights sum to {alpha_sum:.2f} > 1.0 "
                f"(task={alpha_task}, diversity={alpha_diversity}, "
                f"consistency={alpha_consistency}). "
                f"This leaves no weight for the KD loss."
            )

    def forward(
        self,
        student_actions: torch.Tensor,
        teacher_actions_list: list[torch.Tensor],
        ground_truth_actions: torch.Tensor,
        student_features: torch.Tensor,
        teacher_confidences: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute universal ensemble distillation loss.

        Args:
            student_actions: (B, H, D) or (B, D) student predictions
            teacher_actions_list: list of N teacher prediction tensors
            ground_truth_actions: same shape as student_actions
            student_features: (B, D_student) for routing
            teacher_confidences: (B, N, C) per-teacher confidence.
                                 None → fallback to zero-padded routing.

        Returns:
            dict with 'total', 'kd', 'task', 'diversity', 'consistency',
            'router_weights'
        """
        # Routing weights via confidence-augmented router
        weights = self.router(student_features, teacher_confidences)  # (B, N)

        # Per-teacher KD losses
        teacher_losses = []
        for teacher_actions in teacher_actions_list:
            loss_i = functional.mse_loss(
                student_actions,
                teacher_actions,
                reduction="none",
            ).mean(dim=-1)  # reduce action dim → (B,) or (B, H)
            # If chunked (B, H), reduce horizon too
            if loss_i.dim() > 1:
                loss_i = loss_i.mean(dim=-1)  # (B,)
            teacher_losses.append(loss_i)

        teacher_losses_t = torch.stack(teacher_losses, dim=-1)  # (B, N)
        weighted_kd = (weights * teacher_losses_t).sum(dim=-1).mean()

        # Task loss
        task_loss = functional.mse_loss(student_actions, ground_truth_actions)

        # Diversity
        div_loss = self.diversity_loss(weights)

        # Consistency
        con_loss = self.consistency_loss(student_actions, teacher_actions_list)

        # Weighted total
        alpha_kd = 1.0 - self.alpha_task - self.alpha_diversity - self.alpha_consistency
        alpha_kd = max(alpha_kd, 0.0)  # safety clamp

        total = (
            alpha_kd * weighted_kd
            + self.alpha_task * task_loss
            + self.alpha_diversity * div_loss
            + self.alpha_consistency * con_loss
        )

        return {
            "total": total,
            "kd": weighted_kd,
            "task": task_loss,
            "diversity": div_loss,
            "consistency": con_loss,
            "router_weights": weights.detach(),
        }


# ── UniversalRunner ───────────────────────────────────────────────


class UniversalRunner:
    """Orchestrates universal ensemble distillation training.

    Features:
    - ThreadPoolExecutor parallel teacher inference (grouped by device)
    - Checkpoint save/resume
    - Staged mode: rotate teacher subsets
    - Gradient safety asserts
    """

    def __init__(
        self,
        student: nn.Module,
        teacher_slots: list[TeacherSlot],
        loss_fn: UniversalDistillationLoss,
        optimizer: torch.optim.Optimizer,
        *,
        max_steps: int = 100000,
        checkpoint_dir: str = "./checkpoints/universal",
        checkpoint_every: int = 5000,
        staged: bool = False,
        teachers_per_stage: int = 3,
        steps_per_stage: int = 25000,
        device: str = "cpu",
        config: Any = None,
        dataset: Any = None,
        model_dir: str | Path | None = None,
    ):
        self.student = student
        self.teacher_slots = teacher_slots
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.max_steps = max_steps
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_every = checkpoint_every
        self.staged = staged
        self.teachers_per_stage = teachers_per_stage
        self.steps_per_stage = steps_per_stage
        self.device = device
        unwrapped_student = getattr(student, "module", student)
        self.config = config if config is not None else getattr(unwrapped_student, "config", None)
        self.dataset = dataset
        self.model_dir = model_dir
        self.global_step = 0
        self._lock = threading.Lock()

    @property
    def active_teachers(self) -> list[TeacherSlot]:
        """Return currently active teachers (all or staged subset)."""
        if not self.staged:
            return self.teacher_slots

        stage_idx = self.global_step // self.steps_per_stage
        n = self.teachers_per_stage
        start = (stage_idx * n) % len(self.teacher_slots)
        indices = [(start + i) % len(self.teacher_slots) for i in range(n)]
        return [self.teacher_slots[i] for i in indices]

    def _teacher_inference_parallel(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[list[torch.Tensor], torch.Tensor | None]:
        """Run teacher inference in parallel, grouped by device.

        Returns:
            (teacher_actions_list, teacher_confidences)
        """
        teachers = self.active_teachers
        results: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

        # Group teachers by device for efficient batching
        sorted_teachers = sorted(teachers, key=lambda t: t.device)
        grouped = {dev: list(grp) for dev, grp in groupby(sorted_teachers, key=lambda t: t.device)}

        def _run_group(device: str, slots: list[TeacherSlot]) -> None:
            for slot in slots:
                if slot.adapter is None:
                    continue
                with torch.no_grad():
                    # Adapters return ActionChunk; we extract actions + confidence
                    chunk = slot.adapter.predict(
                        batch.get("image"),
                        batch.get("instruction", ""),
                    )
                    actions = torch.as_tensor(
                        chunk.actions,
                        dtype=torch.float32,
                    ).to(self.device)
                    confidence = torch.as_tensor(
                        chunk.confidence,
                        dtype=torch.float32,
                    ).to(self.device)
                    with self._lock:
                        results[slot.name] = (actions, confidence)

        with ThreadPoolExecutor(max_workers=len(grouped)) as pool:
            futures = []
            for dev, slots in grouped.items():
                futures.append(pool.submit(_run_group, dev, slots))
            for f in futures:
                f.result()  # propagate exceptions

        # Assemble in teacher order
        teacher_actions = []
        confidence_list = []
        for slot in teachers:
            if slot.name in results:
                acts, conf = results[slot.name]
                teacher_actions.append(acts)
                confidence_list.append(conf)

        if confidence_list:
            teacher_confidences = torch.stack(confidence_list, dim=1)  # (B, N, C)
        else:
            teacher_confidences = None

        return teacher_actions, teacher_confidences

    def training_step(
        self,
        batch: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        """Execute one training step.

        Args:
            batch: dict with 'student_actions', 'student_features',
                   'ground_truth_actions', and optionally 'teacher_actions_list'
                   + 'teacher_confidences' (pre-computed) or 'image'/'instruction'
                   for live teacher inference.

        Returns:
            Loss dict from UniversalDistillationLoss.forward()
        """
        self.student.train()
        self.loss_fn.train()

        student_actions: torch.Tensor = batch["student_actions"]
        student_features = batch["student_features"]
        gt_actions = batch["ground_truth_actions"]

        # Use pre-computed teacher outputs or run live inference
        if "teacher_actions_list" in batch:
            teacher_actions: list[torch.Tensor] = batch["teacher_actions_list"]
            teacher_confidences = batch.get("teacher_confidences")
        else:
            teacher_actions, teacher_confidences = self._teacher_inference_parallel(batch)

        loss_dict = self.loss_fn(
            student_actions=student_actions,
            teacher_actions_list=teacher_actions,
            ground_truth_actions=gt_actions,
            student_features=student_features,
            teacher_confidences=teacher_confidences,
        )

        # Gradient safety: total must require grad
        assert loss_dict["total"].requires_grad, (
            "Total loss does not require grad — check that student is in train mode"
        )

        self.optimizer.zero_grad()
        loss_dict["total"].backward()
        self.optimizer.step()

        self.global_step += 1

        # Checkpoint
        if self.global_step % self.checkpoint_every == 0:
            self.save_checkpoint()

        return loss_dict

    def save_checkpoint(
        self,
        path: str | Path | None = None,
        *,
        dataset: Any = None,
    ) -> Path:
        """Save training checkpoint."""
        provenance = build_provenance(
            student=self.student,
            config=self.config,
            dataset=dataset if dataset is not None else self.dataset,
            model_dir=self.model_dir,
        )
        if provenance_contains_mock(provenance) and not _runner_allows_mock(self.student, self.config):
            raise ValueError(
                "Universal distillation refuses to write a mock-derived checkpoint. "
                "Use real runtime inputs or enable config.student.allow_mock explicitly."
            )
        save_dir = Path(path) if path else self.checkpoint_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = save_dir / f"step_{self.global_step}.pt"
        torch.save(
            {
                "global_step": self.global_step,
                "student_state_dict": self.student.state_dict(),
                "loss_fn_state_dict": self.loss_fn.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "provenance": provenance,
            },
            ckpt_path,
        )
        logger.info(f"Checkpoint saved: {ckpt_path}")
        return ckpt_path

    def load_checkpoint(self, path: str | Path) -> None:
        """Resume from checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.global_step = ckpt["global_step"]
        self.student.load_state_dict(ckpt["student_state_dict"])
        self.loss_fn.load_state_dict(ckpt["loss_fn_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        logger.info(f"Resumed from step {self.global_step}")

    @torch.no_grad()
    def evaluate(
        self,
        eval_batches: list[dict[str, torch.Tensor]],
    ) -> dict[str, float]:
        """Run evaluation over a list of batches.

        Returns:
            dict with mean loss components
        """
        self.student.eval()
        self.loss_fn.eval()

        totals: dict[str, float] = {}
        count = 0

        for batch in eval_batches:
            student_actions = batch["student_actions"]
            student_features = batch["student_features"]
            gt_actions = batch["ground_truth_actions"]
            teacher_actions = batch["teacher_actions_list"]
            teacher_confidences = batch.get("teacher_confidences")

            loss_dict = self.loss_fn(
                student_actions=student_actions,
                teacher_actions_list=teacher_actions,
                ground_truth_actions=gt_actions,
                student_features=student_features,
                teacher_confidences=teacher_confidences,
            )

            for k, v in loss_dict.items():
                if isinstance(v, torch.Tensor) and v.dim() == 0:
                    totals[k] = totals.get(k, 0.0) + v.item()
            count += 1

        return {k: v / max(count, 1) for k, v in totals.items()}
