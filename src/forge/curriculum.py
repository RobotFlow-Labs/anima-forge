"""PRD-22: Curriculum Learning & Adaptive Training.

Provides smart training strategies for FORGE distillation:
1. DifficultyScorer — scores training samples by difficulty
2. CurriculumScheduler — ramps difficulty over training
3. PlateauDetector — detects loss plateaus and triggers LR reduction
4. TeacherDropout — progressively drops teachers for robustness
5. HardExampleMiner — tracks per-sample loss for hard example mining
6. CurriculumSampler — PyTorch sampler combining all strategies

Design: each component is independent and composable. The
CurriculumSampler ties them together for use with DataLoader.
"""

from __future__ import annotations

import logging
import math
import random
from collections import deque

import torch
from torch.utils.data import Sampler

from forge.config import CurriculumConfig

logger = logging.getLogger(__name__)


# ── Difficulty Scorer ─────────────────────────────────────────────


class DifficultyScorer:
    """Scores training samples by difficulty.

    Supports three metrics:
    - "loss": higher loss = harder (requires pre-computed per-sample losses)
    - "confidence": lower teacher confidence = harder
    - "teacher_disagreement": higher variance across teachers = harder
    """

    def __init__(self, metric: str = "loss"):
        if metric not in ("loss", "confidence", "teacher_disagreement"):
            raise ValueError(f"Unknown difficulty metric: {metric}")
        self.metric = metric

    def score_batch(
        self,
        *,
        losses: torch.Tensor | None = None,
        confidences: torch.Tensor | None = None,
        teacher_actions: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Score a batch of samples by difficulty.

        Args:
            losses: (B,) per-sample losses (for metric="loss")
            confidences: (B, D) teacher confidence (for metric="confidence")
            teacher_actions: list of N (B, D) tensors (for metric="teacher_disagreement")

        Returns:
            (B,) difficulty scores, higher = harder
        """
        if self.metric == "loss":
            if losses is None:
                raise ValueError("losses required for metric='loss'")
            return losses

        elif self.metric == "confidence":
            if confidences is None:
                raise ValueError("confidences required for metric='confidence'")
            # Lower confidence → higher difficulty
            return 1.0 - confidences.mean(dim=-1)

        else:  # teacher_disagreement
            if teacher_actions is None or len(teacher_actions) < 2:
                raise ValueError("teacher_actions (N>=2) required for metric='teacher_disagreement'")
            stacked = torch.stack(teacher_actions)  # (N, B, D)
            variance = stacked.var(dim=0).mean(dim=-1)  # (B,)
            return variance

    def rank_indices(
        self,
        scores: torch.Tensor,
        ascending: bool = True,
    ) -> torch.Tensor:
        """Rank samples by difficulty score.

        Args:
            scores: (N,) difficulty scores
            ascending: if True, easiest first (for curriculum start)

        Returns:
            (N,) indices sorted by difficulty
        """
        return torch.argsort(scores, descending=not ascending)


# ── Curriculum Scheduler ──────────────────────────────────────────


class CurriculumScheduler:
    """Ramps the difficulty fraction over training steps.

    Controls what percentage of the dataset is available, starting
    from the easiest samples and progressively including harder ones.
    """

    VALID_SCHEDULES = ("linear", "cosine", "step")

    def __init__(
        self,
        initial_difficulty: float = 0.3,
        final_difficulty: float = 1.0,
        ramp_steps: int = 50000,
        schedule: str = "linear",
    ):
        if schedule not in self.VALID_SCHEDULES:
            raise ValueError(f"Unknown schedule '{schedule}'. Valid options: {self.VALID_SCHEDULES}")
        if ramp_steps <= 0:
            raise ValueError(f"ramp_steps must be > 0, got {ramp_steps}")
        self.initial = initial_difficulty
        self.final = final_difficulty
        self.ramp_steps = ramp_steps
        self.schedule = schedule

    def get_difficulty(self, step: int) -> float:
        """Get current difficulty fraction at given step.

        Returns:
            float in [initial, final] — fraction of data to use
        """
        if step >= self.ramp_steps:
            return self.final

        progress = step / self.ramp_steps

        if self.schedule == "linear":
            return self.initial + (self.final - self.initial) * progress

        elif self.schedule == "cosine":
            # Cosine annealing from initial to final
            cos_progress = 0.5 * (1 - math.cos(math.pi * progress))
            return self.initial + (self.final - self.initial) * cos_progress

        elif self.schedule == "step":
            # Step function: jump at 33% and 66%
            if progress < 0.33:
                return self.initial
            elif progress < 0.66:
                return self.initial + (self.final - self.initial) * 0.5
            else:
                return self.final

        raise RuntimeError(f"Unreachable: schedule '{self.schedule}' validated in __init__")


# ── Plateau Detector ──────────────────────────────────────────────


class PlateauDetector:
    """Detects loss plateaus and triggers learning rate reduction.

    Tracks a rolling window of losses. If the improvement over the
    window is below threshold, declares a plateau.
    """

    def __init__(
        self,
        window: int = 2000,
        threshold: float = 0.01,
        lr_factor: float = 0.5,
        max_plateaus: int = 3,
    ):
        self.window = window
        self.threshold = threshold
        self.lr_factor = lr_factor
        self.max_plateaus = max_plateaus
        self.loss_history: deque[float] = deque(maxlen=window)
        self.plateau_count = 0
        self._last_check_step = 0

    def update(self, loss: float) -> None:
        """Record a new loss value."""
        self.loss_history.append(loss)

    def check_plateau(self, step: int) -> bool:
        """Check if we're on a plateau.

        Only checks once per window to avoid repeated triggers.

        Returns:
            True if plateau detected (and LR should be reduced)
        """
        if len(self.loss_history) < self.window:
            return False

        if step - self._last_check_step < self.window:
            return False

        self._last_check_step = step

        if self.plateau_count >= self.max_plateaus:
            return False

        losses = list(self.loss_history)
        mid = len(losses) // 2
        first_half = sum(losses[:mid]) / max(mid, 1)
        second_half = sum(losses[mid:]) / max(len(losses) - mid, 1)

        improvement = (first_half - second_half) / (abs(first_half) + 1e-8)

        if improvement < self.threshold:
            self.plateau_count += 1
            logger.info(
                f"Plateau #{self.plateau_count} detected at step {step}. "
                f"Improvement: {improvement:.4f} < {self.threshold}"
            )
            return True

        return False

    def get_lr_multiplier(self) -> float:
        """Get the cumulative LR multiplier from plateau reductions."""
        return self.lr_factor**self.plateau_count


# ── Teacher Dropout ───────────────────────────────────────────────


class TeacherDropout:
    """Progressively drops random teachers during training.

    Forces the student to not rely on any single teacher, improving
    robustness. Dropout rate ramps from start to end over training.
    """

    def __init__(
        self,
        n_teachers: int,
        dropout_start: float = 0.0,
        dropout_end: float = 0.3,
        ramp_steps: int = 30000,
    ):
        self.n_teachers = n_teachers
        self.dropout_start = dropout_start
        self.dropout_end = dropout_end
        self.ramp_steps = ramp_steps

    def get_dropout_rate(self, step: int) -> float:
        """Get current dropout rate."""
        if step >= self.ramp_steps:
            return self.dropout_end
        progress = step / self.ramp_steps
        return self.dropout_start + (self.dropout_end - self.dropout_start) * progress

    def get_active_mask(self, step: int) -> list[bool]:
        """Get a mask of which teachers are active this step.

        Returns:
            list[bool] of length n_teachers, True = active
        """
        rate = self.get_dropout_rate(step)
        if rate <= 0:
            return [True] * self.n_teachers

        # Always keep at least 1 teacher
        mask = [True] * self.n_teachers
        n_drop = min(int(self.n_teachers * rate), self.n_teachers - 1)

        if n_drop > 0:
            drop_indices = random.sample(range(self.n_teachers), n_drop)
            for i in drop_indices:
                mask[i] = False

        return mask


# ── Hard Example Miner ────────────────────────────────────────────


class HardExampleMiner:
    """Tracks per-sample losses for hard example mining.

    Maintains a loss history keyed by sample index. When sampling,
    mixes a fraction of hard (high-loss) examples with random ones.
    """

    def __init__(
        self,
        dataset_size: int,
        hard_ratio: float = 0.3,
        history_size: int = 10000,
    ):
        self.dataset_size = dataset_size
        self.hard_ratio = hard_ratio
        # Per-sample running average loss
        self.loss_table = torch.zeros(dataset_size)
        self.update_count = torch.zeros(dataset_size, dtype=torch.long)
        self._history_size = history_size

    def update_losses(self, indices: torch.Tensor | list[int], losses: torch.Tensor) -> None:
        """Update per-sample loss records.

        Args:
            indices: sample indices in the dataset
            losses: per-sample loss values
        """
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()
        if isinstance(losses, torch.Tensor):
            losses = losses.detach().cpu()

        for idx, loss_val in zip(indices, losses):
            if 0 <= idx < self.dataset_size:
                count = self.update_count[idx].item()
                # Exponential moving average
                alpha = 1.0 / (count + 1)
                self.loss_table[idx] = (1 - alpha) * self.loss_table[idx] + alpha * loss_val
                self.update_count[idx] += 1

    def sample_indices(self, batch_size: int) -> list[int]:
        """Sample a batch mixing hard and random examples.

        Args:
            batch_size: total batch size

        Returns:
            list of sample indices
        """
        n_hard = int(batch_size * self.hard_ratio)
        n_random = batch_size - n_hard

        # Hard examples: top-k by loss
        if self.update_count.sum() > 0 and n_hard > 0:
            # Only consider samples we've seen
            seen_mask = self.update_count > 0
            if seen_mask.sum() >= n_hard:
                seen_losses = self.loss_table.clone()
                seen_losses[~seen_mask] = -1  # exclude unseen
                hard_count = min(n_hard, int(seen_mask.sum().item()))
                _, hard_indices = torch.topk(seen_losses, hard_count)
                hard_list = hard_indices.tolist()
            else:
                hard_list = []
                n_random = batch_size
        else:
            hard_list = []
            n_random = batch_size

        # Random examples from full dataset, excluding hard ones
        hard_set = set(hard_list)
        available = [i for i in range(self.dataset_size) if i not in hard_set]
        random_list = random.sample(available, min(n_random, len(available)))

        return hard_list + random_list

    def get_difficulty_scores(self) -> torch.Tensor:
        """Get current difficulty scores for all samples.

        Returns:
            (dataset_size,) tensor of difficulty scores
        """
        scores = self.loss_table.clone()
        # Unseen samples get median difficulty
        seen = self.update_count > 0
        if seen.any():
            median_loss = scores[seen].median()
            scores[~seen] = median_loss
        return scores


# ── Curriculum Sampler ────────────────────────────────────────────


class CurriculumSampler(Sampler):
    """PyTorch sampler implementing curriculum learning.

    Combines difficulty scoring, curriculum scheduling, and hard
    example mining into a single sampler for DataLoader.
    """

    def __init__(
        self,
        dataset_size: int,
        config: CurriculumConfig,
        difficulty_scores: torch.Tensor | None = None,
    ):
        self.dataset_size = dataset_size
        self.config = config
        self.scheduler = CurriculumScheduler(
            initial_difficulty=config.initial_difficulty,
            final_difficulty=config.final_difficulty,
            ramp_steps=config.ramp_steps,
            schedule=config.ramp_schedule,
        )
        self.miner = (
            HardExampleMiner(
                dataset_size=dataset_size,
                hard_ratio=config.hard_example_ratio if config.hard_example_mining else 0.0,
                history_size=config.loss_history_size,
            )
            if config.hard_example_mining
            else None
        )

        self._step = 0
        self._difficulty_scores = difficulty_scores
        self._sorted_indices: list[int] | None = None

        if difficulty_scores is not None:
            self._sorted_indices = torch.argsort(difficulty_scores).tolist()

    def set_step(self, step: int) -> None:
        """Update the current training step (affects curriculum difficulty)."""
        self._step = step

    def update_difficulty_scores(self, scores: torch.Tensor) -> None:
        """Update difficulty scores (re-sorts the curriculum order)."""
        self._difficulty_scores = scores
        self._sorted_indices = torch.argsort(scores).tolist()

    def __iter__(self):
        difficulty = self.scheduler.get_difficulty(self._step)
        n_available = max(1, int(self.dataset_size * difficulty))

        if self._sorted_indices is not None:
            # Use curriculum order: easiest first
            pool = self._sorted_indices[:n_available]
        else:
            # No scores yet: random subset
            pool = list(range(n_available))

        # Shuffle the available pool
        indices = pool.copy()
        random.shuffle(indices)

        # If hard example mining is active, replace some with hard examples
        if self.miner is not None and self.miner.update_count.sum() > 0:
            n_hard = int(len(indices) * self.config.hard_example_ratio)
            if n_hard > 0:
                hard_indices = self.miner.sample_indices(n_hard)
                # Replace end of shuffled indices with hard examples
                indices = indices[: len(indices) - len(hard_indices)] + hard_indices

        return iter(indices)

    def __len__(self) -> int:
        difficulty = self.scheduler.get_difficulty(self._step)
        return max(1, int(self.dataset_size * difficulty))
