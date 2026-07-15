"""PRD-28: Domain Adaptation & Fine-Tuning Pipeline.

Fine-tune trained FORGE student models on custom robot data, with
configurable LoRA unfreezing, mixed data sources, and replay-based
continual learning to prevent catastrophic forgetting.

Usage:
    from forge.finetune import FinetuneConfig, FinetuneTrainer

    ft_config = FinetuneConfig(
        checkpoint_path="outputs/checkpoints/best.pt",
        strategy="lora",
        lr=5e-5,
        max_steps=5000,
    )
    trainer = FinetuneTrainer(student, ft_config, device="cuda")
    report = trainer.train(domain_dataset)
"""

from __future__ import annotations

import logging
import math
import os
import time
from collections.abc import Sized
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from forge.provenance import build_provenance

logger = logging.getLogger(__name__)


def _student_allows_mock(model: nn.Module) -> bool:
    """Return the explicit mock policy carried by a FORGE student."""
    student = getattr(model, "module", model)
    config = getattr(student, "config", None)
    if config is not None and hasattr(config, "allow_mock"):
        return bool(config.allow_mock)
    return os.environ.get("FORGE_ALLOW_MOCK", "").strip().casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }


# ── Config ────────────────────────────────────────────────────


@dataclass
class FinetuneConfig:
    """Configuration for domain adaptation fine-tuning."""

    checkpoint_path: str = ""  # Path to pretrained student checkpoint
    strategy: str = "lora"  # "lora" | "action_head" | "full"
    lr: float = 5e-5  # Lower than distillation LR
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_steps: int = 5000
    batch_size: int = 8
    eval_every: int = 500
    save_every: int = 1000
    output_dir: str = "./outputs/finetune"

    # Replay buffer for continual learning
    replay_enabled: bool = False
    replay_ratio: float = 0.2  # Fraction of batch from replay
    replay_buffer_size: int = 5000

    # Mixed data sources
    mix_teacher_data: bool = False
    teacher_data_ratio: float = 0.3  # Fraction from teacher labels

    # EWC (Elastic Weight Consolidation) for catastrophic forgetting
    ewc_enabled: bool = False
    ewc_lambda: float = 1000.0  # EWC penalty strength


# ── Replay Buffer ─────────────────────────────────────────────


class ReplayBuffer:
    """Experience replay buffer for continual learning.

    Stores a fixed-size buffer of past training samples to mix
    with new domain data, preventing catastrophic forgetting.
    """

    def __init__(self, max_size: int = 5000):
        self.max_size = max_size
        self._buffer: list[dict[str, torch.Tensor]] = []
        self._idx = 0

    def add(self, sample: dict[str, torch.Tensor]) -> None:
        """Add a single sample (detached CPU tensors)."""
        detached = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in sample.items()}
        if len(self._buffer) < self.max_size:
            self._buffer.append(detached)
        else:
            self._buffer[self._idx % self.max_size] = detached
        self._idx += 1

    def add_batch(self, batch: dict[str, torch.Tensor]) -> None:
        """Add individual samples from a batch dict (unbatches along dim 0)."""
        # Determine batch size from first tensor
        batch_size = None
        for v in batch.values():
            if isinstance(v, torch.Tensor) and v.dim() > 0:
                batch_size = v.shape[0]
                break
        if batch_size is None:
            self.add(batch)
            return
        for i in range(batch_size):
            sample = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor) and v.dim() > 0 and v.shape[0] == batch_size:
                    sample[k] = v[i]
                else:
                    sample[k] = v
            self.add(sample)

    def sample(self, n: int) -> list[dict[str, torch.Tensor]]:
        """Sample n random items from the buffer."""
        if not self._buffer:
            return []
        import random

        n = min(n, len(self._buffer))
        return random.sample(self._buffer, n)

    @property
    def size(self) -> int:
        return len(self._buffer)

    def is_ready(self, min_size: int = 1) -> bool:
        return self.size >= min_size


# ── EWC Penalty ───────────────────────────────────────────────


class EWCPenalty:
    """Elastic Weight Consolidation for catastrophic forgetting prevention.

    Stores Fisher information matrix diagonal and reference parameters.
    Penalizes deviation from pretrained weights proportional to their
    importance (estimated via Fisher information).
    """

    def __init__(self, model: nn.Module, ewc_lambda: float = 1000.0):
        self.ewc_lambda = ewc_lambda
        self._means: dict[str, torch.Tensor] = {}
        self._fisher: dict[str, torch.Tensor] = {}
        self.used_synthetic_data = False

        # Store reference parameters (pretrained weights)
        for name, param in model.named_parameters():
            if param.requires_grad:
                self._means[name] = param.data.clone()

    def compute_fisher(
        self,
        model: nn.Module,
        dataset: Dataset,
        device: str = "cpu",
        n_samples: int = 200,
    ) -> None:
        """Estimate Fisher information diagonal from dataset.

        Uses a subset of the dataset to estimate parameter importance.
        """
        model.eval()
        fisher: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                fisher[name] = torch.zeros_like(param.data)

        loader = DataLoader(dataset, batch_size=1, shuffle=True)
        count = 0

        for batch in loader:
            if count >= n_samples:
                break

            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            model.zero_grad()
            image = batch.get("image")
            if image is None:
                if not _student_allows_mock(model):
                    raise ValueError(
                        "EWC requires dataset images. Synthetic images are only allowed "
                        "when config.student.allow_mock is enabled."
                    )
                self.used_synthetic_data = True
                image = torch.randn(1, 3, 384, 384, device=device)
            output = model(image)
            if isinstance(output, dict) and "actions" in output:
                loss = output["actions"].pow(2).mean()
            elif isinstance(output, torch.Tensor):
                loss = output.pow(2).mean()
            else:
                continue

            loss.backward()

            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.data.pow(2)

            count += 1

        # Average
        for name in fisher:
            fisher[name] /= max(count, 1)

        self._fisher = fisher
        model.train()
        logger.info(f"Fisher information computed from {count} samples")

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """Compute EWC penalty loss."""
        loss = torch.tensor(0.0, device=next(model.parameters()).device)
        for name, param in model.named_parameters():
            if name in self._fisher and name in self._means:
                fisher = self._fisher[name].to(param.device)
                mean = self._means[name].to(param.device)
                loss = loss + (fisher * (param - mean).pow(2)).sum()
        return self.ewc_lambda * loss

    @property
    def has_fisher(self) -> bool:
        return len(self._fisher) > 0


# ── Fine-tune Report ──────────────────────────────────────────


@dataclass
class FinetuneReport:
    """Report from a fine-tuning run."""

    total_steps: int = 0
    elapsed_seconds: float = 0.0
    final_loss: float = 0.0
    best_loss: float = float("inf")
    strategy: str = ""
    checkpoint_path: str = ""
    ewc_used: bool = False
    replay_used: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_steps": self.total_steps,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "final_loss": round(self.final_loss, 6),
            "best_loss": round(self.best_loss, 6),
            "strategy": self.strategy,
            "checkpoint_path": self.checkpoint_path,
            "ewc_used": self.ewc_used,
            "replay_used": self.replay_used,
        }


# ── Strategy: Parameter Freezing ──────────────────────────────


def apply_finetune_strategy(model: nn.Module, strategy: str) -> int:
    """Freeze/unfreeze parameters based on fine-tuning strategy.

    Args:
        model: FORGE student model.
        strategy: One of "lora", "action_head", "full".

    Returns:
        Number of trainable parameters.
    """
    # First freeze everything
    for param in model.parameters():
        param.requires_grad = False

    if strategy == "lora":
        # Unfreeze LoRA adapters + action head
        for name, param in model.named_parameters():
            if "lora" in name.lower() or "action_head" in name:
                param.requires_grad = True
    elif strategy == "action_head":
        # Only unfreeze action head
        for name, param in model.named_parameters():
            if "action_head" in name:
                param.requires_grad = True
    elif strategy == "full":
        # Unfreeze LoRA + bridge + action head (NOT vision encoder)
        for name, param in model.named_parameters():
            if "lora" in name.lower() or "bridge" in name or "action_head" in name:
                param.requires_grad = True
    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Use 'lora', 'action_head', or 'full'.")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Finetune strategy '{strategy}': {trainable:,} trainable params")
    return trainable


# ── Batch Helpers ─────────────────────────────────────────


def _collate_samples(
    samples: list[dict[str, torch.Tensor]],
    device: str,
) -> dict[str, torch.Tensor]:
    """Stack individual samples into a batch dict."""
    if not samples:
        return {}
    keys = samples[0].keys()
    batch: dict[str, torch.Tensor] = {}
    for k in keys:
        vals = [s[k] for s in samples]
        if isinstance(vals[0], torch.Tensor):
            batch[k] = torch.stack(vals).to(device)
        else:
            batch[k] = vals[0]  # Non-tensor: take first
    return batch


def _mix_batches(
    main: dict[str, torch.Tensor],
    replay: dict[str, torch.Tensor],
    n_replay: int,
) -> dict[str, torch.Tensor]:
    """Replace last n_replay samples in main batch with replay samples."""
    mixed: dict[str, torch.Tensor] = {}
    for k in main:
        v = main[k]
        if isinstance(v, torch.Tensor) and v.dim() > 0 and k in replay:
            rv = replay[k]
            n = min(n_replay, v.shape[0], rv.shape[0])
            mixed[k] = torch.cat([v[:-n], rv[:n]], dim=0)
        else:
            mixed[k] = v
    return mixed


# ── Fine-tune Trainer ─────────────────────────────────────────


class FinetuneTrainer:
    """Fine-tune a pretrained FORGE student on domain-specific data.

    Supports:
    - LoRA-only, action-head-only, or full fine-tuning strategies
    - Replay buffer for continual learning
    - EWC penalty to prevent catastrophic forgetting
    - Mixed training with teacher labels + domain data
    """

    def __init__(
        self,
        student: nn.Module,
        config: FinetuneConfig,
        *,
        device: str = "cpu",
    ):
        self.student = student.to(device)
        self.config = config
        self.device = device

        # Apply strategy
        apply_finetune_strategy(self.student, config.strategy)

        # Output dir
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Optimizer
        trainable_params = [p for p in student.parameters() if p.requires_grad]
        self.optimizer = AdamW(
            trainable_params,
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        # Replay buffer
        self.replay: ReplayBuffer | None = None
        if config.replay_enabled:
            self.replay = ReplayBuffer(max_size=config.replay_buffer_size)

        # EWC
        self.ewc: EWCPenalty | None = None
        if config.ewc_enabled:
            self.ewc = EWCPenalty(student, ewc_lambda=config.ewc_lambda)

        self._provenance_dataset: Dataset | None = None
        self._used_synthetic_data = False

    def _cosine_lr(self, step: int) -> float:
        """Cosine LR schedule with warmup."""
        if step < self.config.warmup_steps:
            return self.config.lr * (step / max(1, self.config.warmup_steps))
        progress = (step - self.config.warmup_steps) / max(
            1,
            self.config.max_steps - self.config.warmup_steps,
        )
        return self.config.lr * max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    def _update_lr(self, step: int) -> float:
        lr = self._cosine_lr(step)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr

    def train(
        self,
        dataset: Dataset,
        teacher_dataset: Dataset | None = None,
        log_every: int = 50,
    ) -> FinetuneReport:
        """Run fine-tuning loop.

        Args:
            dataset: Domain-specific training data.
            teacher_dataset: Optional teacher labels to mix in.
            log_every: Log every N steps.

        Returns:
            FinetuneReport with metrics.
        """
        config = self.config
        report = FinetuneReport(strategy=config.strategy)
        self._provenance_dataset = dataset
        self._used_synthetic_data = False

        dataset_provenance = build_provenance(
            student=self.student,
            config=getattr(
                getattr(self.student, "module", self.student),
                "config",
                self.config,
            ),
            dataset=dataset,
        )
        if dataset_provenance["labels"] == "mock" and not _student_allows_mock(self.student):
            raise ValueError(
                "Fine-tuning refuses a dataset with mock or unverified label provenance. "
                "Use real labeled data or enable config.student.allow_mock explicitly."
            )

        # Compute EWC Fisher if enabled
        if self.ewc is not None:
            self.ewc.used_synthetic_data = False
            self.ewc.compute_fisher(
                self.student,
                dataset,
                device=self.device,
                n_samples=min(200, len(cast(Sized, dataset))),
            )

        # DataLoader
        loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,
        )
        data_iter = iter(loader)

        t_start = time.time()
        best_loss = float("inf")
        loss_val = 0.0
        lr = config.lr
        step = 0

        while step < config.max_steps:
            # Update LR before optimizer step
            lr = self._update_lr(step)

            # Get batch
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Mix replay samples into batch if buffer is ready
            if self.replay is not None and self.replay.is_ready(min_size=config.batch_size):
                n_replay = max(1, int(config.batch_size * config.replay_ratio))
                replay_samples = self.replay.sample(n_replay)
                if replay_samples:
                    # Collate replay samples and replace part of batch
                    replay_batch = _collate_samples(replay_samples, self.device)
                    batch = _mix_batches(batch, replay_batch, n_replay)

            # Forward
            self.student.train()
            image = batch.get("image")
            if image is None:
                if not _student_allows_mock(self.student):
                    raise ValueError(
                        "Fine-tuning requires dataset images. Synthetic images are only "
                        "allowed when config.student.allow_mock is enabled."
                    )
                self._used_synthetic_data = True
                image = torch.randn(config.batch_size, 3, 384, 384, device=self.device)
            output = self.student(image, gt_actions=batch.get("ground_truth_actions"))

            # Loss
            if "loss" in output:
                loss = output["loss"]
            elif "actions" in output and "ground_truth_actions" in batch:
                loss = nn.functional.mse_loss(output["actions"], batch["ground_truth_actions"])
            else:
                if not _student_allows_mock(self.student):
                    raise ValueError(
                        "Fine-tuning requires a model loss or ground-truth actions. "
                        "The synthetic fallback objective is only allowed when "
                        "config.student.allow_mock is enabled."
                    )
                self._used_synthetic_data = True
                loss = output["actions"].pow(2).mean()  # Fallback

            # EWC penalty
            if self.ewc is not None and self.ewc.has_fisher:
                ewc_loss = self.ewc.penalty(self.student)
                loss = loss + ewc_loss

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.student.parameters() if p.requires_grad],
                max_norm=1.0,
            )
            self.optimizer.step()

            loss_val = loss.item()

            # Add batch samples to replay buffer
            if self.replay is not None:
                self.replay.add_batch(batch)

            # Track best
            if loss_val < best_loss:
                best_loss = loss_val

            # Logging
            if step % log_every == 0:
                elapsed = time.time() - t_start
                logger.info(
                    f"Finetune step {step}/{config.max_steps} | "
                    f"Loss: {loss_val:.4f} | LR: {lr:.2e} | "
                    f"Strategy: {config.strategy} | "
                    f"{step / max(elapsed, 1):.1f} steps/s"
                )

            # Checkpoint
            if step > 0 and step % config.save_every == 0:
                self._save_checkpoint(step, dataset=dataset)

            step += 1

        # Final checkpoint
        ckpt_path = self._save_checkpoint(step, tag="final", dataset=dataset)

        elapsed = time.time() - t_start
        report.total_steps = step
        report.elapsed_seconds = elapsed
        report.final_loss = loss_val
        report.best_loss = best_loss
        report.checkpoint_path = str(ckpt_path)
        report.ewc_used = self.ewc is not None
        report.replay_used = self.replay is not None

        logger.info(f"Fine-tuning complete: {step} steps, {elapsed:.0f}s, loss={loss_val:.4f}, best={best_loss:.4f}")
        return report

    def _save_checkpoint(
        self,
        step: int,
        tag: str | None = None,
        *,
        dataset: Dataset | None = None,
    ) -> Path:
        """Save fine-tuned model checkpoint."""
        name = tag or f"step_{step}"
        ckpt_path = self.output_dir / f"finetune_{name}.pt"
        state = {
            "global_step": step,
            "strategy": self.config.strategy,
            "student_state_dict": self.student.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "provenance": build_provenance(
                student=self.student,
                config=getattr(
                    getattr(self.student, "module", self.student),
                    "config",
                    self.config,
                ),
                dataset=dataset if dataset is not None else self._provenance_dataset,
                labels=(
                    "mock"
                    if self._used_synthetic_data or (self.ewc is not None and self.ewc.used_synthetic_data)
                    else None
                ),
            ),
        }
        torch.save(state, ckpt_path)
        logger.info(f"Finetune checkpoint: {ckpt_path}")
        return ckpt_path

    def load_pretrained(self, path: str | Path) -> None:
        """Load pretrained weights before fine-tuning."""
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        if "student_state_dict" in ckpt:
            self.student.load_state_dict(ckpt["student_state_dict"])
        else:
            self.student.load_state_dict(ckpt)
        logger.info(f"Loaded pretrained weights from {path}")
