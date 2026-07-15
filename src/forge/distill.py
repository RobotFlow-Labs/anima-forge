"""PRD-03: Knowledge Distillation Training Loop.

Trains the FORGE student model using teacher soft labels.
Three phases:
1. Bridge warmup (5K steps): only bridge + action head trainable
2. Full distillation (45K steps): bridge + action head + LoRA
3. Action fine-tune (10K steps): action head only, hard episodes

Usage:
    forge distill train --config configs/forge_nano.yaml
    forge distill train --config configs/forge_nano.yaml --max-steps 1000 --device cpu
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler

from forge.config import ForgeConfig
from forge.data.teacher_dataset import TeacherLabelDataset
from forge.errors import ForgeDataNotFoundError, ForgeModelNotFoundError
from forge.losses import ForgeDistillationLoss
from forge.provenance import build_provenance
from forge.student import FORGEStudent

logger = logging.getLogger(__name__)


def _preflight_student_weights(config: ForgeConfig) -> None:
    """Fail on missing real backbones before evaluating label provenance."""
    if config.student.allow_mock:
        return

    model_dir = Path(config.paths.model_dir).expanduser()
    required = (
        ("Vision encoder", config.student.vision_encoder),
        ("Language model", config.student.language_model),
    )
    for component, model_id in required:
        path = model_dir / model_id.replace("/", "--")
        if not path.is_dir():
            raise ForgeModelNotFoundError(
                component=component,
                model_id=model_id,
                path=path,
            )


def _coerce_distill_config(config: ForgeConfig) -> None:
    """Normalize legacy string hyper-parameters from YAML / env overrides."""
    numeric_int_fields = [
        "warmup_steps",
        "max_steps",
        "batch_size",
        "gradient_accumulation_steps",
        "eval_every",
        "save_every",
    ]
    numeric_float_fields = [
        "learning_rate",
        "weight_decay",
        "temperature",
        "alpha_kd",
        "alpha_task",
        "alpha_feat",
        "alpha_action",
    ]

    for field in numeric_int_fields:
        value = getattr(config.distill, field)
        if isinstance(value, str):
            try:
                setattr(config.distill, field, int(float(value)))
            except ValueError:
                logger.warning("Ignoring invalid distill int config for %s: %s", field, value)

    for field in numeric_float_fields:
        value = getattr(config.distill, field)
        if isinstance(value, str):
            try:
                setattr(config.distill, field, float(value))
            except ValueError:
                logger.warning("Ignoring invalid distill float config for %s: %s", field, value)


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Cosine schedule with linear warmup."""
    import math

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_forge(
    config: ForgeConfig,
    device: str | None = None,
    max_steps: int | None = None,
    checkpoint_dir: str | Path | None = None,
    resume_from: str | None = None,
    progress_callback: Callable[[dict[str, float | int]], None] | None = None,
) -> dict:
    """Main knowledge distillation training loop.

    Args:
        config: FORGE configuration
        device: Override device (cuda/cpu/mps)
        max_steps: Override max training steps
        checkpoint_dir: Directory for saving checkpoints
        resume_from: Path to checkpoint to resume from
        progress_callback: Optional bounded step/loss/ETA event callback

    Returns:
        Training summary with metrics
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    max_steps = max_steps or config.distill.max_steps
    if isinstance(max_steps, str):
        max_steps = int(float(max_steps))
    _coerce_distill_config(config)
    checkpoint_dir = Path(checkpoint_dir or config.paths.output_dir) / "checkpoints"

    logger.info(f"Starting FORGE distillation: device={device}, max_steps={max_steps}")

    _preflight_student_weights(config)

    # Load and validate teacher labels before initializing the heavyweight student.
    data_dir = Path(config.paths.data_dir) / "teacher_labels"
    metadata_path = data_dir / "metadata.json"
    if (not data_dir.exists()) or (not metadata_path.exists()):
        if config.student.allow_mock:
            logger.warning(
                "Teacher labels not found at %s; explicit allow_mock is enabled",
                data_dir,
            )
            dataset = _create_mock_dataset(data_dir, n_episodes=100)
        else:
            raise ForgeDataNotFoundError(
                f"Teacher labels not found at {data_dir}. Generate them with "
                "forge pipeline --stage labels before starting distillation."
            )
    else:
        dataset = TeacherLabelDataset(data_dir)

    labels_provenance = dataset.labels_provenance
    if labels_provenance != "real" and not config.student.allow_mock:
        dataset.close()
        raise ForgeDataNotFoundError(
            f"Teacher labels at {data_dir} are mock-derived or have no trusted provenance. "
            "Regenerate them from a real teacher and benchmark collector with "
            "`forge pipeline --stage labels`, or use `--allow-mock` only for an explicit "
            "test workflow."
        )

    student = FORGEStudent(config.student, model_dir=config.paths.model_dir)
    student = student.to(device)
    provenance = build_provenance(
        student=student,
        config=config,
        dataset=dataset,
        labels=labels_provenance,
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Bounded real-label validation sets may be smaller than the requested GPU
    # batch. Sampling with replacement is the standard way to keep the real
    # examples while honoring the configured batch/memory contract.
    bounded_sampler = None
    if len(dataset) < config.distill.batch_size:
        bounded_sampler = RandomSampler(
            dataset,
            replacement=True,
            num_samples=config.distill.batch_size,
        )
    dataloader = DataLoader(
        dataset,
        batch_size=config.distill.batch_size,
        sampler=bounded_sampler,
        shuffle=bounded_sampler is None,
        num_workers=0,  # Safe for HDF5
        drop_last=True,
    )

    # Loss function
    criterion = ForgeDistillationLoss(
        temperature=config.distill.temperature,
        alpha_kd=config.distill.alpha_kd,
        alpha_task=config.distill.alpha_task,
        alpha_feat=config.distill.alpha_feat,
        alpha_action=config.distill.alpha_action,
    )

    # Optimizer — only trainable params
    optimizer = AdamW(
        student.trainable_parameters(),
        lr=config.distill.learning_rate,
        weight_decay=config.distill.weight_decay,
    )

    # Scheduler
    scheduler = get_cosine_schedule_with_warmup(optimizer, config.distill.warmup_steps, max_steps)

    # Training state
    global_step = 0
    best_loss = float("inf")
    metrics_history = []

    # Resume from checkpoint
    if resume_from:
        global_step = _load_checkpoint(resume_from, student, optimizer, scheduler, device)
        logger.info(f"Resumed from step {global_step}")

    # Phase management
    phase = _get_phase(global_step, max_steps)
    _set_trainable_for_phase(student, phase)
    logger.info(f"Phase {phase}: {_phase_description(phase)}")

    t_start = time.time()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    data_iter = iter(dataloader)
    progress_interval = max(1, max_steps // 20)

    while global_step < max_steps:
        # Get batch (cycle through dataset)
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        # Move to device
        images = batch["image"].to(device)
        gt_actions = batch["ground_truth_actions"].to(device)
        teacher_logits = batch["teacher_action_logits"].to(device)
        teacher_mean = batch["teacher_action_mean"].to(device)
        teacher_std = batch["teacher_action_std"].to(device)
        confidence = batch["confidence"].to(device)

        _validate_action_batch(
            config,
            teacher_logits=teacher_logits,
            teacher_mean=teacher_mean,
            teacher_std=teacher_std,
            confidence=confidence,
            ground_truth=gt_actions,
        )

        teacher_vis = batch.get("teacher_vision_features")
        if teacher_vis is not None:
            teacher_vis = teacher_vis.to(device)

        # Forward pass
        student_out = student(images, gt_actions=gt_actions)

        # Compute losses
        losses = criterion(
            student_actions=student_out["actions"],
            teacher_action_logits=teacher_logits,
            ground_truth_actions=gt_actions,
            student_vision_features=student_out.get("vision_features"),
            teacher_vision_features=teacher_vis,
            teacher_action_mean=teacher_mean,
            teacher_action_std=teacher_std,
            teacher_confidence=confidence,
        )

        # Add diffusion loss if present
        total_loss = losses["total"]
        if "loss" in student_out:
            total_loss = total_loss + student_out["loss"]

        # Backward
        total_loss.backward()

        if (global_step + 1) % config.distill.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(student.trainable_parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Logging
        if global_step % 100 == 0:
            lr = scheduler.get_last_lr()[0]
            elapsed = time.time() - t_start
            steps_per_sec = (global_step + 1) / max(elapsed, 1)
            logger.info(
                f"Step {global_step}/{max_steps} | "
                f"Loss: {total_loss.item():.4f} (kd={losses['kd'].item():.4f}, "
                f"task={losses['task'].item():.4f}) | "
                f"LR: {lr:.2e} | {steps_per_sec:.1f} steps/s"
            )

        # Phase transition
        new_phase = _get_phase(global_step, max_steps)
        if new_phase != phase:
            phase = new_phase
            _set_trainable_for_phase(student, phase)
            logger.info(f"Phase transition → Phase {phase}: {_phase_description(phase)}")

        # Checkpoint
        if global_step > 0 and global_step % config.distill.save_every == 0:
            ckpt_path = checkpoint_dir / f"step_{global_step}.pt"
            _save_checkpoint(ckpt_path, student, optimizer, scheduler, global_step, provenance, config)
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_path = checkpoint_dir / "best.pt"
                _save_checkpoint(best_path, student, optimizer, scheduler, global_step, provenance, config)

        metrics_history.append(
            {
                "step": global_step,
                "total_loss": total_loss.item(),
                "kd_loss": losses["kd"].item(),
                "task_loss": losses["task"].item(),
            }
        )

        if total_loss.item() < best_loss:
            best_loss = total_loss.item()

        global_step += 1
        if progress_callback is not None and (
            global_step == 1 or global_step == max_steps or global_step % progress_interval == 0
        ):
            elapsed = time.time() - t_start
            steps_per_second = global_step / max(elapsed, 1e-9)
            event: dict[str, float | int] = {
                "step": global_step,
                "total_steps": max_steps,
                "loss": float(total_loss.item()),
                "steps_per_second": steps_per_second,
                "eta_seconds": max(0.0, (max_steps - global_step) / max(steps_per_second, 1e-9)),
            }
            if device.startswith("cuda") and torch.cuda.is_available():
                event["vram_gib"] = torch.cuda.memory_reserved(device) / 1024**3
            progress_callback(event)

    # Save final
    final_path = checkpoint_dir / "final.pt"
    _save_checkpoint(final_path, student, optimizer, scheduler, global_step, provenance, config)

    elapsed = time.time() - t_start
    loss_values = [metric["total_loss"] for metric in metrics_history]
    loss_window = min(100, max(1, len(loss_values) // 10))
    initial_loss_point = loss_values[0] if loss_values else 0.0
    final_loss_point = loss_values[-1] if loss_values else 0.0
    initial_loss = sum(loss_values[:loss_window]) / loss_window if loss_values else 0.0
    final_loss = sum(loss_values[-loss_window:]) / loss_window if loss_values else 0.0
    peak_memory: dict[str, float | bool | None] = {
        "peak_allocated_gib": None,
        "peak_reserved_gib": None,
        "total_gib": None,
        "peak_reserved_utilization": None,
        "target_60_80_percent_met": False,
    }
    if device.startswith("cuda") and torch.cuda.is_available():
        total_bytes = torch.cuda.get_device_properties(device).total_memory
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
        divisor = 1024**3
        utilization = peak_reserved / total_bytes
        peak_memory = {
            "peak_allocated_gib": round(peak_allocated / divisor, 3),
            "peak_reserved_gib": round(peak_reserved / divisor, 3),
            "total_gib": round(total_bytes / divisor, 3),
            "peak_reserved_utilization": round(utilization, 4),
            "target_60_80_percent_met": 0.60 <= utilization <= 0.80,
        }
    summary = {
        "total_steps": global_step,
        "elapsed_seconds": elapsed,
        "steps_per_second": global_step / max(elapsed, 1e-9),
        "loss_window_steps": loss_window,
        "initial_loss_point": initial_loss_point,
        "final_loss_point": final_loss_point,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_reduction": initial_loss - final_loss,
        "loss_reduction_percent": (100.0 * (initial_loss - final_loss) / abs(initial_loss) if initial_loss else 0.0),
        "best_loss": best_loss,
        "cuda_memory": peak_memory,
        "checkpoint_dir": str(checkpoint_dir),
        "device": device,
        "provenance": provenance,
    }

    logger.info(f"Training complete: {global_step} steps, {elapsed:.0f}s, final loss={final_loss:.4f}")
    return summary


def _validate_action_batch(
    config: ForgeConfig,
    *,
    teacher_logits: torch.Tensor,
    teacher_mean: torch.Tensor,
    teacher_std: torch.Tensor,
    confidence: torch.Tensor,
    ground_truth: torch.Tensor,
) -> None:
    """Reject label/student action contracts that cannot be distilled truthfully."""
    action_shape = teacher_logits.shape
    if any(value.shape != action_shape for value in (teacher_mean, teacher_std, confidence)):
        raise ValueError("Teacher action statistics must have identical batch, horizon, and action dimensions")

    chunk_head = config.student.action_head_type == "chunk"
    expected_rank = 3 if chunk_head else 2
    if teacher_logits.ndim != expected_rank:
        head = config.student.action_head_type
        raise ValueError(f"Teacher labels with shape {tuple(action_shape)} are incompatible with {head!r} action head")
    if teacher_logits.shape[-1] != config.student.action_dim:
        raise ValueError(
            f"Teacher action dimension {teacher_logits.shape[-1]} does not match student action_dim "
            f"{config.student.action_dim}"
        )
    if chunk_head and teacher_logits.shape[1] != config.student.action_horizon:
        raise ValueError(
            f"Teacher horizon {teacher_logits.shape[1]} does not match student action_horizon "
            f"{config.student.action_horizon}"
        )
    if ground_truth.shape != teacher_logits.shape:
        raise ValueError(
            f"Ground-truth action shape {tuple(ground_truth.shape)} does not match teacher labels "
            f"{tuple(teacher_logits.shape)}"
        )


def _get_phase(step: int, max_steps: int) -> int:
    """Determine training phase from step count.

    Phase 1 (0-10%): Bridge warmup — only bridge + action head
    Phase 2 (10-83%): Full distillation — bridge + action head + LoRA
    Phase 3 (83-100%): Action fine-tune — action head only
    """
    if step < max_steps * 0.1:
        return 1
    elif step < max_steps * 0.83:
        return 2
    else:
        return 3


def _set_trainable_for_phase(student: FORGEStudent, phase: int) -> None:
    """Set which parameters are trainable for each phase."""
    # First freeze everything
    for param in student.parameters():
        param.requires_grad = False

    if phase == 1:
        # Bridge + Action Head only
        for param in student.bridge.parameters():
            param.requires_grad = True
        for param in student.action_head.parameters():
            param.requires_grad = True
    elif phase == 2:
        # Bridge + Action Head + LoRA
        for param in student.bridge.parameters():
            param.requires_grad = True
        for param in student.action_head.parameters():
            param.requires_grad = True
        for name, param in student.language.named_parameters():
            if "lora" in name.lower():
                param.requires_grad = True
    elif phase == 3:
        # Action Head only
        for param in student.action_head.parameters():
            param.requires_grad = True


def _phase_description(phase: int) -> str:
    return {
        1: "Bridge warmup (bridge + action head)",
        2: "Full distillation (bridge + LoRA + action head)",
        3: "Action fine-tune (action head only)",
    }.get(phase, "unknown")


def _save_checkpoint(
    path: Path,
    student: FORGEStudent,
    optimizer: AdamW,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    step: int,
    provenance: dict[str, str],
    config: ForgeConfig,
) -> None:
    torch.save(
        {
            "step": step,
            "model_state_dict": student.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "provenance": provenance,
            "student_config": asdict(config.student),
        },
        path,
    )
    logger.info(f"Checkpoint saved: {path}")


def _load_checkpoint(
    path: str,
    student: FORGEStudent,
    optimizer: AdamW,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    device: str,
) -> int:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    student.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt["step"]


def _create_mock_dataset(data_dir: Path, n_episodes: int = 100) -> TeacherLabelDataset:
    """Create mock teacher labels for testing."""
    from forge.data.label_writer import LabelWriter
    from forge.types import EpisodeData

    data_dir.mkdir(parents=True, exist_ok=True)
    writer = LabelWriter(
        str(data_dir),
        episodes_per_file=50,
        save_vision_features=False,
        labels_provenance="mock",
    )

    timesteps = 10
    for i in range(n_episodes):
        episode = EpisodeData(
            episode_id=f"mock_{i}",
            task_id=f"task_{i % 5}",
            language_instruction=f"mock task {i % 5}",
            timesteps=timesteps,
            images=np.random.randint(0, 255, (timesteps, 64, 64, 3), dtype=np.uint8),
            proprioception=np.random.randn(timesteps, 7).astype(np.float32),
            teacher_action_logits=np.random.randn(timesteps, 7).astype(np.float32) * 0.1,
            teacher_action_mean=np.random.randn(timesteps, 7).astype(np.float32) * 0.1,
            teacher_action_std=np.abs(np.random.randn(timesteps, 7).astype(np.float32) * 0.1) + 0.01,
            teacher_vision_features=None,
            confidence=np.random.rand(timesteps, 7).astype(np.float32),
            ground_truth_actions=np.random.randn(timesteps, 7).astype(np.float32) * 0.1,
            success=True,
        )
        writer.write_episode(episode)

    writer.finalize()
    return TeacherLabelDataset(str(data_dir))
