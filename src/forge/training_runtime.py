"""Runtime helpers for truthful production-training CLI commands."""

from __future__ import annotations

import json
import math
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch

from forge.config import ForgeConfig


class TrainingRuntimeError(RuntimeError):
    """Raised when a production training run cannot be prepared safely."""


def _sanitize_state(value: Any) -> tuple[Any, bool]:
    if isinstance(value, float) and not math.isfinite(value):
        return None, True
    if isinstance(value, dict):
        clean: dict[str, Any] = {}
        changed = False
        for key, item in value.items():
            clean_item, item_changed = _sanitize_state(item)
            clean[str(key)] = clean_item
            changed = changed or item_changed
        return clean, changed
    if isinstance(value, (list, tuple)):
        clean_items = []
        changed = False
        for item in value:
            clean_item, item_changed = _sanitize_state(item)
            clean_items.append(clean_item)
            changed = changed or item_changed
        return clean_items, changed
    if isinstance(value, Path):
        return str(value), False
    return value, False


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Atomically replace a JSON state file with strict, portable JSON."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        clean_payload, sanitized = _sanitize_state(payload)
        if sanitized and isinstance(clean_payload, dict):
            clean_payload.setdefault("note", "Non-finite numeric values were replaced with null.")
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(clean_payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def read_heartbeat(run_dir: str | Path) -> dict[str, Any]:
    """Read a run heartbeat, producing a useful error for absent/corrupt state."""
    heartbeat = Path(run_dir) / "train_state.json"
    if not heartbeat.is_file():
        raise TrainingRuntimeError(f"Training heartbeat not found: {heartbeat}")
    try:
        payload = json.loads(heartbeat.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingRuntimeError(f"Training heartbeat is unreadable: {heartbeat}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TrainingRuntimeError(f"Training heartbeat is not a JSON object: {heartbeat}")
    return payload


def create_run_dir(output_dir: str | Path) -> Path:
    """Create a collision-resistant run directory under the configured output."""
    root = Path(output_dir).expanduser() / "train-runs"
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = root / f"{timestamp}-{uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir.resolve()


def latest_run_dir(output_dir: str | Path) -> Path:
    """Return the newest run containing a heartbeat."""
    root = Path(output_dir).expanduser() / "train-runs"
    candidates = [path.parent for path in root.glob("*/train_state.json") if path.is_file()]
    if not candidates:
        raise TrainingRuntimeError(f"No training runs found under: {root}")
    return max(candidates, key=lambda path: (path / "train_state.json").stat().st_mtime)


def process_is_running(pid: int | None) -> bool:
    """Return whether a process exists without sending it a signal."""
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def process_start_time_ticks(pid: int | None) -> int | None:
    """Return a Linux process birth token suitable for detecting PID reuse."""
    if not pid or pid <= 0:
        return None
    try:
        stat = (Path("/proc") / str(pid) / "stat").read_text(encoding="utf-8")
        fields_after_command = stat.rsplit(")", 1)[1].split()
        return int(fields_after_command[19])
    except (IndexError, OSError, ValueError):
        return None


def process_identity_matches(pid: int | None, start_time_ticks: object) -> bool:
    """Return whether ``pid`` is still the exact process recorded by a heartbeat."""
    if not isinstance(start_time_ticks, int) or start_time_ticks <= 0:
        return False
    return process_start_time_ticks(pid) == start_time_ticks


def _cuda_total_gib(device: str) -> float:
    index = 0
    if ":" in device:
        try:
            index = int(device.rsplit(":", 1)[1])
        except ValueError:
            index = 0
    properties = torch.cuda.get_device_properties(index)
    return properties.total_memory / (1024**3)


def choose_batch_size(
    config: ForgeConfig,
    *,
    device: str,
    requested: int | None,
    dataset_size: int | None = None,
    target_utilization: float = 0.70,
) -> tuple[int, dict[str, Any]]:
    """Pick a training batch using the profiler and a 70% VRAM target."""
    if requested is not None:
        if requested < 1:
            raise TrainingRuntimeError("Batch size must be at least 1")
        if dataset_size is not None and requested > dataset_size:
            raise TrainingRuntimeError(f"Batch size {requested} exceeds the {dataset_size}-sample dataset")
        return requested, {"source": "explicit", "batch_size": requested}

    if not device.startswith("cuda"):
        batch_size = max(1, int(config.distill.batch_size))
        if dataset_size:
            batch_size = min(batch_size, dataset_size)
        return batch_size, {"source": "config_cpu", "batch_size": batch_size}

    if not torch.cuda.is_available():
        raise TrainingRuntimeError("CUDA batch sizing requested but CUDA is unavailable")

    from forge.profiler.vram import estimate_vram

    total_gib = _cuda_total_gib(device)
    estimate = estimate_vram(config.student, gpu_vram_gb=total_gib)
    target_mb = total_gib * 1024 * target_utilization
    activation_mb = max(float(estimate.per_sample_activation_mb), 1e-6)
    remaining_mb = target_mb - float(estimate.training_fp16_mb)
    vram_target_batch_size = max(1, math.floor(remaining_mb / activation_mb))
    profiler_cap = max(1, int(estimate.recommended_batch_size))
    batch_size = min(vram_target_batch_size, profiler_cap)
    limiting_factor = "profiler_recommended_batch_size" if profiler_cap < vram_target_batch_size else None
    if dataset_size and dataset_size < batch_size:
        batch_size = dataset_size
        limiting_factor = "dataset_size"

    estimated_mb = float(estimate.training_fp16_mb) + batch_size * activation_mb
    utilization = estimated_mb / (total_gib * 1024)
    target_met = 0.60 <= utilization <= 0.80
    details = {
        "source": "vram_estimate",
        "batch_size": batch_size,
        "target_batch_size": vram_target_batch_size,
        "profiler_batch_cap": profiler_cap,
        "gpu_vram_gib": round(total_gib, 3),
        "target_utilization": target_utilization,
        "estimated_utilization": round(utilization, 4),
        "estimated_target_met": target_met,
        "limiting_factor": limiting_factor,
        "training_base_mb": round(float(estimate.training_fp16_mb), 2),
        "per_sample_activation_mb": round(activation_mb, 2),
        "profiler_recommended_batch_size": estimate.recommended_batch_size,
    }
    if limiting_factor == "dataset_size":
        details["note"] = (
            "The available dataset is smaller than the selected batch; "
            "the loader uses the full dataset without duplicating samples."
        )
    elif limiting_factor == "profiler_recommended_batch_size":
        details["note"] = (
            "The profiler recommendation caps the formula-derived target to avoid "
            "an unsafe batch on uncalibrated hardware."
        )
    if not target_met:
        dataset_guidance = ""
        if limiting_factor == "dataset_size":
            dataset_guidance = (
                " The real dataset is too small for the automatically selected batch; "
                "provide enough samples to fill the target batch or pass an explicit "
                "--batch-size after validating measured device utilization."
            )
        raise TrainingRuntimeError(
            "Automatic CUDA batch sizing cannot meet the required 60-80% VRAM utilization range: "
            f"estimated {utilization:.1%} at batch size {batch_size} "
            f"(base={float(estimate.training_fp16_mb):.1f} MiB, available target headroom={remaining_mb:.1f} MiB). "
            "Choose a compatible student variant/GPU or pass an explicit --batch-size after validating memory use."
            f"{dataset_guidance}"
        )
    return batch_size, details


def resolve_teacher_label_dir(data_dir: str | Path) -> Path:
    """Resolve a real teacher-label dataset and refuse silent mock creation."""
    base = Path(data_dir).expanduser()
    candidates = (base / "teacher_labels", base)
    for candidate in candidates:
        if (candidate / "metadata.json").is_file():
            return candidate.resolve()
    expected = candidates[0]
    raise TrainingRuntimeError(
        f"Teacher labels not found at {expected}. Run the label stage or provide a config "
        "whose paths.data_dir contains teacher_labels/metadata.json."
    )


def build_production_trainer(
    config: ForgeConfig,
    *,
    device: str,
    run_dir: str | Path,
    batch_size: int | None,
):
    """Build the real student, label dataset, loss, and ProductionTrainer."""
    from forge.data.teacher_dataset import TeacherLabelDataset
    from forge.losses import ForgeDistillationLoss
    from forge.student import FORGEStudent
    from forge.trainer import ProductionTrainer

    label_dir = resolve_teacher_label_dir(config.paths.data_dir)
    dataset = TeacherLabelDataset(label_dir)
    if dataset.labels_provenance != "real" and not config.student.allow_mock:
        dataset.close()
        raise TrainingRuntimeError(
            f"Teacher labels at {label_dir} are mock-derived or untrusted. Regenerate them from "
            "a real teacher and benchmark collector, or pass --allow-mock only for an explicit "
            "test workflow."
        )
    if len(dataset) < 1:
        dataset.close()
        raise TrainingRuntimeError(f"Teacher-label dataset is empty: {label_dir}")
    try:
        selected_batch, batch_details = choose_batch_size(
            config,
            device=device,
            requested=batch_size,
            dataset_size=len(dataset),
        )
        config.distill.batch_size = selected_batch
        config.paths.output_dir = str(Path(run_dir).resolve())

        student = FORGEStudent(config.student, model_dir=config.paths.model_dir)
        loss_fn = ForgeDistillationLoss(
            temperature=config.distill.temperature,
            alpha_kd=config.distill.alpha_kd,
            alpha_task=config.distill.alpha_task,
            alpha_feat=config.distill.alpha_feat,
            alpha_action=config.distill.alpha_action,
        )
        trainer = ProductionTrainer(
            student=student,
            dataset=dataset,
            loss_fn=loss_fn,
            config=config,
            device=device,
            checkpoint_dir=str(run_dir),
        )
    except BaseException:
        dataset.close()
        raise
    return trainer, label_dir, batch_details


def cuda_memory_snapshot(device: str) -> dict[str, float | None]:
    """Capture portable CUDA memory facts for a heartbeat."""
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return {"allocated_gib": None, "reserved_gib": None, "free_gib": None}
    index = 0
    if ":" in device:
        try:
            index = int(device.rsplit(":", 1)[1])
        except ValueError:
            index = 0
    with torch.cuda.device(index):
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        allocated = torch.cuda.memory_allocated(index)
        reserved = torch.cuda.memory_reserved(index)
    divisor = 1024**3
    return {
        "allocated_gib": round(allocated / divisor, 3),
        "reserved_gib": round(reserved / divisor, 3),
        "free_gib": round(free_bytes / divisor, 3),
        "total_gib": round(total_bytes / divisor, 3),
        "allocated_utilization": round(allocated / total_bytes, 4),
        "reserved_utilization": round(reserved / total_bytes, 4),
        "target_met": 0.60 <= (reserved / total_bytes) <= 0.80,
    }
