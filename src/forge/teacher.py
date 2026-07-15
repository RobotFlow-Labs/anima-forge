"""PRD-01: Teacher Label Generation.

Generates soft labels from teacher VLA models on benchmark tasks.
Captures action logits, vision features, and confidence scores
for knowledge distillation in PRD-03.

Supports:
- OpenVLA 7B (primary teacher)
- RDT2-FM 7B
- Any HuggingFace VLA model

Usage:
    forge labels generate --config configs/forge_nano.yaml
    forge labels generate --teacher openvla/openvla-7b --benchmark libero_spatial
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from forge.config import ForgeConfig, TeacherConfig
from forge.errors import ForgeDataNotFoundError, ForgeModelNotFoundError
from forge.hf_compat import configure_transformers_module_cache
from forge.openvla_loader import load_image_text_model
from forge.processor_compat import install_legacy_tokenization_exports
from forge.types import EpisodeData

logger = logging.getLogger(__name__)


class ExtractionHooks:
    """Forward hooks to extract intermediate representations from teacher model."""

    def __init__(self):
        self.vision_features: torch.Tensor | None = None
        self._handles: list[Any] = []

    def register(self, model: torch.nn.Module, extract_vision: bool = True) -> None:
        """Register forward hooks on the teacher model."""
        if extract_vision:
            # Find vision encoder output layer
            vision_encoder = _find_vision_encoder(model)
            if vision_encoder is not None:
                handle = vision_encoder.register_forward_hook(self._capture_vision)
                self._handles.append(handle)
                logger.info("Registered vision feature extraction hook")

    def _capture_vision(self, module: Any, input: Any, output: Any) -> None:
        """Capture vision encoder output."""
        if isinstance(output, torch.Tensor):
            self.vision_features = output.detach()
        elif hasattr(output, "last_hidden_state"):
            self.vision_features = output.last_hidden_state.detach()

    def remove(self) -> None:
        """Remove all hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def reset(self) -> None:
        """Reset captured features."""
        self.vision_features = None


def _find_vision_encoder(model: torch.nn.Module) -> torch.nn.Module | None:
    """Find the vision encoder module in a VLA model."""
    # Common names for vision encoders in VLA models
    for attr_name in ["vision_backbone", "vision_encoder", "visual", "vit", "image_encoder"]:
        if hasattr(model, attr_name):
            return getattr(model, attr_name)

    # Search nested modules
    for name, module in model.named_modules():
        if "vision" in name.lower() and "encoder" in name.lower():
            return module
        if "siglip" in name.lower() or "clip" in name.lower():
            return module

    logger.warning("Could not find vision encoder in teacher model")
    return None


def _patch_meta_tensor_linspace():
    """Patch torch.linspace to avoid meta tensor issues with timm + transformers 5.x."""
    if getattr(torch, "_forge_linspace_patched", False):
        return
    _orig = torch.linspace

    def _safe_linspace(*args, **kwargs):
        result = _orig(*args, **kwargs)
        if result.device.type == "meta":
            kw = {k: v for k, v in kwargs.items() if k != "device"}
            result = _orig(*args, **kw, device="cpu")
        return result

    torch.linspace = _safe_linspace
    setattr(torch, "_forge_linspace_patched", True)


def _resolve_teacher_source(model_path: str | Path) -> list[Path]:
    """Generate candidate teacher checkpoint sources for robust loading."""
    source = Path(model_path)
    candidates: list[Path] = []
    seen: set[str] = set()

    def _add(path_like: Path | str) -> None:
        path = Path(path_like)
        if str(path) in seen:
            return
        seen.add(str(path))

        # Skip invalid absolute paths that do not exist locally to avoid sending
        # them to the HF loader as repo IDs.
        if path.is_absolute() and not path.exists():
            return

        candidates.append(path)

    if source.exists():
        _add(source)
    else:
        # Common local dataset naming convention uses `namespace--repo`.
        name = source.name
        if "--" in name:
            _add(source.with_name(name.replace("--", "/")))
        model_dir = Path(os.environ.get("FORGE_MODEL_DIR", "")).expanduser()
        if model_dir:
            _add(model_dir / name)
            if "--" in name:
                _add(model_dir / name.replace("--", "/"))
        _add(source)

    # Keep deterministic order, remove duplicates.
    deduped = []
    dedup_seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key not in dedup_seen:
            deduped.append(path)
            dedup_seen.add(key)
    return deduped


def _teacher_hint_for_exception(model_path: str | Path, exc: Exception) -> str | None:
    """Return production-grade troubleshooting hint for known teacher load failures."""
    message = str(exc).lower()
    candidate = str(model_path).lower()
    if "timm version must be" in message and "openvla" in candidate:
        try:
            import timm

            timm_version = getattr(timm, "__version__", "unknown")
        except Exception:
            timm_version = "unknown"

        return (
            f"OpenVLA teacher compatibility warning (detected {candidate}). "
            f"This environment reports timm={timm_version}. OpenVLA expects timm >=0.9.10 and <1.0.0 "
            "for legacy checkpoints. If this is a compatibility mismatch, use a matching teacher variant or "
            "run with a GPU/weights bundle aligned to your installed timm version."
        )

    if "out of memory" in message and "teacher" in candidate:
        return f"Teacher load OOM for {candidate} — retry with CPU device for label generation."

    return None


def load_teacher(
    model_path: str | Path,
    device: str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> torch.nn.Module:
    """Load a teacher VLA model.

    Supports:
    - OpenVLA (transformers AutoModelForImageTextToText)
    - RDT2 (custom loading)
    - Any HuggingFace model with generate() method
    """
    configure_transformers_module_cache(model_path)

    _patch_meta_tensor_linspace()

    model_path = Path(model_path)
    logger.info(f"Loading teacher model from {model_path}")
    errors: list[str] = []

    candidates = _resolve_teacher_source(model_path)
    if not candidates:
        raise ValueError(f"No candidate model paths found for teacher source: {model_path}")

    for candidate in candidates:
        try:
            local_files_only = candidate.exists()
            model = load_image_text_model(
                candidate,
                dtype=dtype,
                # Avoid Accelerate's NVML-dependent device-map dispatch. The
                # checkpoint is already loaded at the requested dtype; a normal
                # PyTorch move works on CUDA hosts even when NVML is unavailable.
                device_map=None,
                local_files_only=local_files_only,
            )
            if device == "cpu":
                model = model.to(dtype=torch.float32)  # bf16 not well supported on CPU
            else:
                model = model.to(device=device)

            model.eval()
            logger.info(f"Teacher loaded: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B params")
            logger.info(f"Teacher source resolved: {candidate}")
            return model
        except Exception as e:
            hint = _teacher_hint_for_exception(candidate, e)
            if hint:
                logger.warning("%s", hint)
                errors.append(f"{candidate}: {hint}")
            errors.append(f"{candidate}: {e}")

    logger.error("Failed to load teacher")
    for msg in errors:
        logger.error(f"  - {msg}")
    raise RuntimeError("Teacher loading failed. Tried: " + "; ".join(str(c) for c in candidates))


def load_processor(model_path: str | Path) -> Any:
    """Load the processor/tokenizer for the teacher model."""
    configure_transformers_module_cache(model_path)
    install_legacy_tokenization_exports()

    from transformers import AutoProcessor

    errors: list[str] = []
    for candidate in _resolve_teacher_source(model_path):
        try:
            return AutoProcessor.from_pretrained(
                str(candidate),
                trust_remote_code=True,
                local_files_only=candidate.exists(),
                use_fast=False,
            )
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")

    raise RuntimeError("Processor loading failed. Tried: " + "; ".join(errors))


def compute_action_confidence(action_std: np.ndarray) -> np.ndarray:
    """Compute confidence from action distribution std.

    High confidence = low std. Normalized to [0, 1].
    """
    values = np.asarray(action_std)
    if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Action standard deviation must contain finite non-negative values")
    return 1.0 / (1.0 + values)


def generate_teacher_labels(
    config: ForgeConfig,
    teacher_override: str | None = None,
    max_episodes: int | None = None,
    device: str | None = None,
) -> dict:
    """Generate labels from a real local dataset, with explicit mock fallback only."""
    try:
        return _generate_real_teacher_labels(
            config,
            teacher_override=teacher_override,
            max_episodes=max_episodes,
            device=device,
        )
    except (ForgeDataNotFoundError, ForgeModelNotFoundError, FileNotFoundError) as exc:
        if not config.student.allow_mock:
            raise
        logger.warning("Real teacher-label inputs unavailable; explicit allow_mock permits fallback: %s", exc)
        return _generate_mock_teacher_labels(
            config,
            teacher_override=teacher_override,
            max_episodes=max_episodes,
            device=device,
        )


def _generate_real_teacher_labels(
    config: ForgeConfig,
    *,
    teacher_override: str | None,
    max_episodes: int | None,
    device: str | None,
) -> dict:
    """Run a registered teacher on genuine local robot demonstrations."""
    if max_episodes == 0:
        raise ForgeDataNotFoundError(
            "Real teacher-label generation requires at least one episode; max_episodes=0 cannot "
            "produce trusted labels. Use `--allow-mock` only for an explicit empty test workflow."
        )

    teacher_path = teacher_override or str(config.paths.teacher_path)
    output_dir = Path(config.paths.data_dir) / "teacher_labels"
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

    dataset_path = _resolve_real_teacher_dataset(config.teacher.dataset)
    from forge.data.real_robot_episodes import load_real_robot_episodes

    real_episodes = load_real_robot_episodes(
        dataset_path,
        max_episodes=max_episodes,
        max_steps=config.teacher.max_steps_per_episode,
    )
    if not real_episodes:
        raise ForgeDataNotFoundError(f"Real teacher dataset {dataset_path} produced no episodes")

    teacher_model_path = Path(teacher_path)
    if not teacher_model_path.is_dir():
        raise ForgeModelNotFoundError(
            component=f"Teacher {config.teacher.adapter}",
            model_id=config.paths.teacher.replace("--", "/"),
            path=teacher_model_path,
        )

    from forge.teachers.registry import get_registry

    adapter = get_registry().create(config.teacher.adapter)
    adapter.load(teacher_model_path, device=device, dtype=dtype)
    info = adapter.info()
    expected_action_dim = real_episodes[0].ground_truth_actions.shape[1]
    if info.action_dim != expected_action_dim:
        adapter.unload()
        raise ValueError(
            f"Teacher {info.name} action dim {info.action_dim} does not match real dataset "
            f"action dim {expected_action_dim} at {dataset_path}"
        )

    data_root = output_dir.parent
    data_root.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=".teacher_labels-", dir=data_root))
    t_start = time.time()

    from forge.data.label_writer import LabelWriter

    writer = LabelWriter(
        output_dir=str(temporary_dir),
        schema_version="1.0",
        save_vision_features=config.teacher.save_vision_features,
        save_attention=config.teacher.save_attention,
        labels_provenance="real",
        source_metadata={
            "teacher_adapter": info.name,
            "teacher_checkpoint": str(teacher_model_path.resolve()),
            "dataset": str(dataset_path.resolve()),
            "dataset_format": "lerobot-episode-parquet",
        },
    )

    successful_episodes = 0
    known_outcome_episodes = 0
    try:
        for source_episode in real_episodes:
            episode = _infer_real_teacher_episode(
                adapter,
                source_episode,
                save_vision_features=config.teacher.save_vision_features,
            )
            writer.write_episode(episode)
            successful_episodes += int(episode.success is True)
            known_outcome_episodes += int(episode.success is not None)
        metadata = writer.finalize()
        _publish_label_directory(temporary_dir, output_dir)
    except Exception:
        writer.finalize()
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
    finally:
        adapter.unload()

    elapsed = time.time() - t_start
    total_episodes = len(real_episodes)
    summary = {
        "total_episodes": total_episodes,
        "successful_episodes": successful_episodes,
        "success_unknown_episodes": total_episodes - known_outcome_episodes,
        "success_rate": successful_episodes / known_outcome_episodes if known_outcome_episodes else None,
        "elapsed_seconds": elapsed,
        "output_dir": str(output_dir),
        "teacher_model": str(teacher_model_path),
        "teacher_adapter": info.name,
        "benchmark": config.teacher.benchmark,
        "dataset": str(dataset_path.resolve()),
        "metadata": metadata,
        "provenance": {"labels": "real"},
    }
    logger.info(
        "Real label generation complete: %d episodes, %d/%d successful, %.1fs",
        total_episodes,
        successful_episodes,
        total_episodes,
        elapsed,
    )
    return summary


def _resolve_real_teacher_dataset(configured: str) -> Path:
    """Resolve a portable exact dataset path or local dataset name."""
    exact = Path(configured).expanduser()
    if exact.is_dir():
        return exact

    project_root = Path(__file__).resolve().parents[2]
    dataset_root_value = os.environ.get("FORGE_DATASET_DIR") or os.environ.get("FORGE_DATA_DIR")
    dataset_root = (
        Path(dataset_root_value).expanduser() if dataset_root_value else project_root.parent.parent / "datasets"
    )
    candidate = dataset_root / configured
    if candidate.is_dir():
        return candidate
    raise ForgeDataNotFoundError(
        f"Real teacher dataset {configured!r} was not found at {exact} or {candidate}. "
        "Set FORGE_TEACHER_DATASET to the local LeRobot dataset path, or use "
        "`--allow-mock` only for an explicit synthetic test workflow."
    )


def _infer_real_teacher_episode(adapter: Any, source_episode: Any, *, save_vision_features: bool) -> EpisodeData:
    """Run genuine teacher inference for every frame in one real episode."""
    action_logits: list[np.ndarray] = []
    action_mean: list[np.ndarray] = []
    action_std: list[np.ndarray] = []
    confidence: list[np.ndarray] = []
    vision_features: list[np.ndarray | None] = []

    for index in range(source_episode.timesteps):
        chunk = adapter.predict(
            source_episode.images[index],
            source_episode.instruction,
            source_episode.proprioception[index],
        )
        chunk_actions = np.asarray(chunk.actions, dtype=np.float32)
        chunk_mean = np.asarray(chunk.action_mean, dtype=np.float32)
        chunk_std = np.asarray(chunk.action_std, dtype=np.float32)
        chunk_confidence = np.asarray(chunk.confidence, dtype=np.float32)
        if chunk_actions.ndim != 2:
            raise ValueError(f"Teacher returned action chunk shape {chunk_actions.shape}; expected (H, D)")
        for name, value in (
            ("action_mean", chunk_mean),
            ("action_std", chunk_std),
            ("confidence", chunk_confidence),
        ):
            if value.shape != chunk_actions.shape:
                raise ValueError(f"Teacher returned {name} shape {value.shape}; expected {chunk_actions.shape}")
        action_logits.append(chunk_actions)
        action_mean.append(chunk_mean)
        action_std.append(chunk_std)
        confidence.append(chunk_confidence)

        if save_vision_features:
            feature = None if chunk.vision_features is None else np.asarray(chunk.vision_features, dtype=np.float16)
            vision_features.append(feature)

    teacher_vision_features = None
    available_features = [feature for feature in vision_features if feature is not None]
    if available_features and len(available_features) != source_episode.timesteps:
        missing = [index for index, feature in enumerate(vision_features) if feature is None]
        raise ValueError(f"Teacher returned vision features for only part of the episode; missing timesteps {missing}")
    if available_features:
        try:
            teacher_vision_features = np.stack(available_features)
        except ValueError as exc:
            raise ValueError(f"Teacher returned inconsistent vision feature shapes: {exc}") from exc

    return EpisodeData(
        episode_id=source_episode.episode_id,
        task_id=source_episode.task_id,
        language_instruction=source_episode.instruction,
        timesteps=source_episode.timesteps,
        images=source_episode.images,
        proprioception=source_episode.proprioception,
        teacher_action_logits=np.stack(action_logits),
        teacher_action_mean=np.stack(action_mean),
        teacher_action_std=np.stack(action_std),
        teacher_vision_features=teacher_vision_features,
        confidence=np.stack(confidence),
        ground_truth_actions=source_episode.ground_truth_actions,
        success=source_episode.success,
    )


def _publish_label_directory(temporary_dir: Path, output_dir: Path) -> None:
    """Publish a complete label set without exposing partially-written HDF5."""
    backup = output_dir.with_name(f".{output_dir.name}.backup-{time.time_ns()}")
    had_previous = output_dir.exists()
    if had_previous:
        output_dir.rename(backup)
    try:
        temporary_dir.rename(output_dir)
    except Exception:
        if had_previous and backup.exists() and not output_dir.exists():
            backup.rename(output_dir)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def _generate_mock_teacher_labels(
    config: ForgeConfig,
    *,
    teacher_override: str | None,
    max_episodes: int | None,
    device: str | None,
) -> dict:
    """Generate explicitly opted-in synthetic labels for tests and demos."""
    teacher_path = teacher_override or str(config.paths.teacher_path)
    output_dir = Path(config.paths.data_dir) / "teacher_labels"
    output_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

    logger.info(f"Generating explicit mock teacher labels: {teacher_path} → {output_dir}")
    logger.info(f"Device: {device}, dtype: {dtype}")

    # Load teacher
    teacher = load_teacher(teacher_path, device=device, dtype=dtype)
    processor = load_processor(teacher_path)

    # Set up extraction hooks
    hooks = ExtractionHooks()
    hooks.register(teacher, extract_vision=config.teacher.save_vision_features)

    # Initialize label writer (local import to avoid circular)
    from forge.data.label_writer import LabelWriter

    writer = LabelWriter(
        output_dir=str(output_dir),
        schema_version="1.0",
        save_vision_features=config.teacher.save_vision_features,
        save_attention=config.teacher.save_attention,
        labels_provenance="mock",
        source_metadata={"mode": "explicit-mock", "teacher_checkpoint": teacher_path},
    )

    # Load benchmark tasks
    tasks = _load_benchmark_tasks(config.teacher.benchmark)
    total_episodes = 0
    successful_episodes = 0

    t_start = time.time()

    for task_idx, task in enumerate(tasks):
        if max_episodes and total_episodes >= max_episodes:
            break

        logger.info(f"Task {task_idx + 1}/{len(tasks)}: {task.get('instruction', 'unknown')}")

        for ep_idx in range(config.teacher.episodes_per_task):
            if max_episodes and total_episodes >= max_episodes:
                break

            episode = _collect_episode(
                teacher=teacher,
                processor=processor,
                task=task,
                hooks=hooks,
                config=config.teacher,
                device=device,
                episode_id=f"{config.teacher.benchmark}_{task_idx}_{ep_idx}",
            )

            writer.write_episode(episode)
            total_episodes += 1
            if episode.success:
                successful_episodes += 1

            hooks.reset()

    elapsed = time.time() - t_start
    hooks.remove()
    writer.finalize()

    summary = {
        "total_episodes": total_episodes,
        "successful_episodes": successful_episodes,
        "success_rate": successful_episodes / max(total_episodes, 1),
        "elapsed_seconds": elapsed,
        "output_dir": str(output_dir),
        "teacher_model": teacher_path,
        "benchmark": config.teacher.benchmark,
        "provenance": {"labels": "mock"},
    }

    logger.info(
        f"Label generation complete: {total_episodes} episodes, "
        f"{successful_episodes}/{total_episodes} successful "
        f"({summary['success_rate']:.1%}), {elapsed:.1f}s"
    )

    return summary


def _load_benchmark_tasks(benchmark: str) -> list[dict]:
    """Load benchmark task definitions.

    For now returns mock tasks. When LIBERO is available,
    this will load real benchmark environments.
    """
    # MOCK: Replace with real LIBERO benchmark loading
    # Install: pip install libero
    # Load: from libero.libero import benchmark as libero_bench
    logger.warning(f"Using mock benchmark tasks for '{benchmark}' — install LIBERO for real tasks")

    mock_tasks = []
    task_names = [
        "pick_up_the_red_block",
        "place_block_in_box",
        "stack_two_blocks",
        "push_block_to_target",
        "open_drawer",
        "close_drawer",
        "pick_and_place_bowl",
        "rotate_object",
        "lift_object_high",
        "sort_objects_by_color",
    ]

    for i, name in enumerate(task_names):
        mock_tasks.append(
            {
                "task_id": f"{benchmark}_task_{i}",
                "instruction": name.replace("_", " "),
                "env_name": f"{benchmark}_{name}",
            }
        )

    return mock_tasks


def _collect_episode(
    teacher: torch.nn.Module,
    processor: Any,
    task: dict,
    hooks: ExtractionHooks,
    config: TeacherConfig,
    device: str,
    episode_id: str,
) -> EpisodeData:
    """Collect a single episode of teacher demonstrations.

    MOCK: Uses synthetic data. Replace with real environment when LIBERO available.
    """
    # MOCK: Generate synthetic episode data
    # Replace with: env.reset(task=task); obs = env.step(action)
    timesteps = config.max_steps_per_episode
    height, width = 256, 256
    action_dim = 7
    proprio_dim = 7

    images = np.random.randint(0, 255, (timesteps, height, width, 3), dtype=np.uint8)
    proprioception = np.random.randn(timesteps, proprio_dim).astype(np.float32) * 0.1
    gt_actions = np.random.randn(timesteps, action_dim).astype(np.float32) * 0.1

    # MOCK: Teacher inference — simulate soft label generation
    # In production: run teacher.generate() with hooks to capture logits
    action_logits = gt_actions + np.random.randn(timesteps, action_dim).astype(np.float32) * 0.02
    action_mean = action_logits
    action_std = np.abs(np.random.randn(timesteps, action_dim).astype(np.float32) * 0.1) + 0.01

    # Vision features from hooks
    vision_features = None
    if config.save_vision_features:
        # MOCK: Simulate vision features
        # In production: hooks.vision_features from forward pass
        vision_tokens = 729  # SigLIP 384x384 / 14x14
        vision_dim = 1152  # SigLIP-SO400M dim
        vision_features = np.random.randn(timesteps, vision_tokens, vision_dim).astype(np.float16) * 0.01

    confidence = compute_action_confidence(action_std)

    return EpisodeData(
        episode_id=episode_id,
        task_id=task["task_id"],
        language_instruction=task["instruction"],
        timesteps=timesteps,
        images=images,
        proprioception=proprioception,
        teacher_action_logits=action_logits,
        teacher_action_mean=action_mean,
        teacher_action_std=action_std,
        teacher_vision_features=vision_features,
        confidence=confidence,
        ground_truth_actions=gt_actions,
        success=np.random.random() > 0.1,  # MOCK: 90% success rate
    )
