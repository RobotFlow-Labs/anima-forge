"""PRD-25: AutoSense — Dynamic Model Config Detection.

Reads model config.json files at load time and auto-populates dimensions,
eliminating config mismatches when switching between models.

Usage:
    from forge.autosense import apply_autosense
    config = apply_autosense(config, model_dir)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _read_model_config(model_path: Path) -> dict[str, Any] | None:
    config_path = model_path / "config.json"
    if not config_path.exists():
        logger.debug("No config.json found at %s", config_path)
        return None
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read %s: %s", config_path, exc)
        return None
    return data if isinstance(data, dict) else None


def sense_model_roles(model_path: Path) -> frozenset[str]:
    """Classify a local Hugging Face config without treating every hidden size as every role."""
    data = _read_model_config(model_path)
    if data is None:
        return frozenset()

    model_type = str(data.get("model_type", "")).lower()
    architectures = " ".join(str(value).lower() for value in data.get("architectures", []) if value)
    vision_config = data.get("vision_config")
    text_config = data.get("text_config")

    nested_vision = isinstance(vision_config, dict) and bool(vision_config.get("hidden_size"))
    top_level_vision_shape = bool(data.get("hidden_size")) and bool(data.get("image_size") or data.get("patch_size"))
    vision_markers = (
        "vision",
        "dinov",
        "vjepa",
        "theia",
        "siglip",
        "clip",
        "vit",
        "sam",
        "detr",
    )
    is_vision = (
        nested_vision
        or top_level_vision_shape
        or any(marker in model_type or marker in architectures for marker in vision_markers)
    )

    contrastive_only = model_type in {"clip", "siglip", "siglip2"} or (
        ("clip" in model_type or "siglip" in model_type) and "generation" not in architectures
    )
    nested_language = (
        isinstance(text_config, dict)
        and bool(text_config.get("hidden_size"))
        and bool(text_config.get("vocab_size"))
        and not contrastive_only
    )
    top_level_language = (
        bool(data.get("hidden_size"))
        and bool(data.get("vocab_size"))
        and (
            not is_vision
            or any(marker in architectures for marker in ("causallm", "generation", "language", "vlmodel"))
        )
    )
    is_language = nested_language or top_level_language

    roles: set[str] = set()
    if is_vision:
        roles.add("vision")
    if is_language:
        roles.add("language")
    return frozenset(roles)


def sense_vision_encoder(model_path: Path) -> dict[str, Any] | None:
    """Read config.json from vision encoder, return dimension info.

    Returns:
        dict with {d_output, n_tokens, patch_size, image_size} or None if unreadable.

    Handles:
        - SigLIP: vision_config.hidden_size, vision_config.image_size, vision_config.patch_size
        - DINOv2: hidden_size, image_size, patch_size (top-level)
        - Theia: hidden_size (top-level)
        - Generic HF vision models
    """
    data = _read_model_config(model_path)
    if data is None:
        return None

    result: dict[str, Any] = {}

    # Try vision_config first (SigLIP, CLIP-style models)
    vision_cfg = data.get("vision_config", {})
    if vision_cfg.get("hidden_size"):
        result["d_output"] = vision_cfg["hidden_size"]
        if "image_size" in vision_cfg:
            result["image_size"] = vision_cfg["image_size"]
        if "patch_size" in vision_cfg:
            result["patch_size"] = vision_cfg["patch_size"]
    elif data.get("hidden_size"):
        # DINOv2, Theia, or other top-level configs
        result["d_output"] = data["hidden_size"]
        if "image_size" in data:
            result["image_size"] = data["image_size"]
        if "patch_size" in data:
            result["patch_size"] = data["patch_size"]

    if not result:
        return None

    # Calculate n_tokens from image_size / patch_size
    if "image_size" in result and "patch_size" in result:
        grid = result["image_size"] // result["patch_size"]
        result["n_tokens"] = grid * grid

    return result


def sense_language_model(model_path: Path) -> dict[str, Any] | None:
    """Read config.json from language model, return dimension info.

    Returns:
        dict with {d_model, vocab_size, n_layers, n_heads} or None if unreadable.

    Handles:
        - Qwen2.5/Qwen3 variants (top-level transformer dimensions)
        - Qwen3.5 multimodal wrappers (dimensions nested under text_config)
        - LLaMA-style models (same keys)
        - Generic HF causal LMs
    """
    data = _read_model_config(model_path)
    if data is None:
        return None

    result: dict[str, Any] = {}
    language_cfg = data.get("text_config") or data

    if "hidden_size" in language_cfg:
        result["d_model"] = language_cfg["hidden_size"]
    if "vocab_size" in language_cfg:
        result["vocab_size"] = language_cfg["vocab_size"]
    if "num_hidden_layers" in language_cfg:
        result["n_layers"] = language_cfg["num_hidden_layers"]
    if "num_attention_heads" in language_cfg:
        result["n_heads"] = language_cfg["num_attention_heads"]

    return result if result else None


def sense_teacher(model_path: Path, adapter_name: str | None = None) -> dict[str, Any] | None:
    """Read teacher config, return action-related info.

    Returns:
        dict with {action_dim, action_horizon, param_count} or None.
    """
    data = _read_model_config(model_path)
    if data is None:
        return None

    result: dict[str, Any] = {}

    # Count parameters from model size on disk (approximate)
    safetensor_files = list(model_path.glob("*.safetensors"))
    bin_files = list(model_path.glob("*.bin"))
    weight_files = safetensor_files or bin_files
    if weight_files:
        total_bytes = sum(f.stat().st_size for f in weight_files)
        # Rough estimate: 2 bytes per param (float16/bfloat16)
        result["param_count"] = total_bytes / 2

    # Try to extract action dim from known architectures
    if "action_dim" in data:
        result["action_dim"] = data["action_dim"]
    if "action_horizon" in data:
        result["action_horizon"] = data["action_horizon"]

    return result if result else None


def autosense_config(
    model_dir: Path,
    vision_name: str | None = None,
    lm_name: str | None = None,
) -> dict[str, Any]:
    """Full auto-detection: returns override dict for StudentConfig fields.

    Args:
        model_dir: Base directory containing model subdirectories.
        vision_name: Vision encoder directory name (e.g., "google--siglip-so400m-patch14-384").
        lm_name: Language model directory name (e.g., "Qwen--Qwen2.5-0.5B").

    Returns:
        Dict of config field overrides (e.g., {"bridge_d_vision": 1152, "bridge_d_model": 896}).
    """
    overrides: dict[str, Any] = {}

    if vision_name:
        vision_path = model_dir / vision_name
        vision_info = sense_vision_encoder(vision_path)
        if vision_info:
            if "d_output" in vision_info:
                overrides["bridge_d_vision"] = vision_info["d_output"]
            if "n_tokens" in vision_info:
                overrides["n_tokens"] = vision_info["n_tokens"]

    if lm_name:
        lm_path = model_dir / lm_name
        lm_info = sense_language_model(lm_path)
        if lm_info:
            if "d_model" in lm_info:
                overrides["bridge_d_model"] = lm_info["d_model"]

    return overrides


def apply_autosense(config: Any, model_dir: Path | str) -> Any:
    """Mutate config in-place with auto-detected values. Log what changed.

    Args:
        config: ForgeConfig instance.
        model_dir: Path to directory containing model subdirectories.

    Returns:
        The same config, mutated in-place.
    """
    model_dir = Path(model_dir)
    if not model_dir.exists():
        logger.debug(f"AutoSense: model_dir {model_dir} does not exist, skipping")
        return config

    # Check if autosense is enabled
    if hasattr(config, "student") and hasattr(config.student, "autosense"):
        if not config.student.autosense:
            logger.debug("AutoSense: disabled via config")
            return config

    # Determine model directory names from config
    vision_name = None
    lm_name = None

    if hasattr(config, "paths"):
        vision_name = getattr(config.paths, "vision_encoder", None)
        lm_name = getattr(config.paths, "language_model", None)
    elif hasattr(config, "vision_encoder"):
        # StudentConfig directly
        vision_name = config.vision_encoder.replace("/", "--")
        lm_name = getattr(config, "language_model", "").replace("/", "--")

    overrides = autosense_config(model_dir, vision_name, lm_name)

    if not overrides:
        logger.debug("AutoSense: no overrides detected")
        return config

    # Apply overrides to student config
    student = config.student if hasattr(config, "student") else config
    changes = []

    for key, value in overrides.items():
        if hasattr(student, key):
            old_value = getattr(student, key)
            if old_value != value:
                setattr(student, key, value)
                changes.append(f"{key}: {old_value} → {value}")

    if changes:
        logger.info(f"AutoSense applied {len(changes)} override(s):")
        for change in changes:
            logger.info(f"  {change}")
    else:
        logger.debug("AutoSense: all values already match")

    return config
