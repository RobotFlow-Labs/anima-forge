"""Trusted OpenVLA loading across the Transformers 4 -> 5 API break."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import torch

from forge.transformers_compat import load_image_text_config


def _patch_legacy_openvla_class(model_class: Any) -> tuple[Any, str | None]:
    """Patch known pre-5.x remote class assumptions, idempotently."""
    if not getattr(model_class, "_forge_transformers5_compat", False):
        original_tie_weights = model_class.tie_weights

        def tie_weights_compat(self, *args, **kwargs):
            return original_tie_weights(self)

        model_class.tie_weights = tie_weights_compat
        # The legacy implementation exposes this as a property that accesses
        # language_model before it is created. Force eager attention during init.
        model_class._supports_sdpa = False
        model_class._forge_transformers5_compat = True

    remote_module = importlib.import_module(model_class.__module__)
    timm_module = getattr(remote_module, "timm", None)
    original_timm_version = getattr(timm_module, "__version__", None)
    if timm_module is not None and original_timm_version not in {"0.9.10", "0.9.11", "0.9.12", "0.9.16"}:
        timm_module.__version__ = "0.9.16"
    return timm_module, original_timm_version


def load_image_text_model(
    model_path: str | Path,
    *,
    dtype: torch.dtype,
    device_map: str | None,
    local_files_only: bool,
) -> torch.nn.Module:
    """Load an image-text model, applying legacy OpenVLA shims when declared."""
    from transformers import AutoModelForImageTextToText

    config = load_image_text_config(model_path, local_files_only=local_files_only)
    auto_map = dict(getattr(config, "auto_map", {}) or {})
    legacy_reference = auto_map.get("AutoModelForVision2Seq")
    if not legacy_reference:
        return AutoModelForImageTextToText.from_pretrained(
            str(model_path),
            config=config,
            dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )

    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from transformers.models.auto.auto_factory import add_generation_mixin_to_remote_model

    model_class = get_class_from_dynamic_module(
        legacy_reference,
        model_path,
        local_files_only=local_files_only,
    )
    timm_module, original_timm_version = _patch_legacy_openvla_class(model_class)
    compatible_class = add_generation_mixin_to_remote_model(model_class)
    try:
        return compatible_class.from_pretrained(
            str(model_path),
            config=config,
            dtype=dtype,
            device_map=device_map,
            local_files_only=local_files_only,
            attn_implementation="eager",
        )
    finally:
        if timm_module is not None and original_timm_version is not None:
            timm_module.__version__ = original_timm_version


__all__ = ["load_image_text_model"]
