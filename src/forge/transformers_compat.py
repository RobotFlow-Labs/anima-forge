"""Transformers 5.x compatibility for trusted VLA checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from forge.hf_compat import configure_transformers_module_cache


def load_image_text_config(
    model_path: str | Path,
    *,
    local_files_only: bool,
) -> Any:
    """Load trusted config and migrate the removed Vision2Seq auto-map key.

    OpenVLA checkpoints published before Transformers 5 map their custom model
    to ``AutoModelForVision2Seq``. Transformers 5.3 removed that auto class in
    favor of ``AutoModelForImageTextToText``; copying the checkpoint-declared
    target preserves trusted remote-code loading without hardcoding its class.
    """
    configure_transformers_module_cache(model_path)

    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    auto_map = dict(getattr(config, "auto_map", {}) or {})
    legacy_target = auto_map.get("AutoModelForVision2Seq")
    if legacy_target and "AutoModelForImageTextToText" not in auto_map:
        auto_map["AutoModelForImageTextToText"] = legacy_target
        config.auto_map = auto_map
    return config


__all__ = ["load_image_text_config"]
