"""Shared CLI helpers used by multiple command groups."""

from __future__ import annotations

import logging
import os
import re
from importlib import resources
from pathlib import Path

from forge.cli_commands.json_output import (
    emit_cli_error as emit_cli_error,
)
from forge.cli_commands.json_output import (
    emit_json as emit_json,
)
from forge.cli_commands.json_output import (
    json_payload as json_payload,
)
from forge.cli_commands.json_output import (
    sanitize_json as sanitize_json,
)

DEFAULT_NANO_CONFIG = "configs/forge_nano.yaml"
_CUDA_DEVICE_PATTERN = re.compile(r"cuda(?::(?P<index>\d+))?\Z")


def load_forge_config(config_path: str | Path, *, required: bool = False):
    """Load config from disk or use defaults.

    Args:
        config_path: Path to a YAML config file.
        required: If True and the file does not exist, fail fast.
    """
    from forge.config import ForgeConfig

    path = Path(config_path).expanduser()
    if path.is_file():
        return ForgeConfig.from_yaml(path)

    # Public commands advertise this repository-relative path as their default.
    # A wheel user can invoke those commands from any working directory, so the
    # same config must be resolved from installed package data instead of being
    # silently replaced with ``ForgeConfig.default()``.
    if path.as_posix() == DEFAULT_NANO_CONFIG:
        packaged = resources.files("forge").joinpath("configs", "forge_nano.yaml")
        if not packaged.is_file():
            raise FileNotFoundError(
                "The packaged FORGE nano config is missing; reinstall anima-forge or pass --config explicitly."
            )
        with resources.as_file(packaged) as packaged_path:
            return ForgeConfig.from_yaml(packaged_path)

    if required:
        raise FileNotFoundError(f"Config file not found: {path}")
    return ForgeConfig.default()


def get_json_payload(obj) -> str:
    """Return a compact JSON string for CLI output."""
    import json

    return json.dumps(obj, indent=2, default=str)


def truncate_output(text: str, *, max_bytes: int) -> tuple[str, bool]:
    """Clamp output text to max bytes and return (text, was_truncated)."""
    if max_bytes <= 0:
        return text, False

    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text, False

    suffix = f"\n... [truncated to {max_bytes} bytes for CLI safety]"
    suffix_bytes = suffix.encode("utf-8")
    if len(suffix_bytes) >= max_bytes:
        return suffix[: max(1, max_bytes - 1)] + "...", True

    keep_bytes = max_bytes - len(suffix_bytes)
    truncated_text = encoded[:keep_bytes].decode("utf-8", errors="replace")
    return f"{truncated_text}{suffix}", True


def format_json_for_cli(obj, *, max_bytes: int) -> tuple[str, bool]:
    """JSON serialize and clamp to max_bytes with truncation metadata."""
    payload = get_json_payload(obj)
    return truncate_output(payload, max_bytes=max_bytes)


def _env_enabled(value: str | None) -> bool:
    """Return True for common truthy env values."""
    return bool(value and value.strip().lower() in {"1", "true", "yes", "on", "enabled"})


def _normalize_command_env_key(command: str | None) -> str:
    """Normalize command keys into env suffix format."""
    return (command or "").strip().upper().replace("-", "_")


def resolve_runtime_device(
    device: str | None,
    *,
    command: str | None = None,
    default: str = "auto",
    strict: bool | None = None,
) -> str:
    """Resolve and normalize runtime device selection for CLI/GPU-first execution.

    Resolution order:
      1) CLI argument (`requested_device`)
      2) command-scoped env `FORGE_<COMMAND>_DEVICE`
      3) global env `FORGE_DEVICE`
      4) `default`

    Accepted values:
      - `auto`: resolved to CUDA if available else CPU
      - `cuda`, `cuda:N`: CUDA-capable device (if unavailable, falls back)
      - `cpu`
      - `mps`

    If CUDA is requested but not available:
      - when `strict` is true and `FORGE_ALLOW_CPU_FALLBACK` is false, raise ValueError.
      - otherwise, fall back to CPU with a warning and continue.
    """
    import torch

    requested = (device or "").strip().lower()
    if not requested:
        if command:
            requested = os.environ.get(f"FORGE_{_normalize_command_env_key(command)}_DEVICE", "").strip().lower()
        if not requested:
            requested = os.environ.get("FORGE_DEVICE", "").strip().lower() or default

    if requested == "auto" or not requested:
        requested = "cuda" if torch.cuda.is_available() else "cpu"

    if requested.startswith("cuda"):
        match = _CUDA_DEVICE_PATTERN.fullmatch(requested)
        if match is None:
            raise ValueError(f"Unsupported CUDA device: {requested}")
        if not torch.cuda.is_available():
            fallback = _env_enabled(os.environ.get("FORGE_ALLOW_CPU_FALLBACK"))
            if strict is None:
                strict = os.environ.get("FORGE_REQUIRE_GPU", "").strip().lower() in {
                    "1",
                    "true",
                    "yes",
                    "on",
                    "enabled",
                }
            if strict and not fallback:
                raise RuntimeError(
                    "FORGE requested CUDA but no CUDA device is available. "
                    "Set --device cpu or run with `FORGE_ALLOW_CPU_FALLBACK=1`."
                )

            if fallback or not strict:
                logging.getLogger(__name__).warning(
                    "CUDA requested but unavailable; falling back to CPU for this command."
                )
            return "cpu"

        index_text = match.group("index")
        if index_text is None:
            return "cuda"
        index = int(index_text)
        device_count = int(torch.cuda.device_count())
        if index >= device_count:
            raise ValueError(
                f"CUDA device index {index} is unavailable; this process exposes {device_count} CUDA device(s)."
            )
        return f"cuda:{index}"

    if requested == "cpu":
        return "cpu"

    if requested == "mps":
        return "mps"

    raise ValueError(f"Unsupported device: {requested}")
