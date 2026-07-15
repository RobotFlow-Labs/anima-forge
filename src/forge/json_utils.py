"""Strict JSON conversion shared by CLI output and persisted reports."""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Any

SANITIZED_NOTE = "Non-finite numeric values were replaced with null."


def sanitize_json(value: Any) -> tuple[Any, bool]:
    """Return a JSON-safe value and whether a non-finite number was replaced."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = dataclasses.asdict(value)
    if isinstance(value, Enum):
        value = value.value
    if isinstance(value, Path):
        return str(value), False
    if value is None or isinstance(value, (str, bool, int)):
        return value, False
    if isinstance(value, float):
        if not math.isfinite(value):
            return None, True
        return value, False
    if isinstance(value, Mapping):
        mapping_result: dict[str, Any] = {}
        sanitized = False
        for key, item in value.items():
            clean, changed = sanitize_json(item)
            mapping_result[str(key)] = clean
            sanitized = sanitized or changed
        return mapping_result, sanitized
    if isinstance(value, (list, tuple, set)):
        sequence_result: list[Any] = []
        sanitized = False
        for item in value:
            clean, changed = sanitize_json(item)
            sequence_result.append(clean)
            sanitized = sanitized or changed
        return sequence_result, sanitized

    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            scalar = item_method()
        except (RuntimeError, TypeError, ValueError):
            scalar = value
        if scalar is not value:
            return sanitize_json(scalar)

    tolist_method = getattr(value, "tolist", None)
    if callable(tolist_method):
        try:
            listed = tolist_method()
        except (RuntimeError, TypeError, ValueError):
            listed = value
        if listed is not value:
            return sanitize_json(listed)

    return str(value), False


def json_ready(value: Any) -> Any:
    """Return a sanitized object and attach the required note when changed."""
    clean, sanitized = sanitize_json(value)
    if sanitized:
        if isinstance(clean, dict):
            clean.setdefault("note", SANITIZED_NOTE)
        else:
            clean = {"data": clean, "note": SANITIZED_NOTE}
    return clean


def json_payload(value: Any) -> str:
    """Serialize one response with RFC-compliant non-finite handling."""
    return json.dumps(json_ready(value), indent=2, allow_nan=False)
