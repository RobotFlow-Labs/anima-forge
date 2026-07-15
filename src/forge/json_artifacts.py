"""Strict, durable, atomic JSON artifact persistence."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def write_json_artifact(path: str | Path, payload: Any) -> None:
    """Validate and atomically publish one strict JSON artifact.

    Serialization completes before a temporary file is created. The temporary
    file is flushed and synced beside the target before an atomic replacement,
    and is removed on every failure path.
    """
    target = Path(path)
    serialized = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    target.parent.mkdir(parents=True, exist_ok=True)

    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
