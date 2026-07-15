"""Artifact provenance for distinguishing real FORGE runs from test mocks.

The provenance block is deliberately small, JSON-compatible, and safe to store
inside torch checkpoints, pipeline summaries, and registry entries.
"""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import torch

type ComponentStatus = Literal["real", "mock"]
type ProvenanceBlock = dict[str, str]

COMPONENT_KEYS = ("vision", "language", "labels")
METADATA_KEYS = ("model_dir", "git_sha", "forge_version", "torch_version")
REQUIRED_KEYS = COMPONENT_KEYS + METADATA_KEYS
MOCK_WARNING = "[MOCK — not a real model]"
GIT_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")


class ProvenanceError(RuntimeError):
    """Base class for invalid or disallowed artifact provenance."""


class ProvenanceValidationError(ProvenanceError):
    """Raised when a present provenance block is malformed."""


class MockArtifactError(ProvenanceError):
    """Raised when a protected operation is attempted on a mock artifact."""


def validate_provenance(provenance: object) -> ProvenanceBlock:
    """Validate and return the canonical provenance block.

    Missing provenance is handled by callers as a legacy-artifact concern. Once
    a provenance block is present, however, every required field must be valid so
    malformed metadata cannot bypass mock detection.
    """
    if not isinstance(provenance, Mapping):
        raise ProvenanceValidationError("Checkpoint provenance must be a mapping.")

    missing = [key for key in REQUIRED_KEYS if key not in provenance]
    if missing:
        raise ProvenanceValidationError("Checkpoint provenance is incomplete; missing: " + ", ".join(missing) + ".")

    canonical: ProvenanceBlock = {}
    for key in COMPONENT_KEYS:
        value = provenance[key]
        if value not in {"real", "mock"}:
            raise ProvenanceValidationError(
                f"Checkpoint provenance field '{key}' must be 'real' or 'mock', got {value!r}."
            )
        canonical[key] = value

    for key in METADATA_KEYS:
        value = provenance[key]
        if not isinstance(value, str) or not value.strip():
            raise ProvenanceValidationError(f"Checkpoint provenance field '{key}' must be a non-empty string.")
        canonical[key] = value.strip()
    return canonical


def provenance_contains_mock(provenance: Mapping[str, Any] | None) -> bool:
    """Return whether any component is marked mock.

    This predicate is intentionally tolerant for registry display of legacy data;
    protected load paths separately call :func:`validate_provenance` first.
    """
    if not isinstance(provenance, Mapping):
        return False
    return any(str(provenance.get(key, "")).strip().casefold() == "mock" for key in COMPONENT_KEYS)


def require_real_provenance(
    provenance: object | None,
    *,
    action: str,
    allow_mock: bool = False,
) -> ProvenanceBlock | None:
    """Validate provenance and refuse mock artifacts for a protected action.

    ``None`` remains accepted for checkpoints created before provenance stamping.
    A present but malformed block is rejected, and a valid mock block requires an
    explicit ``allow_mock`` opt-in.
    """
    if provenance is None:
        return None

    canonical = validate_provenance(provenance)
    mock_components = [key for key in COMPONENT_KEYS if canonical[key] == "mock"]
    if mock_components and not allow_mock:
        components = ", ".join(mock_components)
        raise MockArtifactError(
            f"Refusing to {action} a mock-derived checkpoint ({components} marked mock). "
            "Train with real model weights and teacher labels, then retry. Run `forge doctor` "
            "to verify local assets. Use `--allow-mock` only for explicit test workflows."
        )
    return canonical


def checkpoint_provenance(payload: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    """Extract a provenance mapping from a checkpoint-like payload."""
    if not isinstance(payload, Mapping):
        return None
    value = payload.get("provenance")
    return value if isinstance(value, Mapping) else None


def _explicit_component_status(source: Any, component: str) -> ComponentStatus | None:
    """Read an explicit status marker from an object or its metadata."""
    if source is None:
        return None

    candidates: list[Any] = []
    if isinstance(source, Mapping):
        candidates.append(source)
    for attribute in ("provenance", "_forge_provenance", "component_provenance"):
        value = getattr(source, attribute, None)
        if isinstance(value, Mapping):
            candidates.append(value)

    reader = getattr(source, "reader", None)
    for holder in (source, reader):
        metadata = getattr(holder, "metadata", None)
        if isinstance(metadata, Mapping):
            candidates.append(metadata)
            nested = metadata.get("provenance")
            if isinstance(nested, Mapping):
                candidates.append(nested)

    for candidate in candidates:
        value = candidate.get(component)
        if isinstance(value, str) and value.casefold() in {"real", "mock"}:
            return value.casefold()  # type: ignore[return-value]
        for key in (f"{component}_provenance", f"{component}_status"):
            value = candidate.get(key)
            if isinstance(value, str) and value.casefold() in {"real", "mock"}:
                return value.casefold()  # type: ignore[return-value]

    for attribute in (f"{component}_provenance", f"_{component}_provenance"):
        value = getattr(source, attribute, None)
        if isinstance(value, str) and value.casefold() in {"real", "mock"}:
            return value.casefold()  # type: ignore[return-value]
    return None


def _has_mock_marker(value: Any) -> bool:
    if value is None:
        return True
    for attribute in ("is_mock", "_is_mock", "mock", "_forge_mock"):
        if getattr(value, attribute, False) is True:
            return True
    names = []
    for base in type(value).__mro__:
        # Split CamelCase before matching whole marker words. This recognizes
        # MockVisionEncoder/SyntheticBackbone without treating TinyLlama as a mock.
        normalized = re.sub(r"(?<!^)(?=[A-Z])", "_", base.__name__).casefold()
        names.append(normalized)
    return any(re.search(r"(?:^|_)(?:mock|synthetic)(?:_|$)", name) is not None for name in names)


def _student_component_status(student: Any, component: str) -> ComponentStatus:
    explicit = _explicit_component_status(student, component)
    if explicit is not None:
        return explicit

    wrapped = getattr(student, "module", student)
    attributes = {
        "vision": ("vision_encoder", "vision", "vision_model"),
        "language": ("language", "language_model", "language_backbone"),
    }[component]
    for attribute in attributes:
        if hasattr(wrapped, attribute):
            return "mock" if _has_mock_marker(getattr(wrapped, attribute)) else "real"
    return "mock"


def _label_status(dataset: Any) -> ComponentStatus:
    explicit = _explicit_component_status(dataset, "labels")
    if explicit is not None:
        return explicit
    return "mock"


def _resolved_model_dir(student: Any, config: Any, model_dir: str | Path | None) -> str:
    candidate = model_dir
    if candidate is None and config is not None:
        paths = getattr(config, "paths", None)
        candidate = getattr(paths, "model_dir", None)
    if candidate is None and student is not None:
        candidate = getattr(student, "_model_dir", None)
    if candidate is None:
        return "unknown"
    return str(Path(candidate).expanduser().resolve(strict=False))


@lru_cache(maxsize=1)
def current_git_sha() -> str:
    """Return the build/runtime Git SHA without invoking a shell."""
    for key in ("FORGE_GIT_SHA", "GITHUB_SHA", "GIT_COMMIT"):
        value = os.environ.get(key, "").strip().lower()
        if GIT_SHA_PATTERN.fullmatch(value):
            return value

    try:
        from forge._build_info import SOURCE_GIT_SHA
    except ImportError:  # pragma: no cover - only possible for malformed legacy installs
        SOURCE_GIT_SHA = "unknown"
    baked_revision = str(SOURCE_GIT_SHA).strip().lower()
    if GIT_SHA_PATTERN.fullmatch(baked_revision):
        return baked_revision

    repository = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    revision = result.stdout.strip().lower()
    return revision if GIT_SHA_PATTERN.fullmatch(revision) else "unknown"


def build_provenance(
    *,
    student: Any = None,
    config: Any = None,
    dataset: Any = None,
    vision: ComponentStatus | None = None,
    language: ComponentStatus | None = None,
    labels: ComponentStatus | None = None,
    model_dir: str | Path | None = None,
    git_sha: str | None = None,
    forge_version: str | None = None,
    torch_version: str | None = None,
) -> ProvenanceBlock:
    """Build a validated provenance block from runtime component evidence."""
    if forge_version is None:
        from forge import __version__

        forge_version = __version__

    provenance = {
        "vision": vision if vision is not None else _student_component_status(student, "vision"),
        "language": language if language is not None else _student_component_status(student, "language"),
        "labels": labels if labels is not None else _label_status(dataset),
        "model_dir": _resolved_model_dir(student, config, model_dir),
        "git_sha": git_sha if git_sha is not None else current_git_sha(),
        "forge_version": forge_version,
        "torch_version": torch_version if torch_version is not None else str(torch.__version__),
    }
    return validate_provenance(provenance)


__all__ = [
    "COMPONENT_KEYS",
    "METADATA_KEYS",
    "MOCK_WARNING",
    "MockArtifactError",
    "ProvenanceBlock",
    "ProvenanceError",
    "ProvenanceValidationError",
    "build_provenance",
    "checkpoint_provenance",
    "current_git_sha",
    "provenance_contains_mock",
    "require_real_provenance",
    "validate_provenance",
]
