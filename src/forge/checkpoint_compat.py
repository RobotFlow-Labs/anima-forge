"""Shared checkpoint compatibility helpers used by serve + eval + pipeline."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from forge.provenance import ProvenanceValidationError, require_real_provenance

logger = logging.getLogger(__name__)


STATE_DICT_KEYS = ("model_state_dict", "student_state_dict", "state_dict")
PREFIXES = ("module.student.", "model.student.", "module.", "model.", "student.")


@dataclass
class CheckpointLoadReport:
    """Structured report for non-fatal checkpoint loading outcomes."""

    source: str | None = None
    extracted_key: str | None = None
    raw_key_count: int = 0
    normalized_key_count: int = 0
    normalized_prefix: str | None = None
    shape_compatible_count: int = 0
    coverage_fraction: float = 0.0
    skipped_non_tensor_count: int = 0
    mismatched_shape_count: int = 0
    missing_keys: list[str] = field(default_factory=list)
    unexpected_keys: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    load_mode: str = "direct"


def verify_checkpoint_provenance(
    checkpoint: Mapping[str, Any],
    *,
    action: str,
    allow_mock: bool = False,
) -> None:
    """Refuse mock-derived checkpoints for a protected runtime operation.

    Checkpoints without a provenance block predate PRD-36 and remain loadable.
    Once present, provenance must be complete and truthful.
    """
    if "provenance" not in checkpoint:
        return
    provenance = checkpoint["provenance"]
    if provenance is None:
        raise ProvenanceValidationError("Checkpoint provenance must be a mapping.")
    require_real_provenance(
        provenance,
        action=action,
        allow_mock=allow_mock,
    )


def load_checkpoint_payload(
    checkpoint_path: str,
    map_location: str = "cpu",
    *,
    verify_provenance_for: str | None = None,
    allow_mock: bool = False,
) -> dict[str, Any] | None:
    """Safely load a torch payload and optionally enforce real-only provenance.

    ``verify_provenance_for`` is an operation name such as ``"serve"``,
    ``"eval"``, or ``"export"``. Provenance validation is opt-in, but safe
    tensor-only deserialization is mandatory for every caller.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        return None

    try:
        payload = torch.load(str(path), map_location=map_location, weights_only=True)
    except Exception as exc:
        operation = verify_provenance_for or "checkpoint use"
        raise ValueError(
            f"Refusing unsafe legacy checkpoint load for {operation}: {path}. "
            "Re-save the checkpoint with tensor-only state and provenance metadata."
        ) from exc
    if not isinstance(payload, dict):
        return None
    if verify_provenance_for is not None:
        verify_checkpoint_provenance(
            payload,
            action=verify_provenance_for,
            allow_mock=allow_mock,
        )
    return payload


def extract_checkpoint_state_dict(checkpoint: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """Return a likely model payload and its source key."""
    from forge.quantize.serialization import PACKED_STATE_KEY, unpack_state_dict

    packed = checkpoint.get(PACKED_STATE_KEY)
    if isinstance(packed, dict):
        return unpack_state_dict(packed), PACKED_STATE_KEY

    for key in STATE_DICT_KEYS:
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value, key

    candidate = checkpoint.get("model")
    if isinstance(candidate, dict):
        return candidate, "model"
    return None, None


def apply_checkpoint_structure(model: torch.nn.Module, checkpoint: Mapping[str, Any]) -> None:
    """Apply structural metadata before loading a compressed checkpoint state."""
    pruning = checkpoint.get("pruning")
    if not isinstance(pruning, Mapping):
        return
    removed = pruning.get("removed_layers")
    if not isinstance(removed, list) or not all(
        isinstance(index, int) and not isinstance(index, bool) for index in removed
    ):
        raise ValueError("Checkpoint pruning.removed_layers must be a list of integers")
    if removed:
        pre_prune_layer_count = pruning.get("pre_prune_layer_count")
        if isinstance(pre_prune_layer_count, bool) or not isinstance(pre_prune_layer_count, int):
            raise ValueError("Checkpoint pruning.pre_prune_layer_count must be an integer")
        target_layers = pruning.get("target_layers")
        if target_layers is not None:
            if isinstance(target_layers, bool) or not isinstance(target_layers, int):
                raise ValueError("Checkpoint pruning.target_layers must be an integer")
            if target_layers != pre_prune_layer_count - len(removed):
                raise ValueError("Checkpoint pruning.target_layers is inconsistent with removed_layers")
        from forge.prune import apply_pruning_structure

        apply_pruning_structure(
            model,
            removed,
            pre_prune_layer_count=pre_prune_layer_count,
        )


def strip_known_prefixes(state_dict: Mapping[str, Any]) -> tuple[dict[str, Any], str | None]:
    """Drop one known DDP/wrapper prefix when present."""
    values: dict[str, Any] = {str(k): v for k, v in state_dict.items()}
    if not values:
        return values, None

    for prefix in PREFIXES:
        if any(name.startswith(prefix) for name in values):
            stripped: dict[str, Any] = {}
            for name, value in values.items():
                new_name = name[len(prefix) :] if name.startswith(prefix) else name
                if new_name in stripped and new_name != name:
                    logger.debug("Prefix normalization collision on '%s' for key '%s'", new_name, name)
                stripped[new_name] = value
            return stripped, prefix
    return values, None


def filter_state_dict_by_shape(
    model_state: Mapping[str, Any],
    state_dict: Mapping[str, Any],
) -> tuple[dict[str, Any], int, int]:
    """Return tensors whose shape matches the target model and counts."""
    compatible: dict[str, Any] = {}
    skipped_non_tensor = 0
    mismatched = 0

    for key, value in state_dict.items():
        if not hasattr(value, "shape"):
            skipped_non_tensor += 1
            continue
        expected = model_state.get(key)
        if expected is None or not hasattr(expected, "shape"):
            mismatched += 1
            continue
        if tuple(value.shape) == tuple(expected.shape):
            compatible[key] = value
        else:
            mismatched += 1
    return compatible, skipped_non_tensor, mismatched


def load_model_weights_with_compatibility(
    model: torch.nn.Module,
    state_dict: Mapping[str, Any],
    *,
    context: str = "model",
    minimum_coverage: float = 0.0,
) -> tuple[torch.nn.modules.module._IncompatibleKeys, CheckpointLoadReport]:
    """Load model weights with known-compatible fallback.

    This keeps strict failures observable while preserving deterministic models and
    avoiding noisy hard-failures when only namespace or sparse shape drift exists.
    """
    report = CheckpointLoadReport()
    normalized, prefix = strip_known_prefixes(state_dict)
    report.normalized_prefix = prefix
    report.raw_key_count = len(state_dict)
    report.normalized_key_count = len(normalized)
    model_state = model.state_dict()
    compatible, skipped, mismatched = filter_state_dict_by_shape(model_state, normalized)
    report.shape_compatible_count = len(compatible)
    report.skipped_non_tensor_count = skipped
    report.mismatched_shape_count = mismatched
    report.coverage_fraction = len(compatible) / max(len(model_state), 1)
    if not compatible:
        raise RuntimeError(f"No compatible tensor keys remain for {context}.")
    if report.coverage_fraction < minimum_coverage:
        raise RuntimeError(
            f"Checkpoint coverage for {context} is {report.coverage_fraction:.1%}; "
            f"at least {minimum_coverage:.1%} is required. Refusing to use mostly random weights."
        )

    try:
        report.load_mode = "direct"
        compatibility = model.load_state_dict(normalized, strict=False)
        report.missing_keys = list(compatibility.missing_keys)[:16]
        report.unexpected_keys = list(compatibility.unexpected_keys)[:16]
        return compatibility, report
    except RuntimeError as exc:
        report.load_mode = "shape_filtered"
        report.warnings.append(f"{context} strict-load failed; applying shape-filtered fallback: {exc}")
        compatibility = model.load_state_dict(compatible, strict=False)
        report.missing_keys = list(compatibility.missing_keys)[:16]
        report.unexpected_keys = list(compatibility.unexpected_keys)[:16]
        report.warnings.append(
            "Shape-filtered compatibility load completed. "
            f"{report.shape_compatible_count}/{report.normalized_key_count} tensors restored."
        )
        return compatibility, report


def summarize_checkpoint_report(context: str, report: CheckpointLoadReport) -> str:
    """Build a short human-readable compatibility summary."""
    parts = [f"{context} load mode={report.load_mode}"]
    if report.extracted_key:
        parts.append(f"source={report.extracted_key}")
    if report.normalized_prefix:
        parts.append(f"stripped_prefix={report.normalized_prefix}")
    parts.append(f"raw_keys={report.raw_key_count}")
    parts.append(f"normalized_keys={report.normalized_key_count}")
    if report.shape_compatible_count:
        parts.append(f"shape_compatible={report.shape_compatible_count}")
        parts.append(f"coverage={report.coverage_fraction:.1%}")
    if report.skipped_non_tensor_count:
        parts.append(f"skipped_non_tensor={report.skipped_non_tensor_count}")
    if report.mismatched_shape_count:
        parts.append(f"shape_mismatch={report.mismatched_shape_count}")
    if report.missing_keys:
        parts.append(f"missing={len(report.missing_keys)}")
    if report.unexpected_keys:
        parts.append(f"unexpected={len(report.unexpected_keys)}")
    return "; ".join(parts)
