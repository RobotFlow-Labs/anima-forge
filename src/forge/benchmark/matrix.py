"""Artifact-driven battle validation matrix orchestration."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from forge.benchmark.artifacts import write_json_artifact
from forge.benchmark.execution import benchmark_execution
from forge.cli_commands.shared import load_forge_config
from forge.config import STUDENT_VARIANT_PRESETS
from forge.export.onnx_artifacts import resolve_onnx_artifact_family
from forge.export.onnx_export import benchmark_onnx_runtime
from forge.export.tensorrt_export import benchmark_tensorrt_runtime

MANIFEST_SCHEMA = "forge.validation-manifest.v1"
RESULT_SCHEMA = "forge.validation-matrix.v1"
STANDARD_TRAINING_STEPS = 2_000
FLAGSHIP_TRAINING_STEPS = 5_000
FLAGSHIP_ACCEPTANCE = {"kind": "flagship"}
QUANTIZED_CANDIDATES = frozenset(
    {
        "qvla_int4",
        "qvla_int8",
        "turboquant_mse_int4",
        "turboquant_mse_int8",
    }
)
QUANTIZED_FILENAMES = {
    "qvla_int4": "qvla_4bit.pt",
    "qvla_int8": "qvla_8bit.pt",
    "turboquant_mse_int4": "turboquant_mse_4bit.pt",
    "turboquant_mse_int8": "turboquant_mse_8bit.pt",
}
CHECKPOINT_CONFIG_FIELDS = (
    "variant",
    "vision_encoder",
    "language_model",
    "backbone_dtype",
    "bridge_d_vision",
    "bridge_d_model",
    "bridge_n_queries",
    "bridge_n_heads",
    "bridge_n_layers",
    "action_dim",
    "action_head_layers",
    "action_diffusion_steps",
    "action_horizon",
    "chunk_overlap",
    "action_head_type",
    "flow_inference_steps",
    "lora_rank",
    "lora_alpha",
    "lora_target_modules",
    "autosense",
    "allow_mock",
)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value}")


def _portable_value(value: Any) -> Any:
    """Remove machine-specific absolute paths from persisted benchmark JSON."""
    if isinstance(value, dict):
        return {key: _portable_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_portable_value(item) for item in value]
    if isinstance(value, str) and Path(value).is_absolute():
        return Path(value).name
    return value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_evidence_path(value: object, *, base_dir: Path) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _require_artifact_binding(
    artifact: Path,
    *,
    artifact_sha256: str,
    context: str,
    evidence_base: Path,
    declared_path: object = None,
    declared_sha256: object = None,
    legacy_paths: tuple[Path, ...] = (),
    require_sha256: bool = False,
) -> None:
    """Bind one manifest artifact to summary evidence without trusting basenames."""
    has_binding = False
    if declared_path not in (None, ""):
        has_binding = True
        evidence_path = _resolve_evidence_path(declared_path, base_dir=evidence_base)
        if evidence_path != artifact:
            raise ValueError(f"{context} path does not match manifest artifact: {evidence_path} != {artifact}")
    if declared_sha256 not in (None, ""):
        has_binding = True
        digest = str(declared_sha256).lower()
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"{context} contains an invalid SHA-256 digest")
        if digest != artifact_sha256:
            raise ValueError(f"{context} SHA-256 does not match manifest artifact")
    elif require_sha256:
        raise ValueError(f"{context} requires SHA-256 evidence")
    if not has_binding and legacy_paths:
        has_binding = any(path.resolve() == artifact for path in legacy_paths)
    if not has_binding:
        raise ValueError(f"{context} does not bind the manifest artifact by canonical path or SHA-256")


def _require_summary_origin(summary: dict[str, Any], summary_path: Path, *, context: str) -> None:
    declared = summary.get("pipeline_summary_path")
    if declared in (None, ""):
        return
    resolved = _resolve_evidence_path(declared, base_dir=summary_path.parent)
    if resolved != summary_path:
        raise ValueError(f"{context} summary path does not match the loaded summary")


def _resolve_path(value: object, *, base_dir: Path, required: bool = True) -> Path | None:
    if value in (None, ""):
        if required:
            raise ValueError("Validation manifest is missing a required artifact path")
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    path = path.resolve()
    if required and not path.is_file():
        raise ValueError(f"Validation artifact not found: {path}")
    return path


def _resolve_directory(value: object, *, base_dir: Path) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    path = path.resolve()
    if not path.is_dir():
        raise ValueError(f"Validation data directory not found: {path}")
    return path


def load_validation_manifest(path: str | Path) -> tuple[Path, list[dict[str, Any]]]:
    """Load and validate the v3 artifact manifest."""
    manifest_path = Path(path).expanduser().resolve()
    manifest = _read_json(manifest_path)
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"Manifest schema must be {MANIFEST_SCHEMA!r}")
    variants = manifest.get("variants")
    if not isinstance(variants, list) or not variants:
        raise ValueError("Validation manifest must contain a non-empty variants list")
    seen_variants: set[str] = set()
    for entry in variants:
        if not isinstance(entry, dict):
            raise ValueError("Every validation variant must be an object with a variant name")
        variant = entry.get("variant")
        if not isinstance(variant, str) or variant not in STUDENT_VARIANT_PRESETS:
            raise ValueError(f"Validation variant must be one of {sorted(STUDENT_VARIANT_PRESETS)}, got {variant!r}")
        if variant in seen_variants:
            raise ValueError(f"Validation manifest contains duplicate variant {variant!r}")
        seen_variants.add(variant)

        config_path = _resolve_path(entry.get("config"), base_dir=manifest_path.parent)
        assert config_path is not None
        config = load_forge_config(config_path, required=True)
        config_variant = config.student.variant
        if config_variant != variant:
            raise ValueError(
                f"Validation variant {variant!r} does not match config student.variant {config_variant!r}: "
                f"{config_path}"
            )
        expected_steps = entry.get("expected_training_steps")
        if isinstance(expected_steps, bool) or not isinstance(expected_steps, int):
            raise ValueError(f"Validation variant {variant!r} must explicitly declare integer expected_training_steps")
        acceptance = entry.get("acceptance")
        if expected_steps == STANDARD_TRAINING_STEPS:
            if acceptance is not None:
                raise ValueError("Standard 2,000-step validation must not declare flagship acceptance")
        elif expected_steps == FLAGSHIP_TRAINING_STEPS:
            if acceptance != FLAGSHIP_ACCEPTANCE:
                raise ValueError('5,000-step validation requires acceptance={"kind": "flagship"}')
            if (
                variant != "nano"
                or config.student.action_head_type != "flow"
                or config.student.lora_rank != 64
                or config.distill.max_steps != FLAGSHIP_TRAINING_STEPS
            ):
                raise ValueError("Flagship acceptance requires the 5,000-step nano flow-head LoRA-64 config")
        else:
            raise ValueError(
                f"expected_training_steps must be {STANDARD_TRAINING_STEPS}, or {FLAGSHIP_TRAINING_STEPS} "
                "with typed flagship acceptance"
            )
        if entry.get("evidence_profile") != "sha256-v1":
            raise ValueError("Every release validation variant requires evidence_profile='sha256-v1'")
        config_sha256 = entry.get("config_sha256")
        if (
            not isinstance(config_sha256, str)
            or len(config_sha256) != 64
            or any(character not in "0123456789abcdef" for character in config_sha256.lower())
        ):
            raise ValueError("Every release validation variant requires a valid config_sha256")
    return manifest_path, variants


def _selected_training_metrics(summary: dict[str, Any], *, expected_steps: int) -> dict[str, Any]:
    if summary.get("status") != "completed":
        raise ValueError("Training summary status must be completed")
    distill = summary.get("distill")
    if not isinstance(distill, dict):
        raise ValueError("Training summary has no distill result")
    device = distill.get("device", summary.get("device"))
    if not isinstance(device, str) or not device.startswith("cuda"):
        raise ValueError("Training summary must prove CUDA execution")
    total_steps = distill.get("total_steps")
    if isinstance(total_steps, bool) or not isinstance(total_steps, int) or total_steps != expected_steps:
        raise ValueError(f"Training summary must report exactly {expected_steps} total_steps")
    for key in ("elapsed_seconds", "steps_per_second"):
        value = distill.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Training summary must report positive finite {key}")
    loss_values: dict[str, float] = {}
    for key in ("initial_loss", "final_loss", "best_loss", "loss_reduction_percent"):
        value = distill.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Training summary must report finite {key}")
        loss_values[key] = float(value)
    if loss_values["initial_loss"] <= loss_values["final_loss"] or loss_values["loss_reduction_percent"] <= 0:
        raise ValueError("Training summary must prove a positive loss improvement")
    _require_real_components(distill.get("provenance"), context="Training")
    memory = distill.get("cuda_memory")
    if not isinstance(memory, dict) or memory.get("target_60_80_percent_met") is not True:
        raise ValueError("Training summary did not pass the required 60–80% CUDA memory gate")
    keys = (
        "total_steps",
        "elapsed_seconds",
        "steps_per_second",
        "initial_loss",
        "final_loss",
        "loss_reduction_percent",
        "best_loss",
        "cuda_memory",
        "device",
        "provenance",
    )
    return {key: distill.get(key) for key in keys}


def _selected_compression_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    if summary.get("status") != "completed":
        raise ValueError("Compression summary status must be completed")
    pruning = summary.get("pruning")
    compression = summary.get("compression")
    quantization = summary.get("quantization")
    if not isinstance(pruning, dict):
        raise ValueError("Compression summary does not prove successful pruning")
    pruning_status = pruning.get("status")
    if pruning_status not in (None, "success"):
        raise ValueError("Compression summary does not prove successful pruning")
    removed_layers = pruning.get("removed_layers")
    n_removed = pruning.get("n_removed")
    path = pruning.get("path")
    if (
        isinstance(n_removed, bool)
        or not isinstance(n_removed, int)
        or n_removed <= 0
        or not isinstance(removed_layers, list)
        or len(removed_layers) != n_removed
        or not isinstance(path, str)
        or not path.strip()
    ):
        raise ValueError("Compression summary does not contain complete successful pruning evidence")
    if not isinstance(compression, dict) or compression.get("status") != "success":
        raise ValueError("Compression summary does not prove successful compression")
    if (
        not isinstance(quantization, dict)
        or quantization.get("status") != "success"
        or quantization.get("serialization_schema") != "forge.packed-state.v1"
    ):
        raise ValueError("Compression summary does not prove successful packed quantization")
    _require_real_components(summary.get("provenance"), context="Compression")
    return cast(
        dict[str, Any],
        _portable_value(
            {
                "status": summary.get("status"),
                "source_checkpoint": summary.get("source_checkpoint"),
                "pruning": summary.get("pruning"),
                "quantization": summary.get("quantization"),
                "provenance": summary.get("provenance"),
                "execution": summary.get("execution"),
            },
        ),
    )


def _selected_export_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    if summary.get("status") != "completed":
        raise ValueError("Export summary status must be completed")
    _require_real_components(summary.get("provenance"), context="Export")
    runtime_inputs = summary.get("export_runtime_inputs")
    if (
        summary.get("status") != "completed"
        or not isinstance(runtime_inputs, dict)
        or runtime_inputs.get("status") != "success"
        or runtime_inputs.get("labels_provenance") != "real"
    ):
        raise ValueError("Export summary does not prove successful real runtime inputs")
    onnx = summary.get("export_onnx")
    if not isinstance(onnx, dict) or onnx.get("status") != "success":
        raise ValueError("Export summary does not prove successful ONNX export")
    tensorrt = summary.get("export_tensorrt")
    if (
        not isinstance(tensorrt, dict)
        or tensorrt.get("status") != "success"
        or tensorrt.get("precision") not in {"fp16", "int8"}
    ):
        raise ValueError("Export summary does not prove successful TensorRT export with supported precision")
    return cast(
        dict[str, Any],
        _portable_value(
            {
                "status": summary.get("status"),
                "source_checkpoint": summary.get("source_checkpoint"),
                "execution": summary.get("execution"),
                "runtime_inputs": runtime_inputs,
                "onnx": onnx,
                "tensorrt": tensorrt,
            },
        ),
    )


def _require_real_components(value: object, *, context: str) -> dict[str, Any]:
    required = ("vision", "language", "labels")
    if not isinstance(value, dict) or any(value.get(component) != "real" for component in required):
        raise ValueError(f"{context} summary must prove real vision, language, and labels provenance")
    return value


def _load_runtime_inputs(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            images = torch.from_numpy(np.asarray(archive["images"]).copy())
            language_ids = torch.from_numpy(np.asarray(archive["language_ids"]).copy())
    except (KeyError, OSError, ValueError) as exc:
        raise ValueError(f"Invalid runtime input archive {path}: {exc}") from exc
    if images.ndim != 4 or language_ids.ndim != 2 or images.shape[0] != language_ids.shape[0]:
        raise ValueError("Runtime input archive has incompatible image/language shapes")
    if images.shape[0] < 1 or not torch.isfinite(images).all() or not torch.isfinite(language_ids).all():
        raise ValueError("Runtime input archive must contain finite samples")
    return images.to(dtype=torch.float32), language_ids.to(dtype=torch.int64)


def _require_declared_digest(value: object, actual: str, *, context: str) -> None:
    digest = str(value).lower() if value not in (None, "") else ""
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{context} requires a valid SHA-256 digest")
    if digest != actual:
        raise ValueError(f"{context} SHA-256 does not match the resolved artifact")


def _packed_artifact_metadata(path: Path) -> dict[str, Any]:
    """Read packed-state lineage through PyTorch's restricted loader."""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"Unable to read packed artifact lineage from {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Packed artifact lineage must be an object: {path}")
    return {
        "source_checkpoint_sha256": payload.get("source_checkpoint_sha256"),
        "config_sha256": payload.get("config_sha256"),
        "pruning": payload.get("pruning"),
        "quantization": payload.get("quantization"),
    }


def _require_checkpoint_config_contract(
    checkpoint: Path,
    *,
    config: Any,
    expected_steps: int,
    training_provenance: object,
) -> None:
    """Bind a pre-config-hash D1 artifact to its exact architecture and terminal step."""
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"Unable to read training checkpoint contract from {checkpoint}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Training checkpoint contract must be an object")
    saved = payload.get("student_config")
    if not isinstance(saved, dict):
        saved = payload.get("hp")
    if not isinstance(saved, dict):
        raise ValueError("Training checkpoint must embed student_config or hp architecture evidence")
    configured = asdict(config.student)
    missing = [field for field in CHECKPOINT_CONFIG_FIELDS if field not in saved]
    if missing:
        raise ValueError("Training checkpoint architecture evidence is missing: " + ", ".join(missing))
    mismatched = [field for field in CHECKPOINT_CONFIG_FIELDS if saved[field] != configured[field]]
    if mismatched:
        raise ValueError("Training checkpoint architecture differs from manifest config: " + ", ".join(mismatched))
    checkpoint_steps = payload.get("step", payload.get("global_step"))
    if checkpoint_steps != expected_steps:
        raise ValueError(f"Training checkpoint must prove exactly {expected_steps} completed steps")
    checkpoint_provenance = payload.get("provenance")
    if not isinstance(checkpoint_provenance, dict) or not isinstance(training_provenance, dict):
        raise ValueError("Training checkpoint and summary must contain provenance evidence")
    for component in ("vision", "language", "labels"):
        if checkpoint_provenance.get(component) != "real":
            raise ValueError(f"Training checkpoint provenance {component} must be real")
        if checkpoint_provenance.get(component) != training_provenance.get(component):
            raise ValueError(f"Training checkpoint provenance {component} differs from the training summary")
    checkpoint_git = checkpoint_provenance.get("git_sha")
    summary_git = training_provenance.get("git_sha")
    if (checkpoint_git not in (None, "") or summary_git not in (None, "")) and checkpoint_git != summary_git:
        raise ValueError("Training checkpoint git provenance differs from the training summary")


def _validate_quantized_contract(
    entry: dict[str, Any],
    *,
    base_dir: Path,
    variant: str,
    lineage_root: Path,
    strict_hashes: bool,
) -> dict[str, Path]:
    raw = entry.get("quantized")
    if not isinstance(raw, dict):
        raise ValueError(f"quantized must be an object for variant {variant}")
    supplied = set(raw)
    if supplied != QUANTIZED_CANDIDATES:
        missing = sorted(QUANTIZED_CANDIDATES - supplied)
        unexpected = sorted(supplied - QUANTIZED_CANDIDATES)
        raise ValueError(
            f"quantized must contain exactly {sorted(QUANTIZED_CANDIDATES)} for variant {variant}; "
            f"missing={missing}, unexpected={unexpected}"
        )

    declared_hashes = entry.get("quantized_sha256")
    if declared_hashes is not None and (
        not isinstance(declared_hashes, dict) or set(declared_hashes) != QUANTIZED_CANDIDATES
    ):
        raise ValueError("quantized_sha256 must contain exactly the four canonical quantized candidates")
    if strict_hashes and not isinstance(declared_hashes, dict):
        raise ValueError("evidence_profile='sha256-v1' requires the exact quantized_sha256 map")

    resolved: dict[str, Path] = {}
    for name in sorted(QUANTIZED_CANDIDATES):
        artifact = _resolve_path(raw[name], base_dir=base_dir)
        assert artifact is not None
        if artifact.name != QUANTIZED_FILENAMES[name]:
            raise ValueError(f"Quantized candidate {name} must use canonical filename {QUANTIZED_FILENAMES[name]!r}")
        if not isinstance(declared_hashes, dict):
            try:
                relative = artifact.relative_to(lineage_root)
            except ValueError as exc:
                raise ValueError(f"Legacy quantized candidate {name} escapes the training lineage root") from exc
            if len(relative.parts) < 2 or not relative.parts[0].startswith(f"{variant}-"):
                raise ValueError(f"Legacy quantized candidate {name} is outside the canonical {variant} D2 lineage")
        resolved[name] = artifact
    return resolved


def _prepare_variant(entry: dict[str, Any], *, base_dir: Path) -> dict[str, Any]:
    """Resolve, content-address, and cross-bind every input before execution."""
    variant = str(entry["variant"])
    strict_hashes = entry.get("evidence_profile") == "sha256-v1"
    config = _resolve_path(entry.get("config"), base_dir=base_dir)
    checkpoint = _resolve_path(entry.get("checkpoint"), base_dir=base_dir)
    training_summary_path = _resolve_path(entry.get("training_summary"), base_dir=base_dir)
    compression_summary_path = _resolve_path(entry.get("compression_summary"), base_dir=base_dir)
    export_summary_path = _resolve_path(entry.get("export_summary"), base_dir=base_dir)
    runtime_inputs = _resolve_path(entry.get("runtime_inputs"), base_dir=base_dir)
    onnx = _resolve_path(entry.get("onnx"), base_dir=base_dir)
    tensorrt = _resolve_path(entry.get("tensorrt"), base_dir=base_dir)
    assert config is not None
    assert checkpoint is not None
    assert training_summary_path is not None
    assert compression_summary_path is not None
    assert export_summary_path is not None
    assert runtime_inputs is not None
    assert onnx is not None
    assert tensorrt is not None
    forge_config = load_forge_config(config, required=True)

    training_summary = _read_json(training_summary_path)
    compression_summary = _read_json(compression_summary_path)
    export_summary = _read_json(export_summary_path)
    _require_summary_origin(training_summary, training_summary_path, context="Training")
    _require_summary_origin(compression_summary, compression_summary_path, context="Compression")
    _require_summary_origin(export_summary, export_summary_path, context="Export")
    expected_steps = int(entry["expected_training_steps"])
    training_metrics = _selected_training_metrics(training_summary, expected_steps=expected_steps)
    compression_metrics = _selected_compression_metrics(compression_summary)
    export_metrics = _selected_export_metrics(export_summary)

    training_sha = _sha256(training_summary_path)
    checkpoint_sha = _sha256(checkpoint)
    config_sha = _sha256(config)
    _require_declared_digest(entry.get("config_sha256"), config_sha, context="Manifest config evidence")
    _require_declared_digest(
        entry.get("training_summary_sha256"), training_sha, context="Training summary manifest evidence"
    )
    _require_declared_digest(
        entry.get("training_checkpoint_sha256"), checkpoint_sha, context="Training checkpoint manifest evidence"
    )
    training_config_sha = training_summary.get("config_sha256")
    training_config_binding = entry.get("training_config_binding")
    if training_config_sha in (None, ""):
        if training_config_binding != "checkpoint-contract-v1":
            raise ValueError(
                "Training summaries without config_sha256 require training_config_binding='checkpoint-contract-v1'"
            )
    else:
        if training_config_binding is not None:
            raise ValueError("training_config_binding is only valid when the training summary lacks config_sha256")
        _require_declared_digest(training_config_sha, config_sha, context="Training summary config evidence")
    _require_declared_digest(
        compression_summary.get("config_sha256"), config_sha, context="Compression summary config evidence"
    )
    _require_declared_digest(export_summary.get("config_sha256"), config_sha, context="Export summary config evidence")
    _require_checkpoint_config_contract(
        checkpoint,
        config=forge_config,
        expected_steps=expected_steps,
        training_provenance=training_summary["distill"].get("provenance"),
    )
    distill = training_summary["distill"]
    checkpoint_dir = _resolve_evidence_path(distill.get("checkpoint_dir"), base_dir=training_summary_path.parent)
    legacy_checkpoint_paths = ((checkpoint_dir / "final.pt").resolve(),) if checkpoint_dir is not None else ()
    _require_artifact_binding(
        checkpoint,
        artifact_sha256=checkpoint_sha,
        context="Training checkpoint evidence",
        evidence_base=training_summary_path.parent,
        declared_path=distill.get("checkpoint_path"),
        declared_sha256=distill.get("checkpoint_sha256"),
        legacy_paths=legacy_checkpoint_paths,
        require_sha256=entry.get("acceptance") == FLAGSHIP_ACCEPTANCE,
    )

    lineage_root = training_summary_path.parent.parent.resolve()
    quantized = _validate_quantized_contract(
        entry,
        base_dir=base_dir,
        variant=variant,
        lineage_root=lineage_root,
        strict_hashes=strict_hashes,
    )
    artifact_paths = {
        "config": config,
        "checkpoint": checkpoint,
        "training_summary": training_summary_path,
        "compression_summary": compression_summary_path,
        "export_summary": export_summary_path,
        "runtime_inputs": runtime_inputs,
        "onnx": onnx,
        "tensorrt": tensorrt,
        **{f"quantized:{name}": path for name, path in quantized.items()},
    }
    artifact_sha256 = {name: _sha256(path) for name, path in artifact_paths.items()}
    declared_quantized_hashes = entry.get("quantized_sha256")
    if isinstance(declared_quantized_hashes, dict):
        for name in sorted(QUANTIZED_CANDIDATES):
            _require_declared_digest(
                declared_quantized_hashes[name],
                artifact_sha256[f"quantized:{name}"],
                context=f"Quantized {name}",
            )

    _require_artifact_binding(
        checkpoint,
        artifact_sha256=checkpoint_sha,
        context="Compression source checkpoint evidence",
        evidence_base=compression_summary_path.parent,
        declared_path=compression_summary.get("source_checkpoint"),
        declared_sha256=compression_summary.get("source_checkpoint_sha256"),
        require_sha256=strict_hashes,
    )
    compression = compression_summary["compression"]
    qvla_int4 = quantized["qvla_int4"]
    _require_artifact_binding(
        qvla_int4,
        artifact_sha256=artifact_sha256["quantized:qvla_int4"],
        context="Compression output evidence",
        evidence_base=compression_summary_path.parent,
        declared_path=compression.get("path"),
        declared_sha256=compression.get("sha256"),
        require_sha256=strict_hashes,
    )
    _require_artifact_binding(
        qvla_int4,
        artifact_sha256=artifact_sha256["quantized:qvla_int4"],
        context="Export source checkpoint evidence",
        evidence_base=export_summary_path.parent,
        declared_path=export_summary.get("source_checkpoint"),
        declared_sha256=export_summary.get("source_checkpoint_sha256"),
        require_sha256=strict_hashes,
    )

    runtime_evidence = export_summary["export_runtime_inputs"]
    runtime_legacy = ((export_summary_path.parent / "tensorrt_calibration.npz").resolve(),)
    _require_artifact_binding(
        runtime_inputs,
        artifact_sha256=artifact_sha256["runtime_inputs"],
        context="Export runtime input evidence",
        evidence_base=export_summary_path.parent,
        declared_path=runtime_evidence.get("path"),
        declared_sha256=runtime_evidence.get("sha256"),
        legacy_paths=runtime_legacy,
        require_sha256=strict_hashes,
    )
    onnx_evidence = export_summary["export_onnx"]
    onnx_family = onnx_evidence.get("artifacts_sha256")
    onnx_digest = onnx_family.get(onnx.name) if isinstance(onnx_family, dict) else onnx_evidence.get("sha256")
    _require_artifact_binding(
        onnx,
        artifact_sha256=artifact_sha256["onnx"],
        context="ONNX export evidence",
        evidence_base=export_summary_path.parent,
        declared_path=onnx_evidence.get("path"),
        declared_sha256=onnx_digest,
        require_sha256=strict_hashes,
    )
    if not isinstance(onnx_family, dict):
        raise ValueError("ONNX export evidence requires the exact artifacts_sha256 family")
    actual_onnx_family = resolve_onnx_artifact_family(onnx)
    if set(onnx_family) != set(actual_onnx_family):
        missing = sorted(set(actual_onnx_family) - set(onnx_family))
        unexpected = sorted(set(onnx_family) - set(actual_onnx_family))
        raise ValueError(
            f"ONNX artifact-family evidence does not match graph references; missing={missing}, unexpected={unexpected}"
        )
    onnx_sidecars: dict[str, str] = {}
    for filename, sidecar in actual_onnx_family.items():
        digest = onnx_family[filename]
        sidecar_sha = _sha256(sidecar)
        _require_declared_digest(digest, sidecar_sha, context=f"ONNX artifact-family member {filename}")
        onnx_sidecars[filename] = sidecar_sha
    tensorrt_evidence = export_summary["export_tensorrt"]
    _require_artifact_binding(
        tensorrt,
        artifact_sha256=artifact_sha256["tensorrt"],
        context="TensorRT export evidence",
        evidence_base=export_summary_path.parent,
        declared_path=tensorrt_evidence.get("path"),
        declared_sha256=tensorrt_evidence.get("sha256"),
        require_sha256=strict_hashes,
    )

    pruning = compression_summary["pruning"]
    pruning_path = _resolve_evidence_path(pruning.get("path"), base_dir=compression_summary_path.parent)
    if pruning_path is None or not pruning_path.is_file():
        raise ValueError("Compression pruning evidence does not resolve to a real artifact")
    pruning_sha = _sha256(pruning_path)
    if strict_hashes:
        _require_declared_digest(pruning.get("sha256"), pruning_sha, context="Pruned checkpoint evidence")
        expected_removed = pruning.get("removed_layers")
        reference_pruning: dict[str, Any] | None = None
        for name, artifact in quantized.items():
            metadata = _packed_artifact_metadata(artifact)
            _require_declared_digest(
                metadata.get("config_sha256"),
                config_sha,
                context=f"Quantized {name} config lineage",
            )
            if not isinstance(metadata.get("pruning"), dict):
                raise ValueError(f"Quantized {name} has invalid internal pruning metadata")
            artifact_pruning = metadata["pruning"]
            if artifact_pruning.get("removed_layers") != expected_removed:
                raise ValueError(f"Quantized {name} pruning lineage does not match compression evidence")
            if reference_pruning is None:
                reference_pruning = artifact_pruning
            elif artifact_pruning != reference_pruning:
                raise ValueError(f"Quantized {name} internal pruning metadata differs from qvla_int4")
            quantization = metadata.get("quantization")
            expected_method = "qvla" if name.startswith("qvla_") else "turboquant-mse"
            expected_bits = 4 if name.endswith("int4") else 8
            if (
                not isinstance(quantization, dict)
                or quantization.get("schema") != "forge.packed-state.v1"
                or quantization.get("method") != expected_method
                or quantization.get("bits") != expected_bits
            ):
                raise ValueError(f"Quantized {name} internal method/width metadata does not match its manifest key")
            _require_declared_digest(
                metadata.get("source_checkpoint_sha256"),
                pruning_sha,
                context=f"Quantized {name} pruned-source lineage",
            )

    data_dir = _resolve_directory(entry.get("data_dir"), base_dir=base_dir)
    instruction_value = entry.get("instruction")
    instruction = str(instruction_value).strip() if instruction_value not in (None, "") else None
    runtime_images, runtime_language_ids = _load_runtime_inputs(runtime_inputs)
    evidence = {name: {"artifact": path.name, "sha256": artifact_sha256[name]} for name, path in artifact_paths.items()}
    evidence["pruned_checkpoint"] = {"artifact": pruning_path.name, "sha256": pruning_sha}
    if onnx_sidecars:
        evidence["onnx_artifact_family"] = onnx_sidecars
    return {
        "variant": variant,
        "config": config,
        "checkpoint": checkpoint,
        "training_metrics": training_metrics,
        "compression_metrics": compression_metrics,
        "export_metrics": export_metrics,
        "runtime_inputs": runtime_inputs,
        "runtime_images": runtime_images,
        "runtime_language_ids": runtime_language_ids,
        "onnx": onnx,
        "tensorrt": tensorrt,
        "quantized": quantized,
        "artifact_sha256": artifact_sha256,
        "artifact_evidence": evidence,
        "data_dir": data_dir,
        "instruction": instruction,
        "expected_training_steps": expected_steps,
        "evidence_profile": entry.get("evidence_profile", "legacy-path-v1"),
    }


def _run_pytorch_benchmark(
    *,
    config: Path,
    checkpoint: Path,
    output: Path,
    device: str,
    samples: int,
    duration: float,
    data_dir: Path | None,
    instruction: str | None,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "forge.cli_v2",
        "benchmark",
        "run",
        "--config",
        str(config),
        "--checkpoint",
        str(checkpoint),
        "--device",
        device,
        "--samples",
        str(samples),
        "--duration",
        str(duration),
        "--output",
        str(output),
        "--json",
    ]
    if data_dir is not None:
        command.extend(("--data-dir", str(data_dir)))
    if instruction is not None:
        command.extend(("--instruction", instruction))
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "benchmark command failed"
        return {"status": "failed", "exit_code": completed.returncode, "error": detail}
    try:
        report = json.loads(completed.stdout, parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, ValueError) as exc:
        return {"status": "failed", "error": f"benchmark emitted invalid JSON: {exc}"}
    if not isinstance(report, dict):
        return {"status": "failed", "error": "benchmark emitted a non-object JSON report"}
    declared_status = report.get("status")
    if declared_status not in (None, "success", "completed"):
        detail = report.get("error") or report.get("reason") or f"benchmark declared status {declared_status!r}"
        return {**report, "status": "failed", "error": str(detail)}
    report["status"] = "success"
    report = cast(dict[str, Any], _portable_value(report))
    write_json_artifact(output, report)
    return report


def _quantized_architecture_evidence(
    quantized_artifacts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Require every quantized candidate to reload the same student architecture."""
    parameter_counts: dict[str, float] = {}
    errors: list[str] = []
    for name, artifact in quantized_artifacts.items():
        benchmark = artifact.get("benchmark")
        compression = benchmark.get("compression") if isinstance(benchmark, dict) else None
        params = compression.get("student_params_m") if isinstance(compression, dict) else None
        if not isinstance(params, (int, float)) or isinstance(params, bool) or not math.isfinite(params):
            errors.append(f"{name} did not report a finite student parameter count")
            continue
        parameter_counts[name] = float(params)

    reference = next(iter(parameter_counts.values()), None)
    consistent = False
    if reference is not None and len(parameter_counts) == len(quantized_artifacts):
        consistent = all(
            math.isclose(value, reference, rel_tol=0.0, abs_tol=0.0) for value in parameter_counts.values()
        )
    if not consistent and not errors:
        errors.append("quantized candidates do not share one pruned student architecture")

    result: dict[str, Any] = {
        "status": "success" if consistent else "failed",
        "consistent": consistent,
        "student_params_m": parameter_counts,
    }
    if reference is not None:
        result["reference_student_params_m"] = reference
    if errors:
        result["errors"] = errors
    return result


def _validated_backend_result(
    result: dict[str, Any],
    *,
    target: str,
    expected_precision: str | None = None,
    expected_device: str | None = None,
) -> dict[str, Any]:
    error: str | None = None
    if result.get("status") != "success":
        error = str(result.get("error") or result.get("reason") or f"{target} did not succeed")
    elif target not in {"ONNX", "TensorRT"}:
        execution = result.get("execution")
        result_device = result.get("device")
        if not isinstance(result_device, str) or not result_device.startswith("cuda"):
            error = f"{target} matrix execution did not report CUDA"
        elif expected_device is not None and result_device != expected_device:
            error = f"{target} matrix execution did not preserve selected device {expected_device}"
        elif not isinstance(execution, dict) or any(
            execution.get(key) != result_device for key in ("requested_device", "resolved_device")
        ):
            error = f"{target} matrix execution did not preserve requested and resolved CUDA device"
        elif result.get("actions_finite") is not True:
            error = f"{target} matrix execution did not prove finite actions"
        elif not isinstance(result.get("actions_shape"), list) or not result["actions_shape"]:
            error = f"{target} matrix execution did not report an action shape"
        elif not isinstance(result.get("action_samples"), int) or result["action_samples"] < 1:
            error = f"{target} matrix execution did not report real action samples"
        elif not isinstance(result.get("input_provenance"), dict) or result["input_provenance"].get("kind") != "real":
            error = f"{target} matrix execution did not use real input provenance"
    elif target == "ONNX":
        selected_device = torch.device(expected_device or "cuda")
        expected_device_id = selected_device.index if selected_device.index is not None else 0
        if result.get("provider") != "CUDAExecutionProvider":
            error = "ONNX matrix execution did not use CUDAExecutionProvider"
        elif result.get("device") != str(selected_device) or result.get("provider_device_id") != expected_device_id:
            error = f"ONNX matrix execution did not preserve selected device {selected_device}"
        elif result.get("actions_finite") is not True:
            error = "ONNX matrix execution did not prove finite actions"
        elif not isinstance(result.get("actions_shape"), list) or not result["actions_shape"]:
            error = "ONNX matrix execution did not report an action shape"
        elif not isinstance(result.get("action_samples"), int) or result["action_samples"] < 1:
            error = "ONNX matrix execution did not report action samples"
    elif target == "TensorRT":
        if result.get("provider") != "TensorRT":
            error = "TensorRT matrix execution did not use the TensorRT provider"
        elif result.get("actions_finite") is not True:
            error = "TensorRT matrix execution did not prove finite actions"
        elif result.get("precision") != expected_precision:
            error = f"TensorRT matrix precision does not match expected {expected_precision}"
    if error is None:
        return result
    return {**result, "status": "failed", "error": error}


def _publish_results_directory(staging: Path, destination: Path) -> None:
    """Publish an immutable run with one atomic current-pointer commit.

    The stable destination directory is never renamed or removed. Compatibility
    symlinks keep ``results_dir/summary.json`` and component paths usable, while
    every link resolves through the single atomically replaced ``current`` link.
    """
    if destination.exists() and not destination.is_dir():
        raise ValueError(f"Validation results destination is not a directory: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    runs_dir = destination / ".runs"
    runs_dir.mkdir(exist_ok=True)
    current = destination / "current"

    def fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def atomic_symlink(target: Path, link: Path) -> None:
        temporary = destination / f".{link.name}.tmp-{os.getpid()}"
        temporary.unlink(missing_ok=True)
        os.symlink(str(target), temporary)
        try:
            os.replace(temporary, link)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    # One-time migration of a legacy flat result directory. The old accepted
    # files remain visible while an identical immutable snapshot and pointer are
    # installed; summary.json is converted last.
    if not current.is_symlink():
        legacy_entries = [path for path in destination.iterdir() if path.name not in {".runs", "current"}]
        if legacy_entries:
            legacy_name = datetime.now(UTC).strftime("legacy-%Y%m%dT%H%M%S%fZ")
            legacy_run = runs_dir / legacy_name
            legacy_run.mkdir()
            for source in legacy_entries:
                target = legacy_run / source.name
                if source.is_dir():
                    shutil.copytree(source, target, symlinks=True)
                else:
                    shutil.copy2(source, target, follow_symlinks=True)
            fsync_directory(legacy_run)
            atomic_symlink(Path(".runs") / legacy_name, current)
            fsync_directory(destination)
            compatibility = sorted(legacy_entries, key=lambda path: path.name == "summary.json")
            for source in compatibility:
                if source.is_file() or source.is_symlink():
                    atomic_symlink(Path("current") / source.name, source)
            fsync_directory(destination)

    summary_path = staging / "summary.json"
    run_name = f"run-{_sha256(summary_path)[:20]}"
    published_run = runs_dir / run_name
    if published_run.exists():
        shutil.rmtree(staging)
    else:
        fsync_directory(staging)
        os.replace(staging, published_run)
        fsync_directory(runs_dir)

    output_files = sorted(
        (path for path in published_run.iterdir() if path.is_file()),
        key=lambda path: path.name == "summary.json",
    )
    if current.is_symlink():
        for output in output_files:
            compatibility_link = destination / output.name
            if compatibility_link.exists() and compatibility_link.is_dir():
                raise ValueError(f"Validation compatibility path is a directory: {compatibility_link}")
            atomic_symlink(Path("current") / output.name, compatibility_link)
    atomic_symlink(Path(".runs") / run_name, current)
    fsync_directory(destination)
    if not (destination / "summary.json").is_symlink():
        for output in output_files:
            atomic_symlink(Path("current") / output.name, destination / output.name)
    fsync_directory(destination)


def run_validation_matrix(
    manifest_path: str | Path,
    *,
    results_dir: str | Path,
    device: str = "cuda",
    samples: int = 20,
    duration: float = 2.0,
    onnx_warmup: int = 5,
    onnx_runs: int = 50,
) -> dict[str, Any]:
    """Benchmark every trained/checkpoint/export set in a manifest."""
    if samples < 1 or duration <= 0 or onnx_warmup < 0 or onnx_runs < 1:
        raise ValueError("Benchmark sample counts and durations must be positive")
    try:
        selected_device = torch.device(device)
    except (RuntimeError, ValueError) as exc:
        raise ValueError("Release validation matrix requires --device cuda or cuda:N") from exc
    if selected_device.type != "cuda":
        raise ValueError("Release validation matrix requires --device cuda or cuda:N")
    resolved_device = str(selected_device)
    execution = benchmark_execution(
        command="matrix",
        requested_device=device,
        resolved_device=resolved_device,
    )
    manifest, variants = load_validation_manifest(manifest_path)
    base_dir = manifest.parent
    output_dir = Path(results_dir).expanduser().resolve()
    if output_dir == output_dir.parent:
        raise ValueError("Validation results destination cannot be a filesystem root")
    prepared = [_prepare_variant(entry, base_dir=base_dir) for entry in variants]
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
    try:
        results: dict[str, dict[str, Any]] = {}
        for item in prepared:
            variant = item["variant"]
            runtime_images = item["runtime_images"]
            runtime_language_ids = item["runtime_language_ids"]
            pytorch_result = _run_pytorch_benchmark(
                config=item["config"],
                checkpoint=item["checkpoint"],
                output=staging / f"{variant}_pytorch.json",
                device=resolved_device,
                samples=samples,
                duration=duration,
                data_dir=item["data_dir"],
                instruction=item["instruction"],
            )
            pytorch_result = _validated_backend_result(
                pytorch_result,
                target="PyTorch",
                expected_device=resolved_device,
            )
            onnx_result = _validated_backend_result(
                benchmark_onnx_runtime(
                    item["onnx"],
                    device=resolved_device,
                    n_warmup=onnx_warmup,
                    n_runs=onnx_runs,
                    images=runtime_images,
                    language_ids=runtime_language_ids,
                ),
                target="ONNX",
                expected_device=resolved_device,
            )
            expected_precision = str(item["export_metrics"]["tensorrt"]["precision"])
            tensorrt_result = _validated_backend_result(
                benchmark_tensorrt_runtime(
                    item["tensorrt"],
                    n_warmup=onnx_warmup,
                    n_runs=onnx_runs,
                    images=runtime_images,
                    language_ids=runtime_language_ids,
                    precision=expected_precision,
                    device=resolved_device,
                ),
                target="TensorRT",
                expected_precision=expected_precision,
            )

            quantized_artifacts: dict[str, dict[str, Any]] = {}
            for name in sorted(QUANTIZED_CANDIDATES):
                artifact = item["quantized"][name]
                artifact_benchmark = _run_pytorch_benchmark(
                    config=item["config"],
                    checkpoint=artifact,
                    output=staging / f"{variant}_{name}.json",
                    device=resolved_device,
                    samples=samples,
                    duration=duration,
                    data_dir=item["data_dir"],
                    instruction=item["instruction"],
                )
                artifact_benchmark = _validated_backend_result(
                    artifact_benchmark,
                    target=f"Quantized {name}",
                    expected_device=resolved_device,
                )
                quantized_artifacts[name] = {
                    "artifact": artifact.name,
                    "sha256": item["artifact_sha256"][f"quantized:{name}"],
                    "size_mb": artifact.stat().st_size / 1e6,
                    "benchmark": artifact_benchmark,
                }
            quantized_architecture = _quantized_architecture_evidence(quantized_artifacts)

            variant_result = {
                "schema": RESULT_SCHEMA,
                "variant": variant,
                "timestamp": datetime.now(UTC).isoformat(),
                "execution": execution,
                "expected_training_steps": item["expected_training_steps"],
                "evidence_profile": item["evidence_profile"],
                "artifact_evidence": item["artifact_evidence"],
                "training": item["training_metrics"],
                "compression": item["compression_metrics"],
                "export": item["export_metrics"],
                "runtime_inputs": {
                    "artifact": item["runtime_inputs"].name,
                    "sha256": item["artifact_sha256"]["runtime_inputs"],
                    "images_shape": list(runtime_images.shape),
                    "language_ids_shape": list(runtime_language_ids.shape),
                    "provenance": item["export_metrics"]["runtime_inputs"],
                },
                "pytorch": pytorch_result,
                "onnxruntime": onnx_result,
                "tensorrt": tensorrt_result,
                "quantized_artifacts": quantized_artifacts,
                "quantized_architecture": quantized_architecture,
            }
            failures = [
                value
                for value in (
                    pytorch_result,
                    onnx_result,
                    tensorrt_result,
                    quantized_architecture,
                    *(candidate["benchmark"] for candidate in quantized_artifacts.values()),
                )
                if value.get("status") != "success"
            ]
            variant_result["status"] = "failed" if failures else "completed"
            variant_result = _portable_value(variant_result)
            write_json_artifact(staging / f"{variant}_validation.json", variant_result)
            results[variant] = variant_result

        status = "failed" if any(value["status"] == "failed" for value in results.values()) else "completed"
        summary = {
            "schema": RESULT_SCHEMA,
            "timestamp": datetime.now(UTC).isoformat(),
            "manifest": manifest.name,
            "manifest_sha256": _sha256(manifest),
            "device": resolved_device,
            "execution": execution,
            "status": status,
            "variants": results,
        }
        write_json_artifact(staging / "summary.json", summary)
        if status != "completed":
            return summary
        _publish_results_directory(staging, output_dir)
        return summary
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
