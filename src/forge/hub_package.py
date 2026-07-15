"""Build privacy-safe Hugging Face payloads from accepted FORGE checkpoints."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from forge.checkpoint_compat import extract_checkpoint_state_dict
from forge.config import STUDENT_VARIANT_PRESETS
from forge.provenance import ProvenanceBlock, require_real_provenance

HUB_CHECKPOINT_SCHEMA = "forge.hub-checkpoint.v1"
HUB_MANIFEST_SCHEMA = "forge.hub-package.v1"
PRIVATE_PATTERN = re.compile(
    r"/(?:mnt|home|Users|tmp|root|opt|srv)(?:/|(?=$))|"
    r"(?<![A-Za-z0-9+.-])[A-Za-z]:[\\/]|\\\\[^\\/\s]+[\\/][^\\/\s]+|"  # forge-public-audit: allow[private-unc-path]
    r"\bdatai_srv\w*\b|\bhf_[A-Za-z0-9]{20,}\b|"
    r"\bgh(?:p|o|u|s|r)_[A-Za-z0-9]{20,}(?![A-Za-z0-9])|"
    r"\bgithub_pat_[A-Za-z0-9_]{20,}(?![A-Za-z0-9_])|"
    r"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{20,}(?![A-Za-z0-9_-])|"
    r"\b(?:AKIA|ASIA|AIDA|AROA|AIPA|ANPA|ANVA|ASCA)[A-Z0-9]{16}\b|"
    r"\b(?:aws[_-]?secret[_-]?access[_-]?key|aws[_-]?session[_-]?token)\b\s*[:=]\s*"
    r"['\"]?[A-Za-z0-9/+=]{20,}|-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----",
    re.IGNORECASE,
)
REPO_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}\Z")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
GIT_SHA_PATTERN = re.compile(r"[0-9a-f]{40}\Z")
CONTROL_PATTERN = re.compile(r"[\x00-\x1f\x7f]")
PRIVATE_IPV4_PATTERN = re.compile(
    r"(?<![\d.])(?:"
    r"10(?:\.\d{1,3}){3}|"
    r"192\.168(?:\.\d{1,3}){2}|"
    r"172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2}"
    r")(?![\d.])"
)
STUDENT_CONFIG_FIELDS = frozenset(
    {
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
        "lora_rank",
        "lora_alpha",
        "lora_target_modules",
        "action_horizon",
        "chunk_overlap",
        "action_head_type",
        "flow_inference_steps",
        "autosense",
        "allow_mock",
    }
)
POSITIVE_STUDENT_INTEGER_FIELDS = frozenset(
    {
        "bridge_d_vision",
        "bridge_d_model",
        "bridge_n_queries",
        "bridge_n_heads",
        "bridge_n_layers",
        "action_dim",
        "action_head_layers",
        "action_diffusion_steps",
        "lora_rank",
        "lora_alpha",
        "action_horizon",
        "flow_inference_steps",
    }
)
MODULE_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _finite_number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _accepted_training_summary(
    summary_path: Path,
    *,
    checkpoint_provenance: ProvenanceBlock,
    checkpoint_step: int,
    checkpoint_sha256: str,
    variant: str,
) -> dict[str, Any]:
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Training summary is unreadable: {summary_path}") from exc
    summary = _mapping(summary, name="training summary")
    if summary.get("status") != "completed":
        raise ValueError("Training summary status must be completed")
    if summary.get("config") != variant:
        raise ValueError(f"Training summary config must match checkpoint variant {variant!r}")

    distill = _mapping(summary.get("distill"), name="training summary distill block")
    summary_provenance = require_real_provenance(
        distill.get("provenance"),
        action="package a Hub checkpoint from",
    )
    if summary_provenance is None or summary_provenance != checkpoint_provenance:
        raise ValueError("Training summary provenance does not match checkpoint provenance")

    total_steps = distill.get("total_steps")
    if isinstance(total_steps, bool) or not isinstance(total_steps, int) or total_steps != checkpoint_step:
        raise ValueError("Training summary total_steps does not match checkpoint step")
    accepted_sha256 = distill.get("checkpoint_sha256")
    if not isinstance(accepted_sha256, str) or not SHA256_PATTERN.fullmatch(accepted_sha256.lower()):
        raise ValueError("Training summary must contain a valid checkpoint_sha256")
    if accepted_sha256.lower() != checkpoint_sha256:
        raise ValueError("Training summary is not bound to this exact checkpoint")
    memory = _mapping(distill.get("cuda_memory"), name="training summary CUDA memory block")
    if memory.get("target_60_80_percent_met") is not True:
        raise ValueError("Training summary did not pass the required 60–80% VRAM gate")

    return {
        "total_steps": total_steps,
        "initial_loss": _finite_number(distill.get("initial_loss"), name="initial_loss"),
        "final_loss": _finite_number(distill.get("final_loss"), name="final_loss"),
        "loss_reduction_percent": _finite_number(distill.get("loss_reduction_percent"), name="loss_reduction_percent"),
        "best_loss": _finite_number(distill.get("best_loss"), name="best_loss"),
        "steps_per_second": _finite_number(distill.get("steps_per_second"), name="steps_per_second"),
        "elapsed_seconds": _finite_number(distill.get("elapsed_seconds"), name="elapsed_seconds"),
        "cuda_memory": {
            "peak_reserved_gib": _finite_number(memory.get("peak_reserved_gib"), name="peak_reserved_gib"),
            "total_gib": _finite_number(memory.get("total_gib"), name="total_gib"),
            "peak_reserved_utilization": _finite_number(
                memory.get("peak_reserved_utilization"), name="peak_reserved_utilization"
            ),
            "target_60_80_percent_met": True,
        },
    }


def _private_strings(value: object, path: str = "") -> list[str]:
    findings: list[str] = []
    if isinstance(value, str):
        if _has_private_pattern(value) or CONTROL_PATTERN.search(value):
            findings.append(path or "<root>")
    elif isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            if isinstance(key, str) and (_has_private_pattern(key) or CONTROL_PATTERN.search(key)):
                findings.append(f"{child}.<key>")
            findings.extend(_private_strings(item, child))
    elif isinstance(value, list | tuple):
        for index, item in enumerate(value):
            findings.extend(_private_strings(item, f"{path}[{index}]"))
    return findings


def _has_private_pattern(value: str) -> bool:
    if PRIVATE_PATTERN.search(value):
        return True
    for match in PRIVATE_IPV4_PATTERN.finditer(value):
        prefix = value[: match.start()]
        if re.search(
            r"(?:\bversion(?:\s*[:=]\s*['\"]?|\s+)|(?:==|>=|<=|~=|!=)\s*|"
            r"(?:TensorRT|CUDA|cuDNN)\s+|"
            r"(?:tensorrt(?:[_-][A-Za-z0-9_]+)?|nvidia[_-][A-Za-z0-9_-]+)[_-])$",
            prefix,
            re.IGNORECASE,
        ):
            continue
        return True
    return False


def _validated_student_config(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the complete architecture contract required to restore a Hub student."""
    if any(not isinstance(key, str) for key in value):
        raise ValueError("checkpoint student_config keys must be strings")
    supplied_fields = set(value)
    missing_fields = sorted(STUDENT_CONFIG_FIELDS - supplied_fields)
    unexpected_fields = sorted(supplied_fields - STUDENT_CONFIG_FIELDS)
    if missing_fields:
        raise ValueError("checkpoint student_config is missing required fields: " + ", ".join(missing_fields))
    if unexpected_fields:
        raise ValueError("checkpoint student_config contains unsupported fields: " + ", ".join(unexpected_fields))

    config = dict(value)
    variant = config["variant"]
    if not isinstance(variant, str) or variant not in STUDENT_VARIANT_PRESETS:
        raise ValueError(f"checkpoint student_config.variant must be one of {sorted(STUDENT_VARIANT_PRESETS)}")
    for field in ("vision_encoder", "language_model"):
        model_id = config[field]
        if not isinstance(model_id, str) or not REPO_ID_PATTERN.fullmatch(model_id):
            raise ValueError(f"checkpoint student_config.{field} must be a canonical Hub model ID")
    if config["backbone_dtype"] not in {"auto", "float32", "float16", "bfloat16"}:
        raise ValueError("checkpoint student_config.backbone_dtype is unsupported")
    for field in POSITIVE_STUDENT_INTEGER_FIELDS:
        field_value = config[field]
        if isinstance(field_value, bool) or not isinstance(field_value, int) or field_value < 1:
            raise ValueError(f"checkpoint student_config.{field} must be a positive integer")
    chunk_overlap = config["chunk_overlap"]
    if isinstance(chunk_overlap, bool) or not isinstance(chunk_overlap, int) or chunk_overlap < 0:
        raise ValueError("checkpoint student_config.chunk_overlap must be a non-negative integer")
    if config["bridge_d_model"] % config["bridge_n_heads"]:
        raise ValueError("checkpoint student_config.bridge_d_model must be divisible by bridge_n_heads")
    action_head_type = config["action_head_type"]
    if action_head_type not in {"diffusion", "flow", "chunk", "consistency"}:
        raise ValueError("checkpoint student_config.action_head_type is unsupported")
    target_modules = config["lora_target_modules"]
    if (
        not isinstance(target_modules, list)
        or not target_modules
        or any(not isinstance(name, str) or not MODULE_NAME_PATTERN.fullmatch(name) for name in target_modules)
    ):
        raise ValueError("checkpoint student_config.lora_target_modules must be a non-empty list of module names")
    if not isinstance(config["autosense"], bool):
        raise ValueError("checkpoint student_config.autosense must be a boolean")
    if config["allow_mock"] is not False:
        raise ValueError("checkpoint student_config.allow_mock must be false for a public Hub package")
    return config


def _model_card(
    *,
    repo_id: str,
    artifact_name: str,
    variant: str,
    student_config: Mapping[str, Any],
    provenance: ProvenanceBlock,
    metrics: Mapping[str, Any],
    artifact_sha256: str,
) -> str:
    language_model = student_config.get("language_model", "unknown")
    vision_encoder = student_config.get("vision_encoder", "unknown")
    memory = _mapping(metrics["cuda_memory"], name="packaged CUDA memory")
    return "\n".join(
        [
            "---",
            "license: apache-2.0",
            "library_name: forge",
            "pipeline_tag: robotics",
            "tags:",
            "  - robotics",
            "  - vision-language-action",
            "  - knowledge-distillation",
            "---",
            "",
            f"# FORGE {variant}",
            "",
            "This repository contains a provenance-verified FORGE student checkpoint.",
            "It requires the external backbones named below and the `anima-forge` runtime.",
            "",
            "## Architecture",
            "",
            f"- Vision encoder: `{vision_encoder}`",
            f"- Language backbone: `{language_model}`",
            f"- Action head: `{student_config.get('action_head_type', 'unknown')}`",
            f"- Training step: {metrics['total_steps']}",
            "",
            "## Accepted launch training",
            "",
            f"- Loss: {metrics['initial_loss']:.6f} → {metrics['final_loss']:.6f}",
            f"- Loss reduction: {metrics['loss_reduction_percent']:.2f}%",
            f"- Best loss: {metrics['best_loss']:.6f}",
            f"- Throughput: {metrics['steps_per_second']:.4f} steps/s",
            (
                f"- Peak reserved VRAM: {memory['peak_reserved_gib']:.3f}/"
                f"{memory['total_gib']:.3f} GiB ({memory['peak_reserved_utilization']:.2%})"
            ),
            "- Inputs: real vision weights, real language weights, and real teacher labels",
            "",
            "## Integrity",
            "",
            f"- Artifact: `{artifact_name}`",
            f"- SHA-256: `{artifact_sha256}`",
            f"- Training source revision: `{provenance['git_sha']}`",
            "",
            "## Use",
            "",
            "```bash",
            "pip install anima-forge",
            f"hf download {repo_id} {artifact_name} --local-dir ./forge-model",
            f"forge serve --checkpoint ./forge-model/{artifact_name}",
            "```",
            "",
            "Run `forge models fetch --all-students` to install the required external backbones.",
            "Robot deployment requires task-specific evaluation and safety validation.",
            "",
        ]
    )


def package_hub_checkpoint(
    checkpoint_path: str | Path,
    training_summary_path: str | Path,
    output_dir: str | Path,
    *,
    repo_id: str,
) -> dict[str, Any]:
    """Create an atomic, privacy-safe inference package for Hugging Face Hub."""
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    training_summary_path = Path(training_summary_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise ValueError(f"Checkpoint not found: {checkpoint_path}")
    if not training_summary_path.is_file():
        raise ValueError(f"Training summary not found: {training_summary_path}")
    if output_dir.exists():
        if not output_dir.is_dir() or any(output_dir.iterdir()):
            raise ValueError(f"Output directory must be absent or empty: {output_dir}")
    if not REPO_ID_PATTERN.fullmatch(repo_id) or any(character.isspace() for character in repo_id):
        raise ValueError("repo_id must be a canonical Hugging Face namespace/name")

    checkpoint_sha256 = _sha256(checkpoint_path)

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    except Exception as exc:
        raise ValueError(f"Checkpoint is not a safe tensor-only payload: {checkpoint_path}") from exc
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint payload must be a mapping")

    provenance = require_real_provenance(
        checkpoint.get("provenance"),
        action="package for Hugging Face Hub",
    )
    if provenance is None:
        raise ValueError("Hub packaging requires complete real checkpoint provenance")
    if not GIT_SHA_PATTERN.fullmatch(provenance["git_sha"].lower()):
        raise ValueError("checkpoint provenance git_sha must be a full 40-character revision")
    raw_student_config = _mapping(checkpoint.get("student_config"), name="checkpoint student_config")
    private_student_fields = _private_strings(raw_student_config, "student_config")
    if private_student_fields:
        raise ValueError("Packaged checkpoint metadata contains private values: " + ", ".join(private_student_fields))
    student_config = _validated_student_config(raw_student_config)
    variant = student_config["variant"]
    step = checkpoint.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 1:
        raise ValueError("checkpoint step must be a positive integer")
    state_dict, _ = extract_checkpoint_state_dict(checkpoint)
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("Checkpoint has no model state dictionary")
    invalid_state_entries = [
        str(key)
        for key, value in state_dict.items()
        if not isinstance(key, str)
        or _has_private_pattern(str(key))
        or CONTROL_PATTERN.search(str(key))
        or not torch.is_tensor(value)
    ]
    if invalid_state_entries:
        raise ValueError("Checkpoint state dictionary must contain only named tensors")

    metrics = _accepted_training_summary(
        training_summary_path,
        checkpoint_provenance=provenance,
        checkpoint_step=step,
        checkpoint_sha256=checkpoint_sha256,
        variant=variant,
    )
    public_provenance = dict(provenance)
    public_provenance["model_dir"] = "external Hub model IDs in student_config"
    package = {
        "format": HUB_CHECKPOINT_SCHEMA,
        "step": step,
        "model_state_dict": state_dict,
        "provenance": public_provenance,
        "student_config": dict(student_config),
    }
    private_fields = _private_strings(package)
    if private_fields:
        raise ValueError("Packaged checkpoint metadata contains private values: " + ", ".join(private_fields))

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    artifact_name = f"forge-{variant}.pt"
    try:
        artifact_path = temporary / artifact_name
        torch.save(package, artifact_path)
        artifact_sha256 = _sha256(artifact_path)
        manifest = {
            "schema": HUB_MANIFEST_SCHEMA,
            "repo_id": repo_id,
            "variant": variant,
            "artifact": artifact_name,
            "artifact_sha256": artifact_sha256,
            "artifact_size_bytes": artifact_path.stat().st_size,
            "source_checkpoint": checkpoint_path.name,
            "source_checkpoint_sha256": checkpoint_sha256,
            "state_dict_entries": len(state_dict),
            "stripped_training_state": ["optimizer_state_dict", "scheduler_state_dict"],
            "provenance": public_provenance,
            "training": metrics,
        }
        card = _model_card(
            repo_id=repo_id,
            artifact_name=artifact_name,
            variant=variant,
            student_config=student_config,
            provenance=public_provenance,
            metrics=metrics,
            artifact_sha256=artifact_sha256,
        )
        private_fields = _private_strings(manifest)
        if private_fields or _has_private_pattern(card):
            raise ValueError("Hub metadata contains a private path, address, or credential pattern")
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (temporary / "README.md").write_text(card, encoding="utf-8")
        if output_dir.exists():
            output_dir.rmdir()
        temporary.rename(output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return {
        "status": "completed",
        "output_dir": str(output_dir),
        "repo_id": repo_id,
        "artifact": str(output_dir / artifact_name),
        "manifest": str(output_dir / "manifest.json"),
        "artifact_sha256": artifact_sha256,
        "source_checkpoint_sha256": manifest["source_checkpoint_sha256"],
        "state_dict_entries": len(state_dict),
        "provenance": public_provenance,
        "training": metrics,
    }


__all__ = ["HUB_CHECKPOINT_SCHEMA", "HUB_MANIFEST_SCHEMA", "package_hub_checkpoint"]
