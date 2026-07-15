"""PRD-26: Trained Student Model Registry.

Track, version, and manage trained FORGE student models with metadata.
Uses JSON-based storage for simplicity and portability.

Usage:
    from forge.model_registry import ModelRegistry, ModelEntry

    registry = ModelRegistry(registry_dir="./outputs/registry")
    entry = registry.register(
        checkpoint_path="./outputs/checkpoints/best.pt",
        variant="nano",
        config=config,
        metrics={"final_loss": 0.023, "latency_ms": 45.2},
    )
    best = registry.best(by="final_loss")
    registry.promote(best.model_id, tag="production")
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from forge.provenance import (
    MOCK_WARNING,
    provenance_contains_mock,
    validate_provenance,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelEntry:
    """Metadata for a registered trained model."""

    model_id: str  # Unique ID (hash-based)
    name: str  # Human-readable name (e.g., "nano-siglip-qwen05b")
    variant: str  # Student variant: nano, small, micro
    checkpoint_path: str  # Path to .pt file
    created_at: float  # Unix timestamp
    tags: list[str] = field(default_factory=list)  # e.g., ["production", "best"]

    # Model architecture
    vision_encoder: str = ""
    language_model: str = ""
    action_head_type: str = "diffusion"
    bridge_d_vision: int = 0
    bridge_d_model: int = 0
    action_dim: int = 7
    action_horizon: int = 1

    # Training info
    total_steps: int = 0
    final_loss: float = 0.0
    best_loss: float = 0.0
    training_device: str = ""
    parent_teacher: str = ""

    # Performance metrics
    metrics: dict[str, float] = field(default_factory=dict)

    # Config hash for dedup
    config_hash: str = ""

    # PRD-36: real-vs-mock artifact lineage (None for legacy registry entries)
    provenance: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["is_mock"] = self.is_mock
        data["provenance_warning"] = MOCK_WARNING if self.is_mock else None
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelEntry:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @property
    def is_mock(self) -> bool:
        """Whether this entry contains any mock-derived component."""
        return provenance_contains_mock(self.provenance)

    @property
    def mock_warning(self) -> str:
        """Rich-formatted warning used by registry display surfaces."""
        return f"[red]{MOCK_WARNING}[/red]" if self.is_mock else ""

    @property
    def display_name(self) -> str:
        """Name with a prominent red warning for mock-derived models."""
        return f"{self.name} {self.mock_warning}".rstrip()

    @property
    def age_hours(self) -> float:
        return (time.time() - self.created_at) / 3600

    def summary(self) -> str:
        parts = [
            f"[{self.model_id[:8]}] {self.display_name}",
            f"  variant={self.variant}, steps={self.total_steps}",
            f"  loss={self.final_loss:.4f} (best={self.best_loss:.4f})",
        ]
        if self.tags:
            parts.append(f"  tags: {', '.join(self.tags)}")
        if self.metrics:
            metric_strs = [f"{k}={v:.4f}" for k, v in self.metrics.items()]
            parts.append(f"  metrics: {', '.join(metric_strs)}")
        return "\n".join(parts)


def _generate_model_id(variant: str, config_hash: str, timestamp: float) -> str:
    """Generate a unique model ID from variant + config + time."""
    raw = f"{variant}:{config_hash}:{timestamp}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _config_hash(config: Any) -> str:
    """Hash the student config for dedup detection."""
    if hasattr(config, "student"):
        cfg = config.student
    else:
        cfg = config

    parts = []
    for field_name in [
        "variant",
        "vision_encoder",
        "language_model",
        "bridge_d_vision",
        "bridge_d_model",
        "action_dim",
        "action_head_type",
        "action_horizon",
        "lora_rank",
    ]:
        parts.append(f"{field_name}={getattr(cfg, field_name, '')}")

    return hashlib.md5("|".join(parts).encode()).hexdigest()[:12]


def _load_checkpoint_provenance(path: Path) -> dict[str, str] | None:
    """Read and validate provenance from a checkpoint without legacy unpickling."""
    if not path.is_file():
        return None
    try:
        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        logger.warning("Could not read checkpoint provenance from %s: %s", path, exc)
        return None
    if not isinstance(payload, Mapping) or "provenance" not in payload:
        return None
    return validate_provenance(payload["provenance"])


class ModelRegistry:
    """JSON-based registry for trained FORGE student models.

    Storage layout:
        registry_dir/
        ├── registry.json    # Index of all models
        └── models/          # Copied checkpoints (optional)
    """

    def __init__(self, registry_dir: str | Path = "./outputs/registry"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.registry_dir / "registry.json"
        self._models_dir = self.registry_dir / "models"
        self._entries: dict[str, ModelEntry] = {}
        self._load()

    def _load(self) -> None:
        """Load registry index from disk."""
        if self._index_path.exists():
            try:
                data = json.loads(self._index_path.read_text())
                for entry_data in data.get("models", []):
                    entry = ModelEntry.from_dict(entry_data)
                    self._entries[entry.model_id] = entry
                logger.debug(f"Loaded {len(self._entries)} models from registry")
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Could not load registry: {e}")

    def _save(self) -> None:
        """Persist registry index to disk."""
        data = {
            "version": 1,
            "count": len(self._entries),
            "models": [e.to_dict() for e in self._entries.values()],
        }
        self._index_path.write_text(json.dumps(data, indent=2))

    def register(
        self,
        checkpoint_path: str | Path,
        variant: str,
        config: Any = None,
        metrics: dict[str, float] | None = None,
        training_report: Any = None,
        name: str | None = None,
        tags: list[str] | None = None,
        copy_checkpoint: bool = False,
        provenance: Mapping[str, Any] | None = None,
    ) -> ModelEntry:
        """Register a trained model in the registry.

        Args:
            checkpoint_path: Path to the .pt checkpoint file.
            variant: Student variant (nano, small, micro).
            config: ForgeConfig or StudentConfig (for metadata extraction).
            metrics: Performance metrics dict (latency_ms, throughput_fps, etc.).
            training_report: TrainingReport (for loss/step info).
            name: Human-readable name. Auto-generated if not provided.
            tags: Optional tags (e.g., ["production"]).
            copy_checkpoint: If True, copy checkpoint into registry dir.
            provenance: Explicit provenance override; otherwise read from checkpoint.

        Returns:
            ModelEntry for the registered model.
        """
        checkpoint_path = Path(checkpoint_path)
        now = time.time()

        checkpoint_provenance = _load_checkpoint_provenance(checkpoint_path)
        artifact_provenance: dict[str, str] | None
        if provenance is not None:
            explicit_provenance = validate_provenance(provenance)
            if checkpoint_provenance is not None and explicit_provenance != checkpoint_provenance:
                raise ValueError(
                    "Explicit registry provenance conflicts with the checkpoint provenance; "
                    "the artifact metadata is authoritative."
                )
            artifact_provenance = explicit_provenance
        else:
            artifact_provenance = checkpoint_provenance

        cfg_hash = _config_hash(config) if config else "unknown"
        model_id = _generate_model_id(variant, cfg_hash, now)

        # Extract config info
        student_cfg = None
        if config is not None:
            student_cfg = config.student if hasattr(config, "student") else config

        # Auto-generate name
        if name is None:
            vision_short = ""
            lm_short = ""
            if student_cfg:
                vision_short = getattr(student_cfg, "vision_encoder", "").split("/")[-1].split("-")[0]
                lm_short = getattr(student_cfg, "language_model", "").split("/")[-1]
            name = f"{variant}-{vision_short}-{lm_short}".strip("-")

        # Optionally copy checkpoint
        stored_path = str(checkpoint_path)
        if copy_checkpoint:
            self._models_dir.mkdir(parents=True, exist_ok=True)
            dest = self._models_dir / f"{model_id}.pt"
            shutil.copy2(checkpoint_path, dest)
            stored_path = str(dest)

        entry = ModelEntry(
            model_id=model_id,
            name=name,
            variant=variant,
            checkpoint_path=stored_path,
            created_at=now,
            tags=tags or [],
            config_hash=cfg_hash,
            metrics=metrics or {},
            provenance=artifact_provenance,
        )

        # Fill from config
        if student_cfg:
            entry.vision_encoder = getattr(student_cfg, "vision_encoder", "")
            entry.language_model = getattr(student_cfg, "language_model", "")
            entry.action_head_type = getattr(student_cfg, "action_head_type", "diffusion")
            entry.bridge_d_vision = getattr(student_cfg, "bridge_d_vision", 0)
            entry.bridge_d_model = getattr(student_cfg, "bridge_d_model", 0)
            entry.action_dim = getattr(student_cfg, "action_dim", 7)
            entry.action_horizon = getattr(student_cfg, "action_horizon", 1)

        # Fill from training report
        if training_report is not None:
            entry.total_steps = getattr(training_report, "total_steps", 0)
            entry.final_loss = getattr(training_report, "final_loss", 0.0)
            entry.best_loss = getattr(training_report, "best_loss", 0.0)
            entry.training_device = getattr(training_report, "device", "")

        # Fill from config paths
        if config and hasattr(config, "paths"):
            entry.parent_teacher = getattr(config.paths, "teacher", "")

        self._entries[model_id] = entry
        self._save()
        logger.info(f"Registered model: {entry.name} [{model_id[:8]}]")
        return entry

    def get(self, model_id: str) -> ModelEntry | None:
        """Get a model entry by ID (supports prefix match)."""
        if model_id in self._entries:
            return self._entries[model_id]
        # Prefix match
        matches = [e for mid, e in self._entries.items() if mid.startswith(model_id)]
        if len(matches) == 1:
            return matches[0]
        return None

    def list_models(
        self,
        variant: str | None = None,
        tag: str | None = None,
    ) -> list[ModelEntry]:
        """List all models, optionally filtered."""
        entries = list(self._entries.values())
        if variant:
            entries = [e for e in entries if e.variant == variant]
        if tag:
            entries = [e for e in entries if tag in e.tags]
        return sorted(entries, key=lambda e: e.created_at, reverse=True)

    def best(
        self,
        by: str = "best_loss",
        variant: str | None = None,
        lower_is_better: bool = True,
    ) -> ModelEntry | None:
        """Find the best model by a metric.

        Args:
            by: Metric key. Checks entry attributes first, then metrics dict.
            variant: Filter by variant.
            lower_is_better: If True, return minimum; else maximum.
        """
        entries = self.list_models(variant=variant)
        if not entries:
            return None

        def _get_value(e: ModelEntry) -> float:
            if hasattr(e, by) and isinstance(getattr(e, by), (int, float)):
                return float(getattr(e, by))
            return e.metrics.get(by, float("inf") if lower_is_better else float("-inf"))

        return min(entries, key=_get_value) if lower_is_better else max(entries, key=_get_value)

    def promote(self, model_id: str, tag: str = "production") -> ModelEntry | None:
        """Add a tag to a model. Removes the tag from any other model first."""
        # Remove tag from others
        for candidate in self._entries.values():
            if tag in candidate.tags:
                candidate.tags.remove(tag)

        entry = self.get(model_id)
        if entry is None:
            logger.warning(f"Model {model_id} not found")
            return None

        if tag not in entry.tags:
            entry.tags.append(tag)
        self._save()
        logger.info(f"Promoted [{model_id[:8]}] with tag '{tag}'")
        return entry

    def remove_tag(self, model_id: str, tag: str) -> ModelEntry | None:
        """Remove a tag from a model."""
        entry = self.get(model_id)
        if entry and tag in entry.tags:
            entry.tags.remove(tag)
            self._save()
        return entry

    def delete(self, model_id: str, delete_checkpoint: bool = False) -> bool:
        """Remove a model from the registry."""
        entry = self.get(model_id)
        if entry is None:
            return False

        if delete_checkpoint:
            ckpt = Path(entry.checkpoint_path)
            if ckpt.exists():
                ckpt.unlink()
                logger.info(f"Deleted checkpoint: {ckpt}")

        del self._entries[entry.model_id]
        self._save()
        logger.info(f"Removed model [{entry.model_id[:8]}] from registry")
        return True

    def compare(self, id1: str, id2: str) -> dict[str, Any]:
        """Compare two models side by side."""
        e1 = self.get(id1)
        e2 = self.get(id2)
        if not e1 or not e2:
            return {"error": "One or both models not found"}

        result: dict[str, Any] = {
            "model_1": {
                "id": e1.model_id,
                "name": e1.name,
                "is_mock": e1.is_mock,
                "provenance": e1.provenance,
            },
            "model_2": {
                "id": e2.model_id,
                "name": e2.name,
                "is_mock": e2.is_mock,
                "provenance": e2.provenance,
            },
            "differences": {},
        }

        # Compare key attributes
        for attr in [
            "variant",
            "final_loss",
            "best_loss",
            "total_steps",
            "vision_encoder",
            "language_model",
            "action_head_type",
            "bridge_d_vision",
            "bridge_d_model",
            "action_dim",
            "provenance",
        ]:
            v1 = getattr(e1, attr, None)
            v2 = getattr(e2, attr, None)
            if v1 != v2:
                result["differences"][attr] = {"model_1": v1, "model_2": v2}

        # Compare metrics
        all_metric_keys = set(e1.metrics) | set(e2.metrics)
        for key in sorted(all_metric_keys):
            v1 = e1.metrics.get(key)
            v2 = e2.metrics.get(key)
            if v1 != v2:
                result["differences"][f"metrics.{key}"] = {"model_1": v1, "model_2": v2}

        return result

    @property
    def count(self) -> int:
        return len(self._entries)

    def reset(self) -> None:
        """Clear all entries (for testing)."""
        self._entries.clear()
        self._save()
