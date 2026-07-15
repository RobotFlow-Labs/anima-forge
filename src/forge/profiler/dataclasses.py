"""Profiler dataclasses for FORGE model introspection."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ComponentProfile:
    """Profile of a single model component."""

    name: str
    param_count: int
    trainable_params: int
    frozen_params: int
    input_shape: str
    output_shape: str
    estimated_flops: int
    estimated_memory_mb: float


@dataclass
class FLOPsEstimate:
    """Per-component FLOPs breakdown."""

    vision_encoder: int
    bridge_attention: int
    language_backbone: int
    lora_adapters: int
    action_head: int
    total_gflops: float


@dataclass
class VRAMEstimate:
    """Memory estimation across precisions and use cases."""

    inference_mb: float
    inference_fp16_mb: float
    training_mb: float
    training_fp16_mb: float
    per_sample_activation_mb: float
    recommended_batch_size: int
    fits_gpu: dict[str, bool] = field(default_factory=dict)


@dataclass
class RecommendedHyperparams:
    """Training configuration recommendations."""

    learning_rate: float
    batch_size: int
    gradient_accumulation_steps: int
    effective_batch_size: int
    warmup_steps: int
    max_steps: int
    weight_decay: float
    lora_rank: int
    action_head_type: str
    bridge_n_queries: int
    bridge_n_layers: int
    flow_inference_steps: int
    rationale: dict[str, str] = field(default_factory=dict)


@dataclass
class ModelProfileCard:
    """Full profile card combining all profiler outputs for a FORGE model variant."""

    model_name: str
    variant: str
    vision_encoder: str
    language_model: str
    action_head_type: str
    action_dim: int
    action_horizon: int
    components: list[ComponentProfile] = field(default_factory=list)
    total_params: int = 0
    trainable_params: int = 0
    frozen_params: int = 0
    flops: FLOPsEstimate | None = None
    vram: VRAMEstimate | None = None
    recommended_hp: RecommendedHyperparams | None = None
    fp32_size_mb: float = 0.0
    fp16_size_mb: float = 0.0
    int8_size_mb: float = 0.0
    int4_size_mb: float = 0.0
    bridge_config: dict = field(default_factory=dict)
    architecture_diagram: str = ""
    timestamp: str = ""

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dict.

        Nested dataclasses are converted via ``dataclasses.asdict``.
        ``None`` optional fields are preserved as ``None`` so round-trips
        are lossless.
        """
        result: dict = {}

        for f in dataclasses.fields(self):
            value = getattr(self, f.name)

            if dataclasses.is_dataclass(value) and not isinstance(value, type):
                result[f.name] = dataclasses.asdict(value)
            elif isinstance(value, list):
                result[f.name] = [
                    dataclasses.asdict(item) if dataclasses.is_dataclass(item) and not isinstance(item, type) else item
                    for item in value
                ]
            else:
                result[f.name] = value

        return result

    @classmethod
    def from_dict(cls, d: dict) -> ModelProfileCard:
        """Reconstruct a ``ModelProfileCard`` from a plain dict.

        Re-hydrates nested dataclasses from their sub-dicts.
        """
        data = dict(d)

        if isinstance(data.get("components"), list):
            data["components"] = [
                ComponentProfile(**item) if isinstance(item, dict) else item for item in data["components"]
            ]

        if isinstance(data.get("flops"), dict):
            data["flops"] = FLOPsEstimate(**data["flops"])

        if isinstance(data.get("vram"), dict):
            data["vram"] = VRAMEstimate(**data["vram"])

        if isinstance(data.get("recommended_hp"), dict):
            data["recommended_hp"] = RecommendedHyperparams(**data["recommended_hp"])

        return cls(**data)

    @classmethod
    def from_json(cls, path: str) -> ModelProfileCard:
        """Load a ``ModelProfileCard`` from a JSON file.

        Args:
            path: Absolute or relative path to the JSON file.

        Returns:
            A fully re-hydrated ``ModelProfileCard`` instance.
        """
        raw = Path(path).read_text(encoding="utf-8")
        return cls.from_dict(json.loads(raw))

    def save_json(self, path: str) -> None:
        """Serialise the profile card to a JSON file.

        Args:
            path: Destination file path. Parent directories must exist.
        """
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2),
            encoding="utf-8",
        )
