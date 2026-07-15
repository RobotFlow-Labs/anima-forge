"""Vision Encoder Registry — pluggable vision frontends for FORGE."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch.nn as nn


@dataclass
class VisionEncoderSpec:
    """Specification for a vision encoder."""

    name: str
    d_output: int  # Output feature dimension
    n_tokens: int  # Number of output tokens (for 384x384 input)
    param_count_m: float  # Parameters in millions
    input_size: int  # Expected input image size
    factory: Callable[..., nn.Module]  # Function that creates the encoder


class VisionEncoderRegistry:
    """Registry for vision encoder backends."""

    _encoders: dict[str, VisionEncoderSpec] = {}

    @classmethod
    def register(cls, spec: VisionEncoderSpec) -> None:
        cls._encoders[spec.name] = spec

    @classmethod
    def create(cls, name: str, model_dir: str | None = None, *, allow_mock: bool = False) -> nn.Module:
        if name not in cls._encoders:
            available = ", ".join(cls._encoders.keys())
            raise KeyError(f"Unknown encoder '{name}'. Available: {available}")
        return cls._encoders[name].factory(model_dir=model_dir, allow_mock=allow_mock)

    @classmethod
    def get_spec(cls, name: str) -> VisionEncoderSpec:
        if name not in cls._encoders:
            raise KeyError(f"Unknown encoder: {name}")
        return cls._encoders[name]

    @classmethod
    def list_encoders(cls) -> list[str]:
        return sorted(cls._encoders.keys())

    @classmethod
    def reset(cls) -> None:
        """Reset registry (for testing)."""
        cls._encoders.clear()
