from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch


@dataclass
class ActionChunk:
    """Canonical output from ANY teacher, regardless of architecture type.

    All teachers normalize to this format. Single-step teachers
    produce horizon=1 chunks.
    """

    actions: np.ndarray  # (H, D_action) action chunk, H=horizon
    action_mean: np.ndarray  # (H, D_action) distribution mean
    action_std: np.ndarray  # (H, D_action) distribution std
    confidence: np.ndarray  # (H, D_action) per-dim confidence
    vision_features: np.ndarray | None = None  # (N_tokens, D_vision)
    language_features: np.ndarray | None = None  # (N_tokens, D_lang)
    metadata: dict = field(default_factory=dict)

    @property
    def horizon(self) -> int:
        return self.actions.shape[0]

    @property
    def action_dim(self) -> int:
        return self.actions.shape[1]


@dataclass
class TeacherInfo:
    """Metadata about a teacher model."""

    name: str  # e.g., "openvla-7b"
    architecture: str  # "token-ar" | "diffusion" | "flow" | "parallel"
    param_count: float  # in billions
    action_dim: int  # output action dimensionality
    action_horizon: int  # native action horizon (1 for single-step)
    vision_encoder: str  # e.g., "siglip-so400m"
    language_model: str  # e.g., "llama-2-7b"
    supports_chunking: bool  # can natively output multi-step
    supports_features: bool  # can extract intermediate features


class TeacherAdapter(ABC):
    """Abstract base class for all teacher model adapters.

    Every VLA teacher must implement this interface.
    The adapter handles:
    1. Loading the model from local weights
    2. Running inference and normalizing output to ActionChunk
    3. Extracting intermediate features for distillation
    """

    @abstractmethod
    def load(self, model_path: Path, device: str = "cpu", dtype: torch.dtype = torch.float32) -> None:
        """Load model weights from local path. Never downloads."""
        ...

    @abstractmethod
    def predict(
        self,
        image: np.ndarray,  # (H, W, 3) uint8
        instruction: str,  # natural language instruction
        proprioception: np.ndarray | None = None,  # (D_proprio,)
    ) -> ActionChunk:
        """Run inference and return canonical ActionChunk."""
        ...

    @abstractmethod
    def extract_features(
        self,
        image: np.ndarray,
        instruction: str,
    ) -> dict[str, np.ndarray]:
        """Extract intermediate features for distillation.

        Returns dict with keys like 'vision', 'language', 'hidden_states'.
        """
        ...

    @abstractmethod
    def get_action_space(self) -> dict:
        """Return action space specification.

        Returns:
            dict with 'dim', 'min', 'max', 'names' (per-dimension labels)
        """
        ...

    @abstractmethod
    def info(self) -> TeacherInfo:
        """Return teacher metadata."""
        ...

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Whether the model is currently loaded."""
        ...

    def unload(self) -> None:
        """Free model from memory. Default: no-op."""
        pass
