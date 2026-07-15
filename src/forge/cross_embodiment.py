"""PRD-30: Cross-Embodiment Transfer Learning.

Transfer trained FORGE students between different robot embodiments.
Handles action space mapping, joint remapping, and morphology adaptation
so a model trained on Franka (7-DoF) can be deployed on UR5e (6-DoF)
or ALOHA (14-DoF bimanual).

Usage:
    from forge.cross_embodiment import (
        ActionSpaceMapper, EmbodimentTransfer, TransferConfig,
    )

    transfer = EmbodimentTransfer(
        source_profile=franka_profile,
        target_profile=ur5e_profile,
    )
    target_actions = transfer.map_actions(source_actions)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from forge.training_safety import backward_with_finite_gradients

logger = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────


@dataclass
class TransferConfig:
    """Configuration for cross-embodiment transfer."""

    mapping_strategy: str = "linear"  # "linear" | "learned" | "joint_name"
    scale_actions: bool = True  # Scale to target joint limits
    pad_strategy: str = "zero"  # "zero" | "mirror" | "repeat"
    trim_strategy: str = "first"  # "first" | "last" | "even"
    gripper_passthrough: bool = True  # Pass gripper action through unchanged
    adapter_hidden_dim: int = 64  # Hidden dim for learned mapping


# ── Action Space Mapper ───────────────────────────────────


class ActionSpaceMapper:
    """Maps actions between different action space dimensions.

    Handles dimension mismatches by padding (source < target) or
    trimming (source > target), with optional joint limit scaling.
    """

    def __init__(
        self,
        source_dim: int,
        target_dim: int,
        config: TransferConfig | None = None,
    ):
        self.source_dim = source_dim
        self.target_dim = target_dim
        self.config = config or TransferConfig()

        # Optional scaling parameters
        self._source_min: np.ndarray | None = None
        self._source_max: np.ndarray | None = None
        self._target_min: np.ndarray | None = None
        self._target_max: np.ndarray | None = None

    def set_joint_limits(
        self,
        source_min: list[float],
        source_max: list[float],
        target_min: list[float],
        target_max: list[float],
    ) -> None:
        """Set joint limits for action scaling."""
        self._source_min = np.array(source_min, dtype=np.float32)
        self._source_max = np.array(source_max, dtype=np.float32)
        self._target_min = np.array(target_min, dtype=np.float32)
        self._target_max = np.array(target_max, dtype=np.float32)

    def map(self, actions: np.ndarray) -> np.ndarray:
        """Map actions from source to target space.

        Args:
            actions: Source actions, shape (..., source_dim).

        Returns:
            Mapped actions, shape (..., target_dim).
        """
        original_shape = actions.shape
        flat = actions.reshape(-1, self.source_dim)

        # Scale to [0, 1] if limits are set
        if self.config.scale_actions and self._source_min is not None:
            src_range = self._source_max - self._source_min
            src_range = np.where(src_range == 0, 1.0, src_range)
            normalized = (flat - self._source_min) / src_range
        else:
            normalized = flat

        # Pad or trim to target dim
        if self.source_dim < self.target_dim:
            mapped = self._pad(normalized)
        elif self.source_dim > self.target_dim:
            mapped = self._trim(normalized)
        else:
            mapped = normalized

        # Scale to target limits
        if self.config.scale_actions and self._target_min is not None:
            tgt_range = self._target_max - self._target_min
            mapped = mapped * tgt_range + self._target_min

        # Restore batch dims
        new_shape = original_shape[:-1] + (self.target_dim,)
        return mapped.reshape(new_shape)

    def _pad(self, actions: np.ndarray) -> np.ndarray:
        """Pad actions to target_dim."""
        n_pad = self.target_dim - self.source_dim
        if self.config.pad_strategy == "zero":
            pad = np.zeros((actions.shape[0], n_pad), dtype=actions.dtype)
        elif self.config.pad_strategy == "mirror":
            # Mirror the last joints
            pad = actions[:, -n_pad:]
        elif self.config.pad_strategy == "repeat":
            # Repeat the last action value
            pad = np.tile(actions[:, -1:], (1, n_pad))
        else:
            pad = np.zeros((actions.shape[0], n_pad), dtype=actions.dtype)
        return np.concatenate([actions, pad], axis=-1)

    def _trim(self, actions: np.ndarray) -> np.ndarray:
        """Trim actions to target_dim."""
        if self.config.trim_strategy == "first":
            return actions[:, : self.target_dim]
        elif self.config.trim_strategy == "last":
            return actions[:, -self.target_dim :]
        elif self.config.trim_strategy == "even":
            # Evenly spaced selection
            indices = np.linspace(0, self.source_dim - 1, self.target_dim, dtype=int)
            return actions[:, indices]
        return actions[:, : self.target_dim]


# ── Joint Name Mapper ─────────────────────────────────────


class JointNameMapper:
    """Maps actions by matching joint names between embodiments.

    Uses semantic matching (e.g., "shoulder" in source maps to
    "shoulder" in target) for cross-morphology transfer.
    """

    def __init__(
        self,
        source_joints: list[str],
        target_joints: list[str],
    ):
        self.source_joints = source_joints
        self.target_joints = target_joints
        self._mapping: dict[int, int] = {}
        self._build_mapping()

    def _build_mapping(self) -> None:
        """Build joint index mapping based on name similarity."""
        for t_idx, t_name in enumerate(self.target_joints):
            best_match = -1
            best_score = 0.0
            for s_idx, s_name in enumerate(self.source_joints):
                score = self._similarity(s_name, t_name)
                if score > best_score:
                    best_score = score
                    best_match = s_idx
            if best_match >= 0 and best_score > 0.3:
                self._mapping[t_idx] = best_match

    @staticmethod
    def _similarity(a: str, b: str) -> float:
        """Simple string similarity (Jaccard on character 3-grams)."""
        if a == b:
            return 1.0
        a_lower, b_lower = a.lower(), b.lower()
        if a_lower == b_lower:
            return 0.95

        def trigrams(s: str) -> set[str]:
            return {s[i : i + 3] for i in range(max(0, len(s) - 2))}

        a_set = trigrams(a_lower)
        b_set = trigrams(b_lower)
        if not a_set or not b_set:
            return 0.0
        return len(a_set & b_set) / len(a_set | b_set)

    @property
    def mapping(self) -> dict[int, int]:
        """Target index → source index mapping."""
        return dict(self._mapping)

    @property
    def unmatched_target_joints(self) -> list[str]:
        """Return target joints that cannot be populated by this mapping."""
        return [name for index, name in enumerate(self.target_joints) if index not in self._mapping]

    def map(self, source_actions: np.ndarray) -> np.ndarray:
        """Map actions using joint name correspondence."""
        unmatched = self.unmatched_target_joints
        if unmatched:
            raise ValueError(
                f"Joint-name mapping cannot safely populate every target joint; unmatched target joints: {unmatched!r}"
            )
        target_dim = len(self.target_joints)
        if source_actions.ndim == 1:
            result = np.zeros(target_dim, dtype=source_actions.dtype)
            for t_idx, s_idx in self._mapping.items():
                result[t_idx] = source_actions[s_idx]
            return result

        batch = source_actions.reshape(-1, source_actions.shape[-1])
        result = np.zeros((batch.shape[0], target_dim), dtype=batch.dtype)
        for t_idx, s_idx in self._mapping.items():
            result[:, t_idx] = batch[:, s_idx]
        return result.reshape(source_actions.shape[:-1] + (target_dim,))


# ── Learned Adapter ───────────────────────────────────────


class LearnedActionAdapter(nn.Module):
    """Learned MLP adapter for cross-embodiment action mapping.

    Small MLP that transforms source actions to target action space.
    Trainable via fine-tuning on paired demonstrations.
    """

    def __init__(
        self,
        source_dim: int,
        target_dim: int,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(source_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, target_dim),
        )

    def forward(self, source_actions: torch.Tensor) -> torch.Tensor:
        return self.net(source_actions)


# ── Embodiment Transfer ───────────────────────────────────


@dataclass
class EmbodimentProfile:
    """Simplified profile for cross-embodiment transfer."""

    name: str
    action_dim: int
    joint_names: list[str] = field(default_factory=list)
    joint_min: list[float] = field(default_factory=list)
    joint_max: list[float] = field(default_factory=list)
    has_gripper: bool = False


class EmbodimentTransfer:
    """Cross-embodiment transfer coordinator.

    Manages action space mapping between source and target robots,
    supporting linear, joint-name, and learned mapping strategies.
    """

    def __init__(
        self,
        source_profile: EmbodimentProfile,
        target_profile: EmbodimentProfile,
        config: TransferConfig | None = None,
    ):
        self.source = source_profile
        self.target = target_profile
        self.config = config or TransferConfig()

        # Build mapper based on strategy
        self.mapper: ActionSpaceMapper | None = None
        self.joint_mapper: JointNameMapper | None = None
        self.learned_adapter: LearnedActionAdapter | None = None
        self._learned_is_fitted = False

        self._build()

    def _build(self) -> None:
        if self.config.mapping_strategy == "joint_name":
            self.joint_mapper = JointNameMapper(
                self.source.joint_names,
                self.target.joint_names,
            )
        elif self.config.mapping_strategy == "learned":
            self.learned_adapter = LearnedActionAdapter(
                self.source.action_dim,
                self.target.action_dim,
                self.config.adapter_hidden_dim,
            )
        else:  # linear
            self.mapper = ActionSpaceMapper(
                self.source.action_dim,
                self.target.action_dim,
                self.config,
            )
            if self.source.joint_min and self.target.joint_min:
                self.mapper.set_joint_limits(
                    self.source.joint_min,
                    self.source.joint_max,
                    self.target.joint_min,
                    self.target.joint_max,
                )

        logger.info(
            f"Transfer: {self.source.name}({self.source.action_dim}D) → "
            f"{self.target.name}({self.target.action_dim}D) "
            f"via {self.config.mapping_strategy}"
        )

    def map_actions(self, actions: np.ndarray) -> np.ndarray:
        """Map actions from source to target embodiment."""
        if self.config.mapping_strategy == "joint_name" and self.joint_mapper:
            return self.joint_mapper.map(actions)
        elif self.config.mapping_strategy == "learned" and self.learned_adapter:
            if not self._learned_is_fitted:
                raise RuntimeError(
                    "Learned action mapping is untrained; call fit_learned_adapter() or load trained adapter weights"
                )
            t = torch.from_numpy(actions).float()
            with torch.no_grad():
                result = self.learned_adapter(t)
            return result.numpy()
        elif self.mapper:
            return self.mapper.map(actions)
        return actions

    def fit_learned_adapter(
        self,
        source_actions: np.ndarray,
        target_actions: np.ndarray,
        *,
        steps: int = 100,
        learning_rate: float = 1e-3,
    ) -> dict[str, float | int]:
        """Fit the learned strategy on aligned source/target action pairs."""
        if self.learned_adapter is None or self.config.mapping_strategy != "learned":
            raise ValueError("fit_learned_adapter requires mapping_strategy='learned'")
        if steps < 1:
            raise ValueError("Learned adapter steps must be positive")

        source = torch.as_tensor(source_actions, dtype=torch.float32)
        target = torch.as_tensor(target_actions, dtype=torch.float32)
        if source.ndim != 2 or source.shape[1] != self.source.action_dim:
            raise ValueError(f"Expected source actions shaped [N, {self.source.action_dim}], got {tuple(source.shape)}")
        if target.ndim != 2 or target.shape != (source.shape[0], self.target.action_dim):
            raise ValueError(
                f"Expected target actions shaped [{source.shape[0]}, {self.target.action_dim}], "
                f"got {tuple(target.shape)}"
            )
        if not bool(torch.isfinite(source).all()) or not bool(torch.isfinite(target).all()):
            raise ValueError("Learned adapter training actions must be finite")

        adapter = self.learned_adapter
        adapter.train()
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()
        with torch.no_grad():
            loss_before = float(loss_fn(adapter(source), target).item())
        loss = torch.zeros((), dtype=torch.float32)
        for _ in range(steps):
            optimizer.zero_grad()
            loss = loss_fn(adapter(source), target)
            backward_with_finite_gradients(loss, adapter.parameters())
            optimizer.step()
        adapter.eval()
        with torch.no_grad():
            loss_after = float(loss_fn(adapter(source), target).item())
        self._learned_is_fitted = True
        return {
            "steps": steps,
            "samples": int(source.shape[0]),
            "learning_rate": learning_rate,
            "loss_before": loss_before,
            "loss_after": loss_after,
        }

    def map_actions_torch(self, actions: torch.Tensor) -> torch.Tensor:
        """Map actions using torch tensors (for training)."""
        if self.config.mapping_strategy == "learned" and self.learned_adapter:
            return self.learned_adapter(actions)
        # Fallback to numpy conversion
        np_actions = actions.detach().cpu().numpy()
        mapped = self.map_actions(np_actions)
        return torch.from_numpy(mapped).to(actions.device).float()

    def info(self) -> dict[str, Any]:
        """Transfer configuration info."""
        result: dict[str, Any] = {
            "source": self.source.name,
            "target": self.target.name,
            "source_dim": self.source.action_dim,
            "target_dim": self.target.action_dim,
            "strategy": self.config.mapping_strategy,
            "dim_change": self.target.action_dim - self.source.action_dim,
        }
        if self.joint_mapper:
            result["joint_mapping"] = self.joint_mapper.mapping
            result["unmatched_target_joints"] = self.joint_mapper.unmatched_target_joints
            result["joint_mapping_complete"] = not self.joint_mapper.unmatched_target_joints
        if self.learned_adapter:
            n_params = sum(p.numel() for p in self.learned_adapter.parameters())
            result["adapter_params"] = n_params
            result["learned_fitted"] = self._learned_is_fitted
        return result
