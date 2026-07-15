"""Shared types for the FORGE pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TeacherOutput:
    """Output from a single teacher inference step."""

    action_logits: np.ndarray  # (D_action,) float32
    action_mean: np.ndarray  # (D_action,) float32
    action_std: np.ndarray  # (D_action,) float32
    vision_features: np.ndarray | None  # (N_tokens, D_vision) float16
    confidence: np.ndarray  # (D_action,) float32


@dataclass
class EpisodeData:
    """Complete data for one episode."""

    episode_id: str
    task_id: str
    language_instruction: str
    timesteps: int
    images: np.ndarray  # (T, H, W, 3) uint8
    proprioception: np.ndarray  # (T, D_proprio) float32
    teacher_action_logits: np.ndarray  # (T, H_action, D_action) float32
    teacher_action_mean: np.ndarray  # (T, H_action, D_action) float32
    teacher_action_std: np.ndarray  # (T, H_action, D_action) float32
    teacher_vision_features: np.ndarray | None  # (T, N_tokens, D_vision) float16
    confidence: np.ndarray  # (T, H_action, D_action) float32
    ground_truth_actions: np.ndarray  # (T, D_action) float32
    success: bool | None


@dataclass
class ChunkedTeacherOutput:
    """Teacher output with action chunking support (v2)."""

    action_chunk: np.ndarray  # (H, D_action)
    action_chunk_mean: np.ndarray  # (H, D_action)
    action_chunk_std: np.ndarray  # (H, D_action)
    vision_features: np.ndarray | None
    confidence: np.ndarray  # (H, D_action)
    teacher_name: str
    teacher_architecture: str
