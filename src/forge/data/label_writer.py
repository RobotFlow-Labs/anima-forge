"""HDF5-based label writer and reader for teacher soft labels.

Stores episode data in HDF5 format for efficient random-access loading
during knowledge distillation training (PRD-03).

Format:
    teacher_labels/
    ├── metadata.json          # Schema version, teacher model, stats
    ├── episodes_0000.h5       # Episodes 0-99
    ├── episodes_0001.h5       # Episodes 100-199
    └── ...
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import h5py  # type: ignore[import-untyped]

from forge.types import EpisodeData

logger = logging.getLogger(__name__)

LabelProvenance = Literal["real", "mock"]


def labels_provenance_from_metadata(metadata: Mapping[str, object]) -> LabelProvenance:
    """Return explicit label provenance, treating absent/invalid evidence as mock."""
    provenance = metadata.get("provenance")
    if isinstance(provenance, Mapping) and provenance.get("labels") == "real":
        return "real"
    return "mock"


class LabelWriter:
    """Streaming HDF5 writer for teacher labels with automatic chunking."""

    def __init__(
        self,
        output_dir: str,
        schema_version: str = "1.0",
        episodes_per_file: int = 100,
        save_vision_features: bool = True,
        save_attention: bool = False,
        labels_provenance: LabelProvenance = "mock",
        source_metadata: Mapping[str, object] | None = None,
    ):
        if labels_provenance not in {"real", "mock"}:
            raise ValueError("labels_provenance must be real or mock")
        if isinstance(episodes_per_file, bool) or not isinstance(episodes_per_file, int) or episodes_per_file < 1:
            raise ValueError("episodes_per_file must be a positive integer")

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.schema_version = schema_version
        self.episodes_per_file = int(episodes_per_file)
        self.save_vision_features = save_vision_features
        self.save_attention = save_attention
        self.labels_provenance = labels_provenance
        self.source_metadata = dict(source_metadata or {})

        self._episode_count = 0
        self._current_file: h5py.File | None = None
        self._current_file_idx = 0
        self._stats: dict[str, Any] = {
            "total_episodes": 0,
            "successful_episodes": 0,
            "success_known_episodes": 0,
            "total_timesteps": 0,
            "tasks": set(),
        }

    def _get_file(self) -> h5py.File:
        """Get current HDF5 file, creating new one if needed."""
        file_idx = self._episode_count // self.episodes_per_file

        if self._current_file is None or file_idx != self._current_file_idx:
            if self._current_file is not None:
                self._current_file.close()

            self._current_file_idx = file_idx
            path = self.output_dir / f"episodes_{file_idx:04d}.h5"
            self._current_file = h5py.File(path, "w")
            logger.info(f"Created label file: {path}")

        return self._current_file

    def write_episode(self, episode: EpisodeData) -> None:
        """Write a single episode to HDF5."""
        self._validate_episode(episode)
        f = self._get_file()
        ep_idx = self._episode_count % self.episodes_per_file
        grp = f.create_group(f"episode_{ep_idx:04d}")

        # Metadata
        grp.attrs["episode_id"] = episode.episode_id
        grp.attrs["task_id"] = episode.task_id
        grp.attrs["language_instruction"] = episode.language_instruction
        grp.attrs["timesteps"] = episode.timesteps
        grp.attrs["success_known"] = episode.success is not None
        if episode.success is not None:
            grp.attrs["success"] = episode.success

        # Observations
        obs = grp.create_group("observations")
        obs.create_dataset("images", data=episode.images, compression="gzip", compression_opts=4)
        obs.create_dataset("proprioception", data=episode.proprioception)

        # Teacher outputs
        teacher = grp.create_group("teacher_outputs")
        teacher.create_dataset("action_logits", data=episode.teacher_action_logits)
        teacher.create_dataset("action_mean", data=episode.teacher_action_mean)
        teacher.create_dataset("action_std", data=episode.teacher_action_std)
        teacher.create_dataset("confidence", data=episode.confidence)

        if self.save_vision_features and episode.teacher_vision_features is not None:
            teacher.create_dataset(
                "vision_features",
                data=episode.teacher_vision_features,
                compression="gzip",
                compression_opts=1,
            )

        # Ground truth
        grp.create_dataset("ground_truth_actions", data=episode.ground_truth_actions)

        # Update stats
        self._episode_count += 1
        self._stats["total_episodes"] += 1
        self._stats["total_timesteps"] += episode.timesteps
        self._stats["tasks"].add(episode.task_id)
        if episode.success is not None:
            self._stats["success_known_episodes"] += 1
        if episode.success is True:
            self._stats["successful_episodes"] += 1

    @staticmethod
    def _validate_episode(episode: EpisodeData) -> None:
        """Reject corrupt label payloads before creating or mutating an HDF5 group."""
        import numpy as np

        if isinstance(episode.timesteps, bool) or not isinstance(episode.timesteps, int) or episode.timesteps < 1:
            raise ValueError("Teacher-label episode timesteps must be a positive integer")
        if not isinstance(episode.episode_id, str) or not episode.episode_id.strip():
            raise ValueError("Teacher-label episode_id must be non-empty")
        if not isinstance(episode.task_id, str) or not episode.task_id.strip():
            raise ValueError("Teacher-label task_id must be non-empty")
        if not isinstance(episode.language_instruction, str) or not episode.language_instruction.strip():
            raise ValueError("Teacher-label language_instruction must be non-empty")

        timesteps = int(episode.timesteps)
        images = np.asarray(episode.images)
        if images.ndim != 4 or images.shape[0] != timesteps or images.shape[-1] != 3 or images.dtype != np.uint8:
            raise ValueError("Teacher-label images must have shape (timesteps, height, width, 3) and dtype uint8")
        if images.shape[1] < 1 or images.shape[2] < 1:
            raise ValueError("Teacher-label images must have positive height and width")

        proprioception = np.asarray(episode.proprioception)
        if proprioception.ndim != 2 or proprioception.shape[0] != timesteps:
            raise ValueError("Teacher-label proprioception must have shape (timesteps, dimensions)")
        if not np.issubdtype(proprioception.dtype, np.number) or not np.isfinite(proprioception).all():
            raise ValueError("Teacher-label proprioception must contain finite numeric values")

        action_arrays = {
            "teacher_action_logits": np.asarray(episode.teacher_action_logits),
            "teacher_action_mean": np.asarray(episode.teacher_action_mean),
            "teacher_action_std": np.asarray(episode.teacher_action_std),
            "confidence": np.asarray(episode.confidence),
        }
        action_shape = action_arrays["teacher_action_logits"].shape
        if len(action_shape) not in {2, 3} or action_shape[0] != timesteps or any(size < 1 for size in action_shape):
            raise ValueError("Teacher action arrays must have shape (timesteps, D) or (timesteps, H, D)")
        if any(value.shape != action_shape for value in action_arrays.values()):
            raise ValueError("Teacher action arrays must all have identical shapes")
        for name, value in action_arrays.items():
            if not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
                raise ValueError(f"{name} must contain finite numeric values")
        if np.any(action_arrays["teacher_action_std"] < 0):
            raise ValueError("teacher_action_std must be non-negative")
        confidence = action_arrays["confidence"]
        if np.any((confidence < 0) | (confidence > 1)):
            raise ValueError("confidence must be within [0, 1]")

        action_dim = action_shape[-1]
        ground_truth = np.asarray(episode.ground_truth_actions)
        if ground_truth.shape != (timesteps, action_dim):
            raise ValueError(
                f"ground_truth_actions must have shape {(timesteps, action_dim)}, got {ground_truth.shape}"
            )
        if not np.issubdtype(ground_truth.dtype, np.number) or not np.isfinite(ground_truth).all():
            raise ValueError("ground_truth_actions must contain finite numeric values")

        if episode.teacher_vision_features is not None:
            features = np.asarray(episode.teacher_vision_features)
            if features.ndim < 2 or features.shape[0] != timesteps:
                raise ValueError("teacher_vision_features must begin with the episode timestep dimension")
            if not np.issubdtype(features.dtype, np.number) or not np.isfinite(features).all():
                raise ValueError("teacher_vision_features must contain finite numeric values")

    def finalize(self) -> dict[str, Any]:
        """Close files and write metadata."""
        if self._current_file is not None:
            self._current_file.close()
            self._current_file = None

        known_outcomes = self._stats["success_known_episodes"]
        metadata = {
            "schema_version": self.schema_version,
            "total_episodes": self._stats["total_episodes"],
            "successful_episodes": self._stats["successful_episodes"],
            "success_unknown_episodes": self._stats["total_episodes"] - known_outcomes,
            "success_rate": self._stats["successful_episodes"] / known_outcomes if known_outcomes else None,
            "total_timesteps": self._stats["total_timesteps"],
            "num_tasks": len(self._stats["tasks"]),
            "save_vision_features": self.save_vision_features,
            "save_attention": self.save_attention,
            "episodes_per_file": self.episodes_per_file,
            "num_files": math.ceil(self._episode_count / self.episodes_per_file),
            "provenance": {"labels": self.labels_provenance},
            "source": self.source_metadata,
        }

        meta_path = self.output_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(
            f"Label generation finalized: {metadata['total_episodes']} episodes in {metadata['num_files']} files"
        )
        return metadata

    @property
    def summary(self) -> dict[str, Any]:
        return dict(self._stats)


class LabelReader:
    """Random-access reader for teacher labels (used in PRD-03 training)."""

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self._files: dict[int, h5py.File] = {}
        self.metadata = self._load_metadata()
        self.labels_provenance = labels_provenance_from_metadata(self.metadata)
        self._episode_index = self._build_index()

    def _load_metadata(self) -> dict[str, Any]:
        meta_path = self.data_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"No metadata.json in {self.data_dir}")
        with open(meta_path) as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError(f"Label metadata must contain a JSON object: {meta_path}")
        return payload

    def _build_index(self) -> list[tuple[int, int]]:
        """Build index mapping global episode idx → (file_idx, local_idx)."""
        index: list[tuple[int, int]] = []
        eps_per_file = self.metadata.get("episodes_per_file")
        total = self.metadata.get("total_episodes")
        if isinstance(eps_per_file, bool) or not isinstance(eps_per_file, int) or eps_per_file < 1:
            raise ValueError("Label metadata episodes_per_file must be a positive integer")
        if isinstance(total, bool) or not isinstance(total, int) or total < 0:
            raise ValueError("Label metadata total_episodes must be a non-negative integer")
        expected_files = math.ceil(total / eps_per_file)
        recorded_files = self.metadata.get("num_files")
        if recorded_files is not None and (
            isinstance(recorded_files, bool) or not isinstance(recorded_files, int) or recorded_files < 0
        ):
            raise ValueError("Label metadata num_files must be a non-negative integer")
        if recorded_files is not None and recorded_files != expected_files:
            raise ValueError(
                f"Label metadata num_files={recorded_files} does not match expected file count {expected_files}"
            )

        for i in range(total):
            file_idx = i // eps_per_file
            local_idx = i % eps_per_file
            index.append((file_idx, local_idx))

        for file_idx in range(expected_files):
            path = self.data_dir / f"episodes_{file_idx:04d}.h5"
            if not path.is_file():
                raise ValueError(f"Label metadata references missing episode file: {path}")
            first_episode = file_idx * eps_per_file
            episodes_in_file = min(eps_per_file, total - first_episode)
            try:
                with h5py.File(path, "r") as label_file:
                    missing = [
                        f"episode_{local_idx:04d}"
                        for local_idx in range(episodes_in_file)
                        if f"episode_{local_idx:04d}" not in label_file
                    ]
            except OSError as exc:
                raise ValueError(f"Label episode file is unreadable: {path}: {exc}") from exc
            if missing:
                raise ValueError(f"Label episode file {path} is missing groups: {missing}")

        return index

    def _get_file(self, file_idx: int) -> h5py.File:
        if file_idx not in self._files:
            path = self.data_dir / f"episodes_{file_idx:04d}.h5"
            self._files[file_idx] = h5py.File(path, "r")
        return self._files[file_idx]

    def __len__(self) -> int:
        return int(self.metadata["total_episodes"])

    def __getitem__(self, idx: int) -> dict:
        """Load a single episode by index."""
        file_idx, local_idx = self._episode_index[idx]
        f = self._get_file(file_idx)
        grp = f[f"episode_{local_idx:04d}"]
        success_is_known = bool(grp.attrs["success_known"]) if "success_known" in grp.attrs else "success" in grp.attrs
        success = bool(grp.attrs["success"]) if success_is_known and "success" in grp.attrs else None

        episode = {
            "episode_id": grp.attrs["episode_id"],
            "task_id": grp.attrs["task_id"],
            "language_instruction": grp.attrs["language_instruction"],
            "timesteps": grp.attrs["timesteps"],
            "success": success,
            "images": grp["observations/images"][:],
            "proprioception": grp["observations/proprioception"][:],
            "teacher_action_logits": grp["teacher_outputs/action_logits"][:],
            "teacher_action_mean": grp["teacher_outputs/action_mean"][:],
            "teacher_action_std": grp["teacher_outputs/action_std"][:],
            "confidence": grp["teacher_outputs/confidence"][:],
            "ground_truth_actions": grp["ground_truth_actions"][:],
        }

        if "vision_features" in grp["teacher_outputs"]:
            episode["teacher_vision_features"] = grp["teacher_outputs/vision_features"][:]

        return episode

    def close(self) -> None:
        files = getattr(self, "_files", {})
        for f in files.values():
            f.close()
        files.clear()

    def __del__(self):
        self.close()
