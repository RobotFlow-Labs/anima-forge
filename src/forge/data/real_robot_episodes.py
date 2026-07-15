"""Read genuine robot demonstrations from local LeRobot datasets.

The reader supports both episode-per-parquet LeRobot v2.x datasets with
embedded images and packed LeRobot v3 datasets with MP4-backed observations.
It is deliberately fail-closed: malformed schemas, missing video/image data,
and non-finite state/action values are errors rather than reasons to substitute
generated data.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from PIL import Image

from forge.errors import ForgeDataNotFoundError


@dataclass(frozen=True)
class RealRobotEpisode:
    """One real demonstration episode, ready for teacher inference."""

    episode_id: str
    task_id: str
    instruction: str
    images: np.ndarray
    proprioception: np.ndarray
    ground_truth_actions: np.ndarray
    success: bool | None
    dataset_path: Path
    source_file: Path
    image_key: str

    @property
    def timesteps(self) -> int:
        return int(self.images.shape[0])


def load_real_robot_episodes(
    dataset_dir: str | Path,
    *,
    max_episodes: int | None = None,
    max_steps: int | None = None,
) -> list[RealRobotEpisode]:
    """Load deterministic episodes from an offline LeRobot dataset.

    Only local files are read.  ``max_episodes`` and ``max_steps`` always take
    the earliest sorted episodes/frames so verification runs are reproducible.
    """

    root = Path(dataset_dir).expanduser()
    if max_episodes is not None and max_episodes < 0:
        raise ValueError("max_episodes must be >= 0 or None")
    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps must be >= 1 or None")
    if not root.is_dir():
        raise ForgeDataNotFoundError(
            f"Real teacher dataset not found at {root}. Set FORGE_TEACHER_DATASET "
            "to a local LeRobot dataset with real image observations."
        )

    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise ForgeDataNotFoundError(f"LeRobot dataset at {root} is missing {info_path}")
    info = _read_json(info_path)
    data_template = str(info.get("data_path", ""))
    if "episode_{episode_index" not in data_template:
        if "{file_index" in data_template and str(info.get("video_path", "")):
            return _load_packed_video_episodes(
                root,
                info=info,
                max_episodes=max_episodes,
                max_steps=max_steps,
            )
        raise ValueError(f"Dataset {root} uses unsupported data layout {data_template!r}")

    files = sorted((root / "data").glob("chunk-*/episode_*.parquet"))
    if not files:
        raise ForgeDataNotFoundError(f"No episode parquet files found under {root / 'data'}")
    if max_episodes is not None:
        files = files[:max_episodes]

    task_map = _load_task_map(root)
    episode_map = _load_episode_map(root)
    return [
        _load_episode_file(
            source_file,
            dataset_root=root,
            task_map=task_map,
            episode_map=episode_map,
            max_steps=max_steps,
        )
        for source_file in files
    ]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read valid JSON from {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"Could not read {path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"Expected JSON object in {path}:{line_number}")
        records.append(record)
    return records


def _load_task_map(root: Path) -> dict[int, str]:
    jsonl_path = root / "meta" / "tasks.jsonl"
    parquet_path = root / "meta" / "tasks.parquet"
    if jsonl_path.is_file():
        rows = _read_jsonl(jsonl_path)
    elif parquet_path.is_file():
        rows = pq.read_table(parquet_path).to_pylist()
    else:
        raise ForgeDataNotFoundError(f"LeRobot dataset at {root} has no task metadata")

    task_map: dict[int, str] = {}
    for row in rows:
        task = row.get("task", row.get("__index_level_0__", ""))
        if "task_index" not in row or not str(task).strip():
            raise ValueError(f"Invalid task metadata row in {root}: {row}")
        task_map[int(row["task_index"])] = str(task).strip()
    if not task_map:
        raise ValueError(f"Task metadata is empty in {root}")
    return task_map


def _load_episode_map(root: Path) -> dict[int, dict[str, Any]]:
    path = root / "meta" / "episodes.jsonl"
    if not path.is_file():
        return {}
    result: dict[int, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        if "episode_index" in row:
            result[int(row["episode_index"])] = row
    return result


def _load_packed_video_episodes(
    root: Path,
    *,
    info: dict[str, Any],
    max_episodes: int | None,
    max_steps: int | None,
) -> list[RealRobotEpisode]:
    """Load LeRobot v3 packed parquet rows and their real MP4 observations."""

    episode_files = sorted((root / "meta" / "episodes").glob("chunk-*/*.parquet"))
    if not episode_files:
        raise ForgeDataNotFoundError(f"Packed LeRobot dataset at {root} has no episode metadata parquet files")
    episode_rows: list[dict[str, Any]] = []
    for episode_file in episode_files:
        try:
            episode_rows.extend(pq.read_table(episode_file).to_pylist())
        except Exception as exc:
            raise ValueError(f"Could not read episode metadata {episode_file}: {exc}") from exc
    episode_rows.sort(key=lambda row: int(row.get("episode_index", -1)))
    if not episode_rows:
        raise ValueError(f"Episode metadata is empty in {root}")
    if max_episodes is not None:
        episode_rows = episode_rows[:max_episodes]
    if not episode_rows:
        return []

    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError(f"Packed LeRobot dataset at {root} has no feature metadata")
    video_keys = sorted(
        str(key) for key, value in features.items() if isinstance(value, dict) and value.get("dtype") == "video"
    )
    if not video_keys:
        raise ValueError(f"Packed LeRobot dataset at {root} has no real video observation feature")

    task_map = _load_task_map(root)
    data_template = str(info["data_path"])
    video_template = str(info["video_path"])
    fps = float(info.get("fps", 0.0))
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"Packed LeRobot dataset at {root} has invalid fps {fps!r}")

    return [
        _load_packed_video_episode(
            root,
            episode_meta=row,
            task_map=task_map,
            data_template=data_template,
            video_template=video_template,
            video_key=video_keys[0],
            fps=fps,
            max_steps=max_steps,
        )
        for row in episode_rows
    ]


def _load_packed_video_episode(
    root: Path,
    *,
    episode_meta: dict[str, Any],
    task_map: dict[int, str],
    data_template: str,
    video_template: str,
    video_key: str,
    fps: float,
    max_steps: int | None,
) -> RealRobotEpisode:
    required_meta = {
        "episode_index",
        "data/chunk_index",
        "data/file_index",
        f"videos/{video_key}/chunk_index",
        f"videos/{video_key}/file_index",
        f"videos/{video_key}/from_timestamp",
    }
    missing_meta = sorted(required_meta - episode_meta.keys())
    if missing_meta:
        raise ValueError(f"Packed episode metadata in {root} is missing {missing_meta}")

    episode_index = int(episode_meta["episode_index"])
    source_file = root / data_template.format(
        chunk_index=int(episode_meta["data/chunk_index"]),
        file_index=int(episode_meta["data/file_index"]),
        episode_index=episode_index,
    )
    if not source_file.is_file():
        raise ForgeDataNotFoundError(f"Packed episode parquet not found: {source_file}")
    try:
        table = pq.read_table(source_file, filters=[("episode_index", "=", episode_index)])
    except Exception as exc:
        raise ValueError(f"Could not read packed episode {episode_index} from {source_file}: {exc}") from exc
    columns = set(table.column_names)
    required = {"action", "episode_index", "task_index"}
    missing = sorted(required - columns)
    if missing:
        raise ValueError(f"Invalid packed episode schema in {source_file}: missing {missing}")
    if table.num_rows < 1:
        raise ValueError(f"Packed episode {episode_index} has no rows in {source_file}")
    steps = min(table.num_rows, max_steps) if max_steps is not None else table.num_rows
    rows = table.slice(0, steps).to_pylist()

    task_indices = {int(row["task_index"]) for row in rows}
    if len(task_indices) != 1:
        raise ValueError(f"Packed episode {episode_index} mixes task indices in {source_file}")
    task_index = task_indices.pop()
    instruction = _resolve_instruction(
        episode_index,
        task_index,
        task_map,
        {episode_index: episode_meta},
        source_file,
    )

    video_path = root / video_template.format(
        video_key=video_key,
        chunk_index=int(episode_meta[f"videos/{video_key}/chunk_index"]),
        file_index=int(episode_meta[f"videos/{video_key}/file_index"]),
        episode_index=episode_index,
    )
    if not video_path.is_file():
        raise ForgeDataNotFoundError(f"Packed episode video not found: {video_path}")
    offset = float(episode_meta[f"videos/{video_key}/from_timestamp"])
    if "timestamp" in columns:
        timestamps = [offset + float(row["timestamp"]) for row in rows]
    elif "frame_index" in columns:
        timestamps = [offset + float(row["frame_index"]) / fps for row in rows]
    else:
        raise ValueError(f"Packed episode in {source_file} has no timestamp or frame_index")
    if not np.isfinite(timestamps).all():
        raise ValueError(f"Packed episode in {source_file} contains non-finite timestamps")

    try:
        from lerobot.datasets.video_utils import decode_video_frames  # type: ignore[import-untyped]

        frames = decode_video_frames(
            video_path,
            timestamps,
            tolerance_s=(1.0 / fps) + 1e-6,
            backend="pyav",
            return_uint8=True,
        )
        images = frames.permute(0, 2, 3, 1).cpu().numpy()
    except Exception as exc:
        raise ValueError(f"Could not decode real video frames from {video_path}: {exc}") from exc
    if images.shape[0] != steps or images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"Decoded video {video_path} has invalid shape {images.shape}")

    actions = _finite_matrix([row["action"] for row in rows], name="action", source_file=source_file)
    if "observation.state" in columns:
        state = _finite_matrix(
            [row["observation.state"] for row in rows],
            name="observation.state",
            source_file=source_file,
        )
    else:
        state = np.empty((steps, 0), dtype=np.float32)
    success = _resolve_success(episode_meta, table)

    return RealRobotEpisode(
        episode_id=f"episode_{episode_index:06d}",
        task_id=f"task_{task_index}",
        instruction=instruction,
        images=np.ascontiguousarray(images, dtype=np.uint8),
        proprioception=state,
        ground_truth_actions=actions,
        success=success,
        dataset_path=root.resolve(),
        source_file=source_file.resolve(),
        image_key=video_key,
    )


def _load_episode_file(
    source_file: Path,
    *,
    dataset_root: Path,
    task_map: dict[int, str],
    episode_map: dict[int, dict[str, Any]],
    max_steps: int | None,
) -> RealRobotEpisode:
    try:
        table = pq.read_table(source_file)
    except Exception as exc:
        raise ValueError(f"Could not read episode parquet {source_file}: {exc}") from exc
    columns = set(table.column_names)
    image_keys = sorted(name for name in columns if name.startswith("observation.images."))
    required = {"action", "episode_index", "task_index"}
    missing = sorted(required - columns)
    if missing or not image_keys:
        detail = f"missing {missing}" if missing else "missing embedded observation.images.* column"
        raise ValueError(f"Invalid real episode schema in {source_file}: {detail}")

    row_count = table.num_rows
    if row_count < 1:
        raise ValueError(f"Real episode parquet is empty: {source_file}")
    steps = min(row_count, max_steps) if max_steps is not None else row_count
    rows = table.slice(0, steps).to_pylist()

    episode_indices = {int(row["episode_index"]) for row in rows}
    task_indices = {int(row["task_index"]) for row in rows}
    if len(episode_indices) != 1 or len(task_indices) != 1:
        raise ValueError(f"Episode file {source_file} mixes episode/task indices")
    episode_index = episode_indices.pop()
    task_index = task_indices.pop()

    instruction = _resolve_instruction(episode_index, task_index, task_map, episode_map, source_file)
    image_key = image_keys[0]
    images = np.stack(
        [_decode_image(row[image_key], dataset_root=dataset_root, source_file=source_file) for row in rows]
    )
    actions = _finite_matrix([row["action"] for row in rows], name="action", source_file=source_file)

    if "observation.state" in columns:
        state = _finite_matrix(
            [row["observation.state"] for row in rows],
            name="observation.state",
            source_file=source_file,
        )
    else:
        state = np.empty((steps, 0), dtype=np.float32)

    episode_meta = episode_map.get(episode_index, {})
    success = _resolve_success(episode_meta, table)

    return RealRobotEpisode(
        episode_id=f"episode_{episode_index:06d}",
        task_id=f"task_{task_index}",
        instruction=instruction,
        images=np.ascontiguousarray(images, dtype=np.uint8),
        proprioception=state,
        ground_truth_actions=actions,
        success=success,
        dataset_path=dataset_root.resolve(),
        source_file=source_file.resolve(),
        image_key=image_key,
    )


def _resolve_success(episode_meta: dict[str, Any], table: Any) -> bool | None:
    """Resolve an explicitly recorded outcome without inventing one."""
    metadata_value = episode_meta.get("success")
    if metadata_value is not None:
        return bool(metadata_value)
    if "next.success" not in table.column_names:
        return None
    recorded = [value for value in table["next.success"].to_pylist() if value is not None]
    return any(bool(value) for value in recorded) if recorded else None


def _resolve_instruction(
    episode_index: int,
    task_index: int,
    task_map: dict[int, str],
    episode_map: dict[int, dict[str, Any]],
    source_file: Path,
) -> str:
    episode_tasks = episode_map.get(episode_index, {}).get("tasks")
    if isinstance(episode_tasks, list) and episode_tasks and str(episode_tasks[0]).strip():
        return str(episode_tasks[0]).strip()
    instruction = task_map.get(task_index, "").strip()
    if not instruction:
        raise ValueError(f"No instruction for task {task_index} in {source_file}")
    return instruction


def _decode_image(value: object, *, dataset_root: Path, source_file: Path) -> np.ndarray:
    raw: bytes | None = None
    image_path: Path | None = None
    if isinstance(value, dict):
        candidate = value.get("bytes")
        if isinstance(candidate, (bytes, bytearray, memoryview)):
            raw = bytes(candidate)
        path_value = value.get("path")
        if path_value:
            image_path = Path(str(path_value))
    elif isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
    elif isinstance(value, str):
        image_path = Path(value)

    try:
        if raw is not None:
            with Image.open(io.BytesIO(raw)) as image:
                array = np.asarray(image.convert("RGB"), dtype=np.uint8)
        elif image_path is not None:
            if not image_path.is_absolute():
                root_candidate = dataset_root / image_path
                image_path = root_candidate if root_candidate.exists() else source_file.parent / image_path
            with Image.open(image_path) as image:
                array = np.asarray(image.convert("RGB"), dtype=np.uint8)
        else:
            raise ValueError("image record has neither bytes nor path")
    except Exception as exc:
        raise ValueError(f"Could not decode real image from {source_file}: {exc}") from exc
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Decoded image from {source_file} has invalid shape {array.shape}")
    return array


def _finite_matrix(values: list[object], *, name: str, source_file: Path) -> np.ndarray:
    try:
        matrix = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {name} values in {source_file}: {exc}") from exc
    if matrix.ndim != 2 or matrix.shape[1] < 1:
        raise ValueError(f"{name} in {source_file} must be a non-empty 2-D matrix, got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} in {source_file} contains non-finite values")
    return np.ascontiguousarray(matrix)


__all__ = ["RealRobotEpisode", "load_real_robot_episodes"]
