"""Truthfulness tests for local real-robot episode ingestion."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from PIL import Image

from forge.data.real_robot_episodes import load_real_robot_episodes
from forge.errors import ForgeDataNotFoundError


def _png_bytes(value: int) -> bytes:
    image = np.full((8, 10, 3), value, dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return buffer.getvalue()


def _write_dataset(
    root: Path,
    *,
    actions: list[list[float]] | None = None,
    include_state: bool = True,
) -> Path:
    (root / "meta").mkdir(parents=True)
    data_dir = root / "data" / "chunk-000"
    data_dir.mkdir(parents=True)
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v2.1",
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            }
        ),
        encoding="utf-8",
    )
    (root / "meta" / "tasks.jsonl").write_text(
        json.dumps({"task_index": 4, "task": "move the blue block"}) + "\n",
        encoding="utf-8",
    )
    (root / "meta" / "episodes.jsonl").write_text(
        json.dumps({"episode_index": 7, "tasks": ["move the blue block carefully"], "length": 3}) + "\n",
        encoding="utf-8",
    )
    action_rows = actions or [[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]]
    rows = []
    for index, action in enumerate(action_rows):
        row = {
            "observation.images.main": {"bytes": _png_bytes(20 + index), "path": f"frame_{index}.png"},
            "action": action,
            "episode_index": 7,
            "frame_index": index,
            "task_index": 4,
        }
        if include_state:
            row["observation.state"] = [float(index), float(index + 1), float(index + 2)]
        rows.append(row)
    path = data_dir / "episode_000007.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def test_loads_embedded_real_episode_deterministically(tmp_path: Path) -> None:
    source = _write_dataset(tmp_path)

    episodes = load_real_robot_episodes(tmp_path, max_episodes=1, max_steps=2)

    assert len(episodes) == 1
    episode = episodes[0]
    assert episode.episode_id == "episode_000007"
    assert episode.task_id == "task_4"
    assert episode.instruction == "move the blue block carefully"
    assert episode.timesteps == 2
    assert episode.images.shape == (2, 8, 10, 3)
    assert episode.images.dtype == np.uint8
    assert episode.images[0, 0, 0].tolist() == [20, 20, 20]
    assert episode.proprioception.shape == (2, 3)
    np.testing.assert_allclose(episode.ground_truth_actions, [[0.1, 0.2], [0.2, 0.3]])
    assert episode.success is None
    assert episode.source_file == source.resolve()
    assert episode.dataset_path == tmp_path.resolve()
    assert episode.image_key == "observation.images.main"


def test_missing_optional_state_is_explicit_empty_matrix(tmp_path: Path) -> None:
    _write_dataset(tmp_path, include_state=False)

    episode = load_real_robot_episodes(tmp_path, max_steps=1)[0]

    assert episode.proprioception.shape == (1, 0)
    assert episode.ground_truth_actions.shape == (1, 2)


def test_missing_dataset_fails_with_actionable_path(tmp_path: Path) -> None:
    missing = tmp_path / "missing"

    with pytest.raises(ForgeDataNotFoundError, match="FORGE_TEACHER_DATASET") as exc_info:
        load_real_robot_episodes(missing)

    assert str(missing) in str(exc_info.value)
    assert not missing.exists()


def test_non_finite_actions_fail_instead_of_generating_fallback(tmp_path: Path) -> None:
    _write_dataset(tmp_path, actions=[[0.1, 0.2], [float("nan"), 0.3]])

    with pytest.raises(ValueError, match="contains non-finite"):
        load_real_robot_episodes(tmp_path)


def test_loads_packed_video_episode_with_real_decoder_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "meta").mkdir(parents=True)
    (tmp_path / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (tmp_path / "data" / "chunk-000").mkdir(parents=True)
    video_dir = tmp_path / "videos" / "observation.images.top" / "chunk-000"
    video_dir.mkdir(parents=True)
    (tmp_path / "meta" / "info.json").write_text(
        json.dumps(
            {
                "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
                "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
                "fps": 50,
                "features": {"observation.images.top": {"dtype": "video"}},
            }
        ),
        encoding="utf-8",
    )
    pq.write_table(
        pa.Table.from_pylist([{"task_index": 4, "__index_level_0__": "move the blue block"}]),
        tmp_path / "meta" / "tasks.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "episode_index": 7,
                    "data/chunk_index": 0,
                    "data/file_index": 0,
                    "videos/observation.images.top/chunk_index": 0,
                    "videos/observation.images.top/file_index": 0,
                    "videos/observation.images.top/from_timestamp": 2.0,
                    "tasks": ["move the blue block carefully"],
                }
            ]
        ),
        tmp_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
    )
    source = tmp_path / "data" / "chunk-000" / "file-000.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "episode_index": 7,
                    "task_index": 4,
                    "frame_index": index,
                    "timestamp": index / 50,
                    "observation.state": [float(index), float(index + 1)],
                    "action": [0.1 + index, 0.2 + index],
                }
                for index in range(3)
            ]
        ),
        source,
    )
    video_path = video_dir / "file-000.mp4"
    video_path.touch()
    decoder_calls: list[tuple[Path, list[float], float, str | None, bool]] = []

    def fake_decode(
        path: Path,
        timestamps: list[float],
        tolerance_s: float,
        backend: str | None = None,
        return_uint8: bool = False,
    ) -> torch.Tensor:
        decoder_calls.append((Path(path), timestamps, tolerance_s, backend, return_uint8))
        return torch.full((len(timestamps), 3, 8, 10), 23, dtype=torch.uint8)

    monkeypatch.setattr("lerobot.datasets.video_utils.decode_video_frames", fake_decode)

    episode = load_real_robot_episodes(tmp_path, max_episodes=1, max_steps=2)[0]

    assert episode.episode_id == "episode_000007"
    assert episode.instruction == "move the blue block carefully"
    assert episode.images.shape == (2, 8, 10, 3)
    assert episode.images[0, 0, 0].tolist() == [23, 23, 23]
    assert episode.source_file == source.resolve()
    assert episode.image_key == "observation.images.top"
    assert episode.success is None
    np.testing.assert_allclose(episode.ground_truth_actions, [[0.1, 0.2], [1.1, 1.2]])
    assert decoder_calls == [(video_path, [2.0, 2.02], pytest.approx(0.020001), "pyav", True)]


def test_explicit_episode_success_is_preserved(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    (tmp_path / "meta" / "episodes.jsonl").write_text(
        json.dumps(
            {
                "episode_index": 7,
                "tasks": ["move the blue block carefully"],
                "length": 3,
                "success": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert load_real_robot_episodes(tmp_path)[0].success is False


def test_success_uses_full_episode_when_steps_are_truncated(tmp_path: Path) -> None:
    source = _write_dataset(tmp_path)
    table = pq.read_table(source)
    table = table.append_column("next.success", pa.array([False, False, True]))
    pq.write_table(table, source)

    episode = load_real_robot_episodes(tmp_path, max_steps=1)[0]

    assert episode.timesteps == 1
    assert episode.success is True


def test_unknown_data_layout_is_rejected_explicitly(tmp_path: Path) -> None:
    (tmp_path / "meta").mkdir(parents=True)
    (tmp_path / "meta" / "info.json").write_text(
        json.dumps({"data_path": "data/unknown.parquet"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported data layout"):
        load_real_robot_episodes(tmp_path)


@pytest.mark.parametrize("max_episodes", [-1])
def test_invalid_episode_limit_rejected_before_io(tmp_path: Path, max_episodes: int) -> None:
    with pytest.raises(ValueError, match="max_episodes"):
        load_real_robot_episodes(tmp_path / "missing", max_episodes=max_episodes)


def test_zero_episode_limit_validates_dataset_but_reads_no_parquet(tmp_path: Path) -> None:
    _write_dataset(tmp_path)

    assert load_real_robot_episodes(tmp_path, max_episodes=0) == []
