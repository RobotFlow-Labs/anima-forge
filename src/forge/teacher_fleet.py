"""Real-frame, real-weight verification for the registered teacher fleet."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch

from forge.json_utils import json_ready
from forge.provenance import current_git_sha
from forge.teachers.base import TeacherAdapter

TEACHER_MODEL_DIRS = {
    "openvla-7b": "openvla--openvla-7b",
    "rdt2-fm": "robotics-diffusion-transformer--RDT2-FM",
    "smolvla-base": "lerobot--smolvla_base",
    "molmoact2-libero": "allenai--MolmoAct2-LIBERO-LeRobot",
    "vla-jepa-3b": "lerobot--VLA-JEPA-Pretrain",
}

TEACHER_COMPANION_PATHS = {
    "rdt2-fm": (
        "robotics-diffusion-transformer--RDT2-VQ",
        "rdt2-umi-normalizer.pt",
    ),
    "smolvla-base": ("HuggingFaceTB--SmolVLM2-500M-Video-Instruct",),
    "vla-jepa-3b": (
        "Qwen--Qwen3-VL-2B-Instruct",
        "facebook--vjepa2-vitl-fpc64-256",
    ),
    "molmoact2-libero": (
        "allenai--MolmoAct2-LIBERO",
        "allenai--MolmoAct2-FAST-Tokenizer",
    ),
}


@dataclass(frozen=True)
class VerificationFrame:
    """One decoded real robot observation used for fleet inference."""

    image: np.ndarray
    instruction: str
    proprioception: np.ndarray
    dataset: str
    frame_index: int


def load_fleet_verification_frames(dataset_root: str | Path, count: int = 5) -> list[VerificationFrame]:
    """Load real frames from both PushT and ALOHA LeRobot video datasets."""
    if count < 2:
        raise ValueError("Fleet verification requires at least two real frames")
    root = Path(dataset_root).expanduser()
    specifications = [
        (root / "lerobot--pusht", (count + 1) // 2),
        (root / "lerobot--aloha_sim_transfer_cube_human", count // 2),
    ]
    frames: list[VerificationFrame] = []
    for dataset_dir, source_count in specifications:
        frames.extend(_load_video_dataset_frames(dataset_dir, source_count))
    if len(frames) != count:
        raise ValueError(f"Expected {count} real verification frames, decoded {len(frames)}")
    return frames


def _load_video_dataset_frames(dataset_dir: Path, count: int) -> list[VerificationFrame]:
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Real verification dataset not found: {dataset_dir}")
    from forge.data.real_robot_episodes import load_real_robot_episodes

    # Read images and state through the episode-aware path so packed parquet
    # rows, video file indices, and video timestamp offsets stay aligned.
    episodes = load_real_robot_episodes(dataset_dir, max_episodes=count, max_steps=count)
    frames: list[VerificationFrame] = []
    for episode in episodes:
        for frame_index in range(episode.timesteps):
            frames.append(
                VerificationFrame(
                    image=np.ascontiguousarray(episode.images[frame_index]),
                    instruction=episode.instruction,
                    proprioception=np.asarray(episode.proprioception[frame_index], dtype=np.float32),
                    dataset=dataset_dir.name,
                    frame_index=frame_index,
                )
            )
            if len(frames) == count:
                return frames
    raise ValueError(f"Dataset {dataset_dir} has fewer than {count} aligned real observations")


def verify_teacher(
    name: str,
    adapter: TeacherAdapter,
    model_path: str | Path,
    frames: list[VerificationFrame],
    *,
    device: str,
    dtype: torch.dtype = torch.bfloat16,
    companion_paths: tuple[str | Path, ...] = (),
    include_predictions: bool = False,
) -> dict[str, Any]:
    """Load one teacher and record real inference latency, memory, and actions."""
    path = Path(model_path)
    if not path.is_dir():
        raise FileNotFoundError(f"Teacher {name} checkpoint not found: {path}")
    if not frames:
        raise ValueError("At least one real frame is required")

    cuda = device.startswith("cuda")
    if cuda:
        cuda_device = torch.device(device)
        torch.cuda.set_device(cuda_device)
        torch.empty(0, device=cuda_device)
        torch.cuda.reset_peak_memory_stats(cuda_device)
        torch.cuda.synchronize(cuda_device)
    load_started = time.perf_counter()
    adapter.load(path, device=device, dtype=dtype)
    if cuda:
        torch.cuda.synchronize(device)
    load_seconds = time.perf_counter() - load_started

    latencies_ms: list[float] = []
    action_arrays: list[np.ndarray] = []
    confidence_arrays: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    try:
        info = adapter.info()
        for frame in frames:
            if cuda:
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            chunk = adapter.predict(frame.image, frame.instruction, frame.proprioception)
            if cuda:
                torch.cuda.synchronize(device)
            latencies_ms.append((time.perf_counter() - started) * 1000)
            actions = np.asarray(chunk.actions, dtype=np.float32)
            if actions.shape != (info.action_horizon, info.action_dim):
                raise ValueError(
                    f"Teacher {name} returned {actions.shape}, expected {(info.action_horizon, info.action_dim)}"
                )
            if not np.isfinite(actions).all():
                raise ValueError(f"Teacher {name} returned non-finite actions")
            if chunk.metadata.get("inference") != "real":
                raise ValueError(f"Teacher {name} did not attest real inference in ActionChunk metadata")
            action_arrays.append(actions)
            action_mean = np.asarray(chunk.action_mean, dtype=np.float32)
            action_std = np.asarray(chunk.action_std, dtype=np.float32)
            confidence = np.asarray(chunk.confidence, dtype=np.float32)
            for statistic_name, statistic in (("action mean", action_mean), ("action std", action_std)):
                if statistic.shape != actions.shape or not np.isfinite(statistic).all():
                    raise ValueError(f"Teacher {name} returned invalid {statistic_name} shape or values")
            if np.any(action_std < 0):
                raise ValueError(f"Teacher {name} returned negative action std")
            if (
                confidence.shape != actions.shape
                or not np.isfinite(confidence).all()
                or np.any((confidence < 0) | (confidence > 1))
            ):
                raise ValueError(f"Teacher {name} returned invalid confidence shape or values")
            confidence_arrays.append(confidence)
            metadata.append(dict(chunk.metadata))
    finally:
        adapter.unload()

    combined = np.stack(action_arrays)
    peak_allocated = torch.cuda.max_memory_allocated(device) if cuda else 0
    peak_reserved = torch.cuda.max_memory_reserved(device) if cuda else 0
    artifact_paths = [path, *(Path(item) for item in companion_paths)]
    for artifact in artifact_paths:
        if not artifact.exists():
            raise FileNotFoundError(f"Teacher {name} companion artifact not found: {artifact}")

    def artifact_bytes(artifact: Path) -> int:
        if artifact.is_file():
            return artifact.stat().st_size
        return sum(file.stat().st_size for file in artifact.rglob("*") if file.is_file())

    weight_bytes = sum(artifact_bytes(artifact) for artifact in artifact_paths)
    record = {
        "status": "ok",
        "teacher": name,
        "model_path": str(path.resolve()),
        "artifact_paths": [str(artifact.resolve()) for artifact in artifact_paths],
        "model_bytes": weight_bytes,
        "device": device,
        "gpu_name": torch.cuda.get_device_name(device) if cuda else None,
        "load_seconds": load_seconds,
        "predictions": len(frames),
        "latency_ms": {
            "values": latencies_ms,
            "mean": mean(latencies_ms),
            "min": min(latencies_ms),
            "max": max(latencies_ms),
        },
        "cuda_memory_bytes": {
            "peak_allocated": peak_allocated,
            "peak_reserved": peak_reserved,
        },
        "actions": {
            "shape_per_prediction": [info.action_horizon, info.action_dim],
            "finite": True,
            "min": float(combined.min()),
            "max": float(combined.max()),
            "mean": float(combined.mean()),
            "std": float(combined.std()),
        },
        "frame_sources": [asdict(frame) | {"image": list(frame.image.shape)} for frame in frames],
        "prediction_metadata": metadata,
    }
    if include_predictions:
        record["prediction_actions"] = [array.tolist() for array in action_arrays]
        record["prediction_confidences"] = [array.tolist() for array in confidence_arrays]
    return record


def build_fleet_report(
    *,
    teacher_names: list[str],
    model_dir: str | Path,
    dataset_dir: str | Path,
    gpu_ids: list[int],
    predictions: int = 5,
    include_predictions: bool = False,
) -> dict[str, Any]:
    """Verify every requested registry teacher, continuing to record failures."""
    from forge.teachers.registry import get_registry

    if not teacher_names:
        raise ValueError("At least one teacher name is required")
    if not gpu_ids:
        raise ValueError("At least one GPU id is required")
    frames = load_fleet_verification_frames(dataset_dir, count=predictions)
    registry = get_registry()
    records: list[dict[str, Any]] = []
    for index, name in enumerate(teacher_names):
        device = f"cuda:{gpu_ids[index % len(gpu_ids)]}"
        directory = TEACHER_MODEL_DIRS.get(name)
        if directory is None:
            records.append({"status": "error", "teacher": name, "device": device, "error": "no model mapping"})
            continue
        adapter = registry.create(name)
        companions = tuple(Path(model_dir) / item for item in TEACHER_COMPANION_PATHS.get(name, ()))
        try:
            records.append(
                verify_teacher(
                    name,
                    adapter,
                    Path(model_dir) / directory,
                    frames,
                    device=device,
                    companion_paths=companions,
                    include_predictions=include_predictions,
                )
            )
        except Exception as exc:
            records.append({"status": "error", "teacher": name, "device": device, "error": str(exc)})
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    failures = [record for record in records if record["status"] != "ok"]
    return {
        "schema_version": "1.0",
        "git_sha": current_git_sha(),
        "model_dir": str(Path(model_dir).resolve()),
        "dataset_dir": str(Path(dataset_dir).resolve()),
        "predictions_per_teacher": predictions,
        "teachers_requested": len(teacher_names),
        "teachers_verified": len(records) - len(failures),
        "all_real": not failures,
        "results": records,
    }


def combine_fleet_reports(
    reports: list[dict[str, Any]],
    *,
    model_dir: str | Path,
    dataset_dir: str | Path,
    predictions: int,
) -> dict[str, Any]:
    """Combine process-isolated teacher reports without weakening acceptance."""
    records: list[dict[str, Any]] = []
    for index, report in enumerate(reports):
        result = report.get("results")
        if not isinstance(result, list) or len(result) != 1 or not isinstance(result[0], dict):
            raise ValueError(f"Isolated teacher report {index} must contain exactly one result")
        records.append(result[0])
    failures = [record for record in records if record.get("status") != "ok"]
    return {
        "schema_version": "1.0",
        "git_sha": current_git_sha(),
        "model_dir": str(Path(model_dir).resolve()),
        "dataset_dir": str(Path(dataset_dir).resolve()),
        "predictions_per_teacher": predictions,
        "teachers_requested": len(records),
        "teachers_verified": len(records) - len(failures),
        "all_real": bool(records) and not failures,
        "execution_isolation": "one-process-per-teacher",
        "results": records,
    }


def build_isolated_fleet_report(
    *,
    teacher_names: list[str],
    model_dir: str | Path,
    dataset_dir: str | Path,
    gpu_ids: list[int],
    predictions: int = 5,
    include_predictions: bool = False,
    timeout_seconds: float = 1_800,
) -> dict[str, Any]:
    """Verify each heavyweight teacher in a fresh Python/CUDA process."""
    if not gpu_ids:
        raise ValueError("At least one GPU id is required")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    reports: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="forge-teacher-fleet-") as directory:
        for index, teacher_name in enumerate(teacher_names):
            gpu_id = gpu_ids[index % len(gpu_ids)]
            report_path = Path(directory) / f"teacher-{index}.json"
            command = [
                sys.executable,
                "-m",
                "forge.teacher_fleet",
                "--teacher",
                teacher_name,
                "--gpu",
                str(gpu_id),
                "--model-dir",
                str(model_dir),
                "--dataset-dir",
                str(dataset_dir),
                "--predictions",
                str(predictions),
                "--output",
                str(report_path),
            ]
            if include_predictions:
                command.append("--include-predictions")
            try:
                completed = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                )
            except subprocess.TimeoutExpired:
                reports.append(
                    {
                        "results": [
                            {
                                "status": "error",
                                "teacher": teacher_name,
                                "device": f"cuda:{gpu_id}",
                                "error": f"isolated verifier timed out after {timeout_seconds:g}s",
                            }
                        ]
                    }
                )
                continue
            if not report_path.is_file():
                detail = completed.stderr.strip() or completed.stdout.strip() or f"exit {completed.returncode}"
                reports.append(
                    {
                        "results": [
                            {
                                "status": "error",
                                "teacher": teacher_name,
                                "device": f"cuda:{gpu_id}",
                                "error": f"isolated verifier produced no report: {detail}",
                            }
                        ]
                    }
                )
                continue
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                reports.append(
                    {
                        "results": [
                            {
                                "status": "error",
                                "teacher": teacher_name,
                                "device": f"cuda:{gpu_id}",
                                "error": f"isolated verifier produced invalid JSON: {exc}",
                            }
                        ]
                    }
                )
            else:
                if not isinstance(report, dict):
                    raise ValueError("Isolated teacher report must be a JSON object")
                reports.append(report)
    return combine_fleet_reports(
        reports,
        model_dir=model_dir,
        dataset_dir=dataset_dir,
        predictions=predictions,
    )


def write_fleet_report(report: dict[str, Any], output: str | Path) -> Path:
    """Atomically write the JSON fleet report."""
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_ready(report), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def _worker_main(argv: list[str] | None = None) -> int:
    """Internal process entrypoint used by ``build_isolated_fleet_report``."""
    parser = argparse.ArgumentParser(description="Internal FORGE teacher-fleet worker")
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--predictions", required=True, type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--include-predictions", action="store_true")
    args = parser.parse_args(argv)
    report = build_fleet_report(
        teacher_names=[args.teacher],
        model_dir=args.model_dir,
        dataset_dir=args.dataset_dir,
        gpu_ids=[args.gpu],
        predictions=args.predictions,
        include_predictions=args.include_predictions,
    )
    write_fleet_report(report, args.output)
    return 0 if report["all_real"] else 2


__all__ = [
    "TEACHER_MODEL_DIRS",
    "TEACHER_COMPANION_PATHS",
    "VerificationFrame",
    "build_fleet_report",
    "build_isolated_fleet_report",
    "combine_fleet_reports",
    "load_fleet_verification_frames",
    "verify_teacher",
    "write_fleet_report",
]


if __name__ == "__main__":
    raise SystemExit(_worker_main())
