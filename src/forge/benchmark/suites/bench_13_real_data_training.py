"""Benchmark 13: Real Data Training — Train FORGE student on lerobot/pusht dataset.

Proves the full pipeline works with actual robot demonstration data,
not just random tensors. Uses video frames + action labels from PushT.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from forge.benchmark.artifacts import write_json_artifact
from forge.benchmark.suites.real_data import BENCHMARK_SEED, data_provenance, fixed_action_loss, reset_benchmark_rng
from forge.benchmark.suites.runtime import results_dir
from forge.data.lerobot_video_dataset import LeRobotVideoActionDataset
from forge.training_safety import backward_with_finite_gradients

RESULTS_DIR = results_dir()

MODEL_DIR = Path(os.environ.get("FORGE_MODEL_DIR", "./models"))
DATASET_DIR = Path(os.environ.get("FORGE_BENCHMARK_DATA_DIR", MODEL_DIR / "datasets" / "lerobot--pusht"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def reset_configuration_peak_memory(device: str) -> None:
    """Start an isolated peak-memory window on the selected CUDA device."""
    selected_device = torch.device(device)
    if selected_device.type != "cuda":
        return
    torch.cuda.synchronize(selected_device)
    torch.cuda.reset_peak_memory_stats(selected_device)


def configuration_peak_memory_gb(device: str) -> float:
    """Read the isolated peak after all selected-device CUDA work completes."""
    selected_device = torch.device(device)
    if selected_device.type != "cuda":
        return 0.0
    torch.cuda.synchronize(selected_device)
    return torch.cuda.max_memory_allocated(selected_device) / 1024**3


def training_metrics(
    losses: list[float],
    *,
    total_steps: int,
    train_time_s: float,
    evaluation_loss_before: float,
    evaluation_loss_after: float,
) -> dict[str, Any]:
    """Separate sample-comparable quality from raw optimization diagnostics."""
    values = np.asarray(losses, dtype=np.float64)
    if not losses or not np.isfinite(values).all():
        raise RuntimeError("Real-data training produced missing or non-finite losses")
    if not np.isfinite((evaluation_loss_before, evaluation_loss_after)).all() or evaluation_loss_before <= 0:
        raise RuntimeError("Fixed real-data evaluation produced invalid losses")
    n_10 = min(10, len(losses))
    return {
        "total_steps": total_steps,
        "train_time_s": train_time_s,
        "steps_per_sec": round(total_steps / train_time_s, 2),
        "loss_metric": "fixed-real-evaluation-mean",
        "evaluation_loss_before": round(evaluation_loss_before, 6),
        "evaluation_loss_after": round(evaluation_loss_after, 6),
        "loss_reduction_pct": round((1 - evaluation_loss_after / evaluation_loss_before) * 100, 1),
        "training_loss_start": round(losses[0], 4),
        "training_loss_end": round(losses[-1], 4),
        "training_loss_min": round(min(losses), 4),
        "training_loss_first_10_mean": round(float(np.mean(losses[:n_10])), 4),
        "training_loss_last_10_mean": round(float(np.mean(losses[-n_10:])), 4),
        "training_loss_curve_sampled": [round(losses[i], 4) for i in range(0, len(losses), max(1, len(losses) // 20))],
    }


def inference_action_mse(predicted: torch.Tensor, target: torch.Tensor) -> float:
    """Measure free-running actions against aligned real actions."""
    if predicted.dim() == 3:
        predicted = predicted[:, 0, :]
    if target.dim() == 3:
        target = target[:, 0, :]
    if predicted.dim() != 2 or target.dim() != 2 or predicted.shape[0] != target.shape[0]:
        raise ValueError(f"Cannot align predicted {tuple(predicted.shape)} and target {tuple(target.shape)} actions")
    width = min(predicted.shape[-1], target.shape[-1])
    if width < 1:
        raise ValueError("Action tensors must have at least one scalar dimension")
    predicted = predicted[:, :width]
    target = target[:, :width]
    if not torch.isfinite(predicted).all() or not torch.isfinite(target).all():
        raise RuntimeError("Real-data inference produced non-finite actions")
    return float(F.mse_loss(predicted, target).item())


def train_on_real_data(
    config_overrides: dict[str, Any],
    max_samples: int = 2000,
    train_steps: int = 200,
    batch_size: int = 8,
    lr: float = 2e-4,
) -> dict[str, Any]:
    """Train FORGE student on real PushT data."""
    from forge.config import ForgeConfig, apply_student_variant
    from forge.student import FORGEStudent

    print(f"\n{'=' * 60}")
    print("Real Data Training — PushT")
    print(f"{'=' * 60}")

    seed = reset_benchmark_rng()
    result: dict[str, Any] = {
        "dataset": "lerobot/pusht",
        "config": config_overrides,
        "random_seed": seed,
    }
    timings: dict[str, float] = {}
    reset_configuration_peak_memory(DEVICE)

    # Build model (action_dim=2 for PushT)
    config = ForgeConfig.default()
    apply_student_variant(config.student, str(config_overrides.get("variant", "nano")))
    config.student.action_dim = 2  # PushT is 2-DOF
    for key, val in config_overrides.items():
        setattr(config.student, key, val)

    t0 = time.perf_counter()
    student = FORGEStudent(config.student, model_dir=str(MODEL_DIR)).to(DEVICE)
    timings["build_s"] = round(time.perf_counter() - t0, 1)

    result["model"] = {
        "total_params_m": round(student.total_params / 1e6, 1),
        "trainable_params_m": round(student.trainable_params / 1e6, 1),
    }
    print(f"  Model: {result['model']['total_params_m']}M params, {result['model']['trainable_params_m']}M trainable")

    # Load dataset
    t0 = time.perf_counter()
    dataset = LeRobotVideoActionDataset(DATASET_DIR, max_samples=max_samples)
    provenance = data_provenance(dataset)
    if provenance.get("kind") != "real":
        raise RuntimeError("Real-data benchmark requires genuine LeRobot observations and actions")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    timings["data_load_s"] = round(time.perf_counter() - t0, 1)
    print(f"  Data loaded in {timings['data_load_s']}s")
    evaluation_loss_before = fixed_action_loss(student, dataset, DEVICE, action_dim=2)

    # Train
    student.train()
    optimizer = torch.optim.AdamW(student.trainable_parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_steps)

    losses = []
    step = 0

    t0 = time.perf_counter()
    for epoch in range(100):  # max epochs, break on step count
        for batch in loader:
            if step >= train_steps:
                break

            images = batch["image"].to(DEVICE)
            actions = batch["ground_truth_actions"].to(DEVICE)

            # Forward pass
            out = student(images, gt_actions=actions)
            loss = out["loss"]

            optimizer.zero_grad()
            backward_with_finite_gradients(loss, student.trainable_parameters())
            optimizer.step()
            scheduler.step()

            losses.append(loss.item())
            step += 1

            if step % 50 == 0 or step == 1:
                avg_loss = np.mean(losses[-10:])
                print(f"  Step {step}/{train_steps}: loss={avg_loss:.4f}, lr={scheduler.get_last_lr()[0]:.2e}")

        if step >= train_steps:
            break

    timings["train_s"] = round(time.perf_counter() - t0, 1)
    del optimizer, scheduler
    evaluation_loss_after = fixed_action_loss(student, dataset, DEVICE, action_dim=2)

    # Training analysis
    result["training"] = training_metrics(
        losses,
        total_steps=step,
        train_time_s=timings["train_s"],
        evaluation_loss_before=evaluation_loss_before,
        evaluation_loss_after=evaluation_loss_after,
    )

    print(
        f"\n  Training complete: {step} steps in {timings['train_s']}s ({result['training']['steps_per_sec']} steps/s)"
    )
    print(
        f"  Fixed real evaluation: {evaluation_loss_before:.4f} → {evaluation_loss_after:.4f} "
        f"({result['training']['loss_reduction_pct']}% reduction)"
    )

    # Inference benchmark on real data
    student.eval()
    inf_times = []
    inference_action_mses = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(DEVICE)
            actions = batch["ground_truth_actions"].to(DEVICE)
            # Warmup
            for _ in range(3):
                _ = student(images[:1].to(DEVICE))

            # Benchmark
            for i in range(min(20, len(images))):
                img = images[i : i + 1].to(DEVICE)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                out = student(img)
                torch.cuda.synchronize()
                inf_times.append((time.perf_counter() - t0) * 1000)
                inference_action_mses.append(inference_action_mse(out["actions"], actions[i : i + 1]))
            break  # Only need one batch

    if inf_times:
        arr = np.array(inf_times[3:])
        if not np.isfinite(arr).all():
            raise RuntimeError("Real-data inference produced non-finite latency")
        result["inference"] = {
            "fp32_p50_ms": round(float(np.percentile(arr, 50)), 2),
            "fp32_p95_ms": round(float(np.percentile(arr, 95)), 2),
            "fp32_fps": round(float(1000 / arr.mean()), 1),
            "gpu_mem_gb": round(configuration_peak_memory_gb(DEVICE), 2),
            "action_mse_mean": round(float(np.mean(inference_action_mses)), 6),
            "actions_finite": True,
        }
        print(f"  Inference: {arr.mean():.1f}ms p50, {1000 / arr.mean():.1f} FPS")

    result["timings"] = timings
    result["data_provenance"] = provenance

    del student
    torch.cuda.empty_cache()

    return result


def main():
    if DEVICE == "cpu":
        print("SKIP: No CUDA")
        sys.exit(0)

    if not DATASET_DIR.exists():
        print(f"SKIP: Dataset not found at {DATASET_DIR}")
        sys.exit(0)

    print("=== Real Data Training Benchmark ===")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Dataset: {DATASET_DIR}")

    configs = [
        # 1. Flow + LoRA64 (recommended balanced config)
        (
            "flow_lora64",
            {
                "variant": "nano",
                "language_model": "Qwen/Qwen3-0.6B",
                "lora_rank": 64,
                "action_head_type": "flow",
            },
        ),
        # 2. Diffusion + LoRA32 (baseline)
        (
            "diff_lora32",
            {
                "variant": "nano",
                "language_model": "Qwen/Qwen3-0.6B",
                "lora_rank": 32,
                "action_head_type": "diffusion",
            },
        ),
    ]

    all_results = {}
    for name, overrides in configs:
        try:
            all_results[name] = train_on_real_data(
                overrides,
                max_samples=2000,
                train_steps=200,
                batch_size=8,
                lr=2e-4,
            )
        except Exception as e:
            import traceback

            print(f"  ERROR: {e}")
            traceback.print_exc()
            all_results[name] = {"error": str(e)[:300]}
            torch.cuda.empty_cache()

    # Summary
    print(f"\n{'=' * 80}")
    print(f"{'Config':<20} {'Params':>8} {'Loss↓':>8} {'FPS':>8} {'Steps/s':>8} {'Time':>6}")
    print("-" * 80)
    for name, r in all_results.items():
        if "error" in r and isinstance(r["error"], str):
            print(f"{name:<20} ERROR: {r['error'][:50]}")
            continue
        t = r.get("training", {})
        inf = r.get("inference", {})
        m = r.get("model", {})
        print(
            f"{name:<20} {m.get('total_params_m', 0):>6.1f}M "
            f"{t.get('loss_reduction_pct', 0):>6.1f}% "
            f"{inf.get('fp32_fps', 0):>6.1f} "
            f"{t.get('steps_per_sec', 0):>6.2f} "
            f"{t.get('train_time_s', 0):>4.0f}s"
        )

    output = {
        "benchmark": "real_data_training",
        "timestamp": datetime.now(UTC).isoformat(),
        "device": DEVICE,
        "gpu": torch.cuda.get_device_name(0),
        "dataset": "lerobot/pusht",
        "random_seed": BENCHMARK_SEED,
        "data_provenance": next(
            (
                value["data_provenance"]
                for value in all_results.values()
                if isinstance(value, dict) and isinstance(value.get("data_provenance"), dict)
            ),
            {"kind": "missing"},
        ),
        "dataset_info": {
            "episodes": 206,
            "total_frames": 25650,
            "action_dim": 2,
            "image_size": "96x96 (resized to 384x384)",
        },
        "results": all_results,
    }

    out_path = RESULTS_DIR / "bench_13_real_data_training.json"
    write_json_artifact(out_path, output)

    print(f"\nResults saved to {out_path}")
    print("BENCH 13: DONE")


if __name__ == "__main__":
    main()
