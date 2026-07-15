"""Benchmark 03: KD Training Loop — loss convergence, speed, memory."""

from __future__ import annotations

import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from forge.benchmark.artifacts import write_json_artifact
from forge.benchmark.suites.real_data import (
    BENCHMARK_SEED,
    data_provenance,
    fixed_action_loss,
    load_real_dataset,
    real_batch,
    reset_benchmark_rng,
)
from forge.benchmark.suites.runtime import results_dir
from forge.training_safety import backward_with_finite_gradients

RESULTS_DIR = results_dir()

MODEL_DIR = Path(os.environ.get("FORGE_MODEL_DIR", "./models"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run_training(student, dataset, n_steps: int, lr: float, label: str):
    """Run training for n_steps and return metrics."""
    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.01,
    )
    student.train()
    torch.cuda.reset_peak_memory_stats()

    losses = []
    step_times = []
    evaluation_loss_before = fixed_action_loss(student, dataset, DEVICE)

    for step in range(n_steps):
        img, gt = real_batch(dataset, 1, DEVICE, start=step)

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        optimizer.zero_grad()
        out = student(img, gt_actions=gt)
        loss = out["loss"]
        backward_with_finite_gradients(loss, student.parameters())
        optimizer.step()

        torch.cuda.synchronize()
        step_times.append((time.perf_counter() - t0) * 1000)
        losses.append(loss.item())

        if step % 10 == 0:
            print(f"  [{label}] Step {step}: loss={losses[-1]:.4f}")

    evaluation_loss_after = fixed_action_loss(student, dataset, DEVICE)
    gpu_mem = torch.cuda.max_memory_allocated() / 1e9
    return losses, step_times, gpu_mem, evaluation_loss_before, evaluation_loss_after


def main():
    if DEVICE != "cuda":
        print("SKIP: No CUDA device")
        sys.exit(0)

    from forge.config import ForgeConfig
    from forge.student import FORGEStudent

    dataset = load_real_dataset(MODEL_DIR, max_samples=100)
    config = ForgeConfig.default()
    config.student.action_dim = dataset.action_dim

    print("Building FORGEStudent...")
    reset_benchmark_rng()
    student = FORGEStudent(config.student, model_dir=str(MODEL_DIR)).to(DEVICE)

    # Run 1: 50 steps, lr=2e-4
    print("\n=== Training Run 1: 50 steps, lr=2e-4 ===")
    losses_1, times_1, mem_1, eval_before_1, eval_after_1 = run_training(student, dataset, 50, 2e-4, "run1")

    # Reset model for run 2
    reset_benchmark_rng()
    student2 = FORGEStudent(config.student, model_dir=str(MODEL_DIR)).to(DEVICE)

    # Run 2: 50 steps, lr=5e-4
    print("\n=== Training Run 2: 50 steps, lr=5e-4 ===")
    losses_2, times_2, mem_2, eval_before_2, eval_after_2 = run_training(student2, dataset, 50, 5e-4, "run2")

    # Run 3: 100 steps, lr=2e-4 (longer convergence)
    reset_benchmark_rng()
    student3 = FORGEStudent(config.student, model_dir=str(MODEL_DIR)).to(DEVICE)

    print("\n=== Training Run 3: 100 steps, lr=2e-4 ===")
    losses_3, times_3, mem_3, eval_before_3, eval_after_3 = run_training(student3, dataset, 100, 2e-4, "run3")

    def training_stats(losses, times, gpu_mem, evaluation_loss_before, evaluation_loss_after):
        la = np.array(losses)
        ta = np.array(times)
        evaluation_loss_reduction_pct = (
            (evaluation_loss_before - evaluation_loss_after) / max(evaluation_loss_before, 1e-12) * 100
        )
        return {
            "n_steps": len(losses),
            "loss_metric": "fixed-real-evaluation-mean",
            "evaluation_samples": 5,
            "evaluation_loss_before": round(float(evaluation_loss_before), 4),
            "evaluation_loss_after": round(float(evaluation_loss_after), 4),
            "training_loss_min": round(float(la.min()), 4),
            "loss_reduction_pct": round(float(evaluation_loss_reduction_pct), 1),
            "training_loss_first": round(float(la[0]), 4),
            "training_loss_last": round(float(la[-1]), 4),
            "training_loss_first_10_mean": round(float(la[:10].mean()), 4),
            "training_loss_last_10_mean": round(float(la[-10:].mean()), 4),
            "step_time_mean_ms": round(float(ta.mean()), 1),
            "step_time_std_ms": round(float(ta.std()), 1),
            "step_time_p50_ms": round(float(np.percentile(ta, 50)), 1),
            "step_time_p95_ms": round(float(np.percentile(ta, 95)), 1),
            "steps_per_sec": round(float(1000 / ta.mean()), 2),
            "total_time_s": round(float(ta.sum() / 1000), 1),
            "gpu_mem_peak_gb": round(gpu_mem, 2),
            "training_loss_curve": [round(float(x), 4) for x in la.tolist()],
        }

    results = {
        "benchmark": "kd_training",
        "timestamp": datetime.now(UTC).isoformat(),
        "device": torch.cuda.get_device_name(0),
        "model": "forge-nano",
        "optimizer": "AdamW",
        "random_seed": BENCHMARK_SEED,
        "data_provenance": data_provenance(dataset),
        "grad_clip": 1.0,
        "run_1_lr2e4_50steps": training_stats(losses_1, times_1, mem_1, eval_before_1, eval_after_1),
        "run_2_lr5e4_50steps": training_stats(losses_2, times_2, mem_2, eval_before_2, eval_after_2),
        "run_3_lr2e4_100steps": training_stats(losses_3, times_3, mem_3, eval_before_3, eval_after_3),
    }
    results["run_1_lr2e4_50steps"]["lr"] = 2e-4
    results["run_2_lr5e4_50steps"]["lr"] = 5e-4
    results["run_3_lr2e4_100steps"]["lr"] = 2e-4

    out_path = RESULTS_DIR / "bench_03_training.json"
    write_json_artifact(out_path, results)

    print(f"\nResults saved to {out_path}")
    for name, key in [
        ("Run 1 (lr=2e-4, 50s)", "run_1_lr2e4_50steps"),
        ("Run 2 (lr=5e-4, 50s)", "run_2_lr5e4_50steps"),
        ("Run 3 (lr=2e-4, 100s)", "run_3_lr2e4_100steps"),
    ]:
        r = results[key]
        print(
            f"  {name}: {r['evaluation_loss_before']:.4f} → {r['evaluation_loss_after']:.4f} "
            f"({r['loss_reduction_pct']}%), "
            f"{r['steps_per_sec']:.1f} steps/s, {r['gpu_mem_peak_gb']} GB"
        )
    print("BENCH 03: DONE")


if __name__ == "__main__":
    main()
