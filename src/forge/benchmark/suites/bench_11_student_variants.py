"""Benchmark 11: Student Variants — Nano (0.5B) vs Small (1.5B), different configs.

Compares different student architectures, LoRA ranks, action heads, and
demonstrates FORGE's flexibility across model sizes.
"""

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


def bench_student_variant(dataset, variant_name, config_overrides, n_train_steps=30, n_infer_iters=30, train_batch=4):
    """Build, train, and benchmark a student variant."""
    from forge.config import ForgeConfig, apply_student_variant
    from forge.student import FORGEStudent

    print(f"\n=== Variant: {variant_name} ===")

    seed = reset_benchmark_rng()
    config = ForgeConfig.default()
    apply_student_variant(config.student, str(config_overrides["variant"]))
    for key, val in config_overrides.items():
        setattr(config.student, key, val)
    config.student.action_dim = dataset.action_dim
    image_b1, _ = real_batch(dataset, 1, DEVICE)

    # Build
    t0 = time.perf_counter()
    student = FORGEStudent(config.student, model_dir=str(MODEL_DIR)).to(DEVICE)
    build_time = time.perf_counter() - t0
    print(
        f"  Build: {build_time:.1f}s, {student.total_params / 1e6:.1f}M params "
        f"({student.trainable_params / 1e6:.1f}M trainable)"
    )

    # Inference benchmark
    student.eval()
    times_fp32 = []
    times_fp16 = []
    with torch.no_grad():
        # Warmup
        for _ in range(5):
            student(image_b1)

        # FP32
        for _ in range(n_infer_iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = student(image_b1)
            torch.cuda.synchronize()
            times_fp32.append((time.perf_counter() - t0) * 1000)

        # FP16
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            for _ in range(5):
                student(image_b1)

            for _ in range(n_infer_iters):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = student(image_b1)
                torch.cuda.synchronize()
                times_fp16.append((time.perf_counter() - t0) * 1000)

    fp32 = np.array(times_fp32[5:])
    fp16 = np.array(times_fp16[5:])
    infer_mem = torch.cuda.max_memory_allocated() / 1024**3
    torch.cuda.reset_peak_memory_stats()

    print(f"  FP32: {fp32.mean():.1f}ms ({1000 / fp32.mean():.1f} fps)")
    print(f"  FP16: {fp16.mean():.1f}ms ({1000 / fp16.mean():.1f} fps)")

    # Training benchmark
    evaluation_loss_before = fixed_action_loss(student, dataset, DEVICE)
    student.train()
    optimizer = torch.optim.AdamW(student.trainable_parameters(), lr=2e-4)
    losses = []
    train_times = []

    for step in range(n_train_steps):
        img, gt = real_batch(dataset, train_batch, DEVICE, start=step * train_batch)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = student(img, gt_actions=gt)
        loss = out["loss"]
        optimizer.zero_grad()
        backward_with_finite_gradients(loss, student.trainable_parameters())
        optimizer.step()
        torch.cuda.synchronize()

        train_times.append((time.perf_counter() - t0) * 1000)
        losses.append(loss.item())

        if step % 10 == 0:
            print(f"  Train step {step}: loss={loss.item():.4f}")

    evaluation_loss_after = fixed_action_loss(student, dataset, DEVICE)
    evaluation_loss_reduction_pct = (
        (evaluation_loss_before - evaluation_loss_after) / max(evaluation_loss_before, 1e-12) * 100
    )
    ta = np.array(train_times[5:])
    la = np.array(losses)
    train_mem = torch.cuda.max_memory_allocated() / 1024**3
    torch.cuda.reset_peak_memory_stats()

    result = {
        "variant": variant_name,
        "random_seed": seed,
        "config": config_overrides,
        "total_params_m": round(student.total_params / 1e6, 1),
        "trainable_params_m": round(student.trainable_params / 1e6, 1),
        "frozen_params_m": round((student.total_params - student.trainable_params) / 1e6, 1),
        "build_time_s": round(build_time, 1),
        "inference": {
            "fp32_p50_ms": round(float(np.percentile(fp32, 50)), 2),
            "fp32_fps": round(float(1000 / fp32.mean()), 1),
            "fp16_p50_ms": round(float(np.percentile(fp16, 50)), 2),
            "fp16_fps": round(float(1000 / fp16.mean()), 1),
            "fp16_speedup": round(float(fp32.mean() / fp16.mean()), 2),
            "gpu_mem_gb": round(infer_mem, 2),
        },
        "training": {
            "n_steps": n_train_steps,
            "loss_metric": "fixed-real-evaluation-mean",
            "evaluation_samples": 5,
            "evaluation_loss_before": round(float(evaluation_loss_before), 4),
            "evaluation_loss_after": round(float(evaluation_loss_after), 4),
            "loss_reduction_pct": round(float(evaluation_loss_reduction_pct), 1),
            "training_loss_first": round(float(la[0]), 4),
            "training_loss_last": round(float(la[-1]), 4),
            "step_time_ms": round(float(ta.mean()), 1),
            "steps_per_sec": round(float(1000 / ta.mean()), 2),
            "gpu_mem_gb": round(train_mem, 2),
            "training_loss_curve": [round(loss, 4) for loss in losses],
        },
    }

    del student, optimizer
    torch.cuda.empty_cache()

    return result


def main():
    if DEVICE == "cpu":
        print("SKIP: No CUDA")
        sys.exit(0)

    dataset = load_real_dataset(MODEL_DIR, max_samples=120)

    variants = [
        # Canonical v3 micro baseline.
        (
            "micro_baseline",
            {
                "variant": "micro",
                "action_head_type": "diffusion",
            },
        ),
        # Canonical v3 nano baseline.
        (
            "nano_baseline",
            {
                "variant": "nano",
                "action_head_type": "diffusion",
            },
        ),
        # Flagship nano flow configuration.
        (
            "nano_flow_lora64",
            {
                "variant": "nano",
                "lora_rank": 64,
                "action_head_type": "flow",
            },
        ),
        # Canonical v3 small backbone.
        (
            "small_baseline",
            {
                "variant": "small",
                "action_head_type": "diffusion",
            },
            1,
        ),
        # Canonical v3 medium bf16 backbone.
        (
            "medium_baseline",
            {
                "variant": "medium",
                "action_head_type": "diffusion",
            },
            1,
        ),
    ]

    all_results = {}
    for item in variants:
        name, overrides = item[0], item[1]
        batch = item[2] if len(item) > 2 else 4
        try:
            all_results[name] = bench_student_variant(dataset, name, overrides, train_batch=batch)
        except Exception as e:
            print(f"  ERROR: {e}")
            all_results[name] = {"error": str(e)}
            torch.cuda.empty_cache()

    # Summary table
    print("\n" + "=" * 80)
    print(f"{'Variant':<20} {'Params':>8} {'FP32':>8} {'FP16':>8} {'Train':>8} {'Loss↓':>8} {'VRAM':>6}")
    print("-" * 80)
    for name, r in all_results.items():
        if "error" in r:
            print(f"{name:<20} ERROR: {r['error'][:50]}")
            continue
        print(
            f"{name:<20} {r['total_params_m']:>6.1f}M "
            f"{r['inference']['fp32_fps']:>5.1f}fps "
            f"{r['inference']['fp16_fps']:>5.1f}fps "
            f"{r['training']['steps_per_sec']:>5.2f}s/s "
            f"{r['training']['loss_reduction_pct']:>5.1f}% "
            f"{r['training']['gpu_mem_gb']:>4.1f}G"
        )

    results = {
        "benchmark": "student_variants",
        "timestamp": datetime.now(UTC).isoformat(),
        "device": DEVICE,
        "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else "none",
        "random_seed": BENCHMARK_SEED,
        "data_provenance": data_provenance(dataset),
        "variants": all_results,
    }

    out_path = RESULTS_DIR / "bench_11_student_variants.json"
    write_json_artifact(out_path, results)

    print(f"\nResults saved to {out_path}")
    print("BENCH 11: DONE")


if __name__ == "__main__":
    main()
