"""Benchmark 08: End-to-End Pipeline — build → train → prune → quantize → inference."""

from __future__ import annotations

import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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


def main():
    if DEVICE != "cuda":
        print("SKIP: No CUDA device")
        sys.exit(0)

    from forge.config import ForgeConfig
    from forge.prune_v2 import compute_chunk_layer_importance, prune_chunk_aware
    from forge.quantize_v2 import calibrate_chunk_ranges, quantize_chunk_aware
    from forge.student import FORGEStudent

    reset_benchmark_rng()
    dataset = load_real_dataset(MODEL_DIR, max_samples=30)
    config = ForgeConfig.default()
    config.student.action_dim = dataset.action_dim
    image_b1, _ = real_batch(dataset, 1, DEVICE)

    pipeline_steps: dict[str, dict[str, Any]] = {}
    total_t0 = time.time()

    # ── Step 1: Build ──
    print("=== Step 1: Build FORGEStudent ===")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    student = FORGEStudent(config.student, model_dir=str(MODEL_DIR)).to(DEVICE)
    build_time = time.time() - t0
    n_params = sum(p.numel() for p in student.parameters())

    pipeline_steps["build"] = {
        "time_s": round(build_time, 2),
        "params_m": round(n_params / 1e6, 1),
        "gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    }
    print(f"  {n_params / 1e6:.1f}M params in {build_time:.1f}s")

    # ── Step 2: Train (30 steps) ──
    print("\n=== Step 2: KD Training (30 steps) ===")
    student.train()
    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=2e-4,
        weight_decay=0.01,
    )
    torch.cuda.reset_peak_memory_stats()

    evaluation_loss_before = fixed_action_loss(student, dataset, DEVICE)
    losses = []
    t0 = time.time()
    for step in range(30):
        optimizer.zero_grad()
        images, actions = real_batch(dataset, 1, DEVICE, start=step)
        out = student(images, gt_actions=actions)
        backward_with_finite_gradients(out["loss"], student.parameters())
        optimizer.step()
        losses.append(out["loss"].item())
    train_time = time.time() - t0
    evaluation_loss_after = fixed_action_loss(student, dataset, DEVICE)
    evaluation_loss_reduction_pct = round(
        (evaluation_loss_before - evaluation_loss_after) / max(evaluation_loss_before, 1e-12) * 100,
        1,
    )

    pipeline_steps["train"] = {
        "n_steps": 30,
        "time_s": round(train_time, 1),
        "steps_per_sec": round(30 / train_time, 2),
        "evaluation_samples": 5,
        "loss_metric": "fixed-real-evaluation-mean",
        "evaluation_loss_before": round(evaluation_loss_before, 4),
        "evaluation_loss_after": round(evaluation_loss_after, 4),
        "evaluation_loss_reduction_pct": evaluation_loss_reduction_pct,
        "loss_reduction_pct": evaluation_loss_reduction_pct,
        "training_loss_first": round(losses[0], 4),
        "training_loss_last": round(losses[-1], 4),
        "gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    }
    print(f"  Fixed-set loss: {evaluation_loss_before:.4f} → {evaluation_loss_after:.4f} in {train_time:.1f}s")

    # ── Step 3: Prune ──
    print("\n=== Step 3: Layer Pruning (keep 75%) ===")
    student.eval()
    calib = [real_batch(dataset, 1, DEVICE, start=index)[0] for index in range(3)]

    t0 = time.time()
    importance = compute_chunk_layer_importance(student, calib, alpha=0.6)
    imp_time = time.time() - t0

    from forge.prune import _find_transformer_layers

    n_layers = len(_find_transformer_layers(student))
    target = max(4, int(n_layers * 0.75))

    t0 = time.time()
    pruned, removed = prune_chunk_aware(student, importance, target_layers=target)
    prune_time = time.time() - t0
    pruned = pruned.to(DEVICE)
    n_pruned = sum(p.numel() for p in pruned.parameters())

    pipeline_steps["prune"] = {
        "importance_time_s": round(imp_time, 2),
        "prune_time_s": round(prune_time, 2),
        "layers_before": n_layers,
        "layers_after": n_layers - len(removed),
        "layers_removed": removed,
        "params_before_m": round(n_params / 1e6, 1),
        "params_after_m": round(n_pruned / 1e6, 1),
        "params_retained_pct": round(n_pruned / n_params * 100, 1),
    }
    print(f"  {n_layers} → {n_layers - len(removed)} layers, {n_pruned / 1e6:.1f}M params")

    # ── Step 4: Quantize ──
    print("\n=== Step 4: INT4 Quantization ===")
    t0 = time.time()
    ranges = calibrate_chunk_ranges(pruned, calib)
    q_model = quantize_chunk_aware(pruned, target_bits=4.0, chunk_calibration=ranges, action_head_bits=8)
    quant_time = time.time() - t0
    q_model = q_model.to(DEVICE)

    fp32_size = n_pruned * 4 / 1e6
    int4_size = fp32_size * 4 / 32

    pipeline_steps["quantize"] = {
        "time_s": round(quant_time, 1),
        "fp32_size_mb": round(fp32_size, 1),
        "estimated_int4_packed_size_mb": round(int4_size, 1),
        "estimated_weight_compression_ratio": round(fp32_size / int4_size, 1),
        "quantization_storage": "fake-quantized-fp32-for-quality-analysis",
    }
    print(f"  {fp32_size:.0f} MB → {int4_size:.0f} MB ({fp32_size / int4_size:.1f}x)")

    # ── Step 5: Inference Benchmark ──
    print("\n=== Step 5: Inference Benchmark ===")
    q_model.eval()
    torch.cuda.reset_peak_memory_stats()

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = q_model(image_b1)
    torch.cuda.synchronize()

    # FP32 (fake-quantized) latency
    latencies_fp32 = []
    with torch.no_grad():
        for _ in range(50):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = q_model(image_b1)
            torch.cuda.synchronize()
            latencies_fp32.append((time.perf_counter() - t0) * 1000)

    # FP16 autocast
    latencies_fp16 = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        for _ in range(5):
            _ = q_model(image_b1)
        torch.cuda.synchronize()
        for _ in range(50):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = q_model(image_b1)
            torch.cuda.synchronize()
            latencies_fp16.append((time.perf_counter() - t0) * 1000)

    la32 = np.array(latencies_fp32)
    la16 = np.array(latencies_fp16)
    gpu_mem = torch.cuda.max_memory_allocated() / 1e9

    pipeline_steps["inference"] = {
        "fp32_p50_ms": round(float(np.percentile(la32, 50)), 2),
        "fp32_p95_ms": round(float(np.percentile(la32, 95)), 2),
        "fp32_fps": round(float(1000 / np.mean(la32)), 1),
        "fp16_p50_ms": round(float(np.percentile(la16, 50)), 2),
        "fp16_p95_ms": round(float(np.percentile(la16, 95)), 2),
        "fp16_fps": round(float(1000 / np.mean(la16)), 1),
        "fp16_speedup": round(float(np.mean(la32) / np.mean(la16)), 2),
        "gpu_mem_gb": round(gpu_mem, 2),
    }

    total_time = time.time() - total_t0

    results: dict[str, Any] = {
        "benchmark": "e2e_pipeline",
        "timestamp": datetime.now(UTC).isoformat(),
        "device": torch.cuda.get_device_name(0),
        "model": "forge-nano",
        "random_seed": BENCHMARK_SEED,
        "data_provenance": data_provenance(dataset),
        "total_pipeline_time_s": round(total_time, 1),
        "steps": pipeline_steps,
        "summary": {
            "original_params_m": pipeline_steps["build"]["params_m"],
            "pruned_params_m": pipeline_steps["prune"]["params_after_m"],
            "estimated_int4_packed_size_mb": pipeline_steps["quantize"]["estimated_int4_packed_size_mb"],
            "estimated_weight_compression_ratio": pipeline_steps["quantize"]["estimated_weight_compression_ratio"],
            "evaluation_loss_reduction_pct": pipeline_steps["train"]["evaluation_loss_reduction_pct"],
            "loss_reduction_pct": pipeline_steps["train"]["loss_reduction_pct"],
            "fp32_latency_ms": pipeline_steps["inference"]["fp32_p50_ms"],
            "fp16_latency_ms": pipeline_steps["inference"]["fp16_p50_ms"],
            "fp16_fps": pipeline_steps["inference"]["fp16_fps"],
        },
    }

    out_path = RESULTS_DIR / "bench_08_e2e_pipeline.json"
    write_json_artifact(out_path, results)

    print(f"\n{'=' * 60}")
    print(f"Results saved to {out_path}")
    print(f"Total pipeline time: {total_time:.0f}s")
    s = results["summary"]
    print(f"  Params: {s['original_params_m']}M → {s['pruned_params_m']}M (pruned)")
    print(
        f"  Estimated packed size: {s['estimated_int4_packed_size_mb']:.0f} MB INT4 "
        f"({s['estimated_weight_compression_ratio']}x)"
    )
    print(
        "  Fixed-set loss: "
        f"{pipeline_steps['train']['evaluation_loss_before']:.4f} → "
        f"{pipeline_steps['train']['evaluation_loss_after']:.4f}"
    )
    print(f"  Latency: FP32 {s['fp32_latency_ms']}ms, FP16 {s['fp16_latency_ms']}ms ({s['fp16_fps']} fps)")
    print("BENCH 08: DONE")


if __name__ == "__main__":
    main()
