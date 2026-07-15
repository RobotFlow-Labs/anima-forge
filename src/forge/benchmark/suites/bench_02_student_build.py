"""Benchmark 02: FORGEStudent Build — load time, param counts, memory, forward pass."""

from __future__ import annotations

import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from forge.benchmark.artifacts import write_json_artifact
from forge.benchmark.suites.real_data import data_provenance, load_real_dataset, real_batch
from forge.benchmark.suites.runtime import results_dir

RESULTS_DIR = results_dir()

MODEL_DIR = Path(os.environ.get("FORGE_MODEL_DIR", "./models"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_WARMUP = 5
N_ITERS = 50


def main():
    if DEVICE != "cuda":
        print("SKIP: No CUDA device")
        sys.exit(0)

    from forge.config import ForgeConfig
    from forge.student import FORGEStudent

    dataset = load_real_dataset(MODEL_DIR, max_samples=4)
    image_b1, _ = real_batch(dataset, 1, DEVICE)
    image_b2, _ = real_batch(dataset, 2, DEVICE)
    image_b4, _ = real_batch(dataset, 4, DEVICE)

    # Build with real models
    print("Building FORGEStudent with real SigLIP + Qwen...")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    student = FORGEStudent(ForgeConfig.default().student, model_dir=str(MODEL_DIR))
    build_time_cpu = time.time() - t0

    t0 = time.time()
    student = student.to(DEVICE)
    build_time_gpu = time.time() - t0
    student.eval()

    total_params = sum(p.numel() for p in student.parameters())
    trainable_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    gpu_mem_loaded = torch.cuda.max_memory_allocated() / 1e9

    # Component params
    vision_params = sum(p.numel() for p in student.vision_encoder.parameters())
    bridge_params = student.bridge.param_count()
    action_params = student.action_head.param_count()
    lang_params = sum(p.numel() for p in student.language.parameters())

    # Warmup
    print(f"Warmup ({N_WARMUP} passes)...")
    with torch.no_grad():
        for _ in range(N_WARMUP):
            _ = student(image_b1)
    torch.cuda.synchronize()

    # Forward pass benchmark batch=1
    print(f"Benchmarking forward pass batch=1 ({N_ITERS} iterations)...")
    latencies_b1 = []
    with torch.no_grad():
        for _ in range(N_ITERS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = student(image_b1)
            torch.cuda.synchronize()
            latencies_b1.append((time.perf_counter() - t0) * 1000)

    output_actions_shape = list(out["actions"].shape)
    output_vision_shape = list(out["vision_features"].shape)

    # Forward pass batch=2
    print(f"Benchmarking forward pass batch=2 ({N_ITERS // 2} iterations)...")
    latencies_b2 = []
    with torch.no_grad():
        for _ in range(N_ITERS // 2):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = student(image_b2)
            torch.cuda.synchronize()
            latencies_b2.append((time.perf_counter() - t0) * 1000)

    # Forward pass batch=4
    print(f"Benchmarking forward pass batch=4 ({N_ITERS // 2} iterations)...")
    latencies_b4 = []
    with torch.no_grad():
        for _ in range(N_ITERS // 2):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = student(image_b4)
            torch.cuda.synchronize()
            latencies_b4.append((time.perf_counter() - t0) * 1000)

    # FP16 autocast
    print(f"Benchmarking FP16 autocast batch=1 ({N_ITERS} iterations)...")
    latencies_fp16 = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        for _ in range(N_WARMUP):
            _ = student(image_b1)
        torch.cuda.synchronize()
        for _ in range(N_ITERS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = student(image_b1)
            torch.cuda.synchronize()
            latencies_fp16.append((time.perf_counter() - t0) * 1000)

    gpu_mem_peak = torch.cuda.max_memory_allocated() / 1e9

    def stats(arr):
        a = np.array(arr)
        return {
            "mean_ms": round(float(np.mean(a)), 2),
            "std_ms": round(float(np.std(a)), 2),
            "p50_ms": round(float(np.percentile(a, 50)), 2),
            "p95_ms": round(float(np.percentile(a, 95)), 2),
            "p99_ms": round(float(np.percentile(a, 99)), 2),
            "min_ms": round(float(np.min(a)), 2),
            "max_ms": round(float(np.max(a)), 2),
            "fps": round(float(1000 / np.mean(a)), 1),
            "n_samples": len(a),
        }

    results = {
        "benchmark": "student_build",
        "timestamp": datetime.now(UTC).isoformat(),
        "variant": "nano",
        "data_provenance": data_provenance(dataset),
        "device": torch.cuda.get_device_name(0),
        "build_time_cpu_s": round(build_time_cpu, 2),
        "build_time_gpu_s": round(build_time_gpu, 2),
        "params": {
            "total": total_params,
            "total_m": round(total_params / 1e6, 1),
            "trainable": trainable_params,
            "trainable_m": round(trainable_params / 1e6, 1),
            "frozen": frozen_params,
            "frozen_m": round(frozen_params / 1e6, 1),
            "trainable_pct": round(trainable_params / total_params * 100, 1),
        },
        "components": {
            "vision_m": round(vision_params / 1e6, 1),
            "bridge_m": round(bridge_params / 1e6, 1),
            "language_m": round(lang_params / 1e6, 1),
            "action_head_m": round(action_params / 1e6, 1),
        },
        "memory": {
            "loaded_gb": round(gpu_mem_loaded, 2),
            "peak_gb": round(gpu_mem_peak, 2),
        },
        "output_shapes": {
            "actions": output_actions_shape,
            "vision_features": output_vision_shape,
        },
        "latency_fp32_b1": stats(latencies_b1),
        "latency_fp32_b2": stats(latencies_b2),
        "latency_fp32_b4": stats(latencies_b4),
        "latency_fp16_b1": stats(latencies_fp16),
        "fp16_speedup": round(np.mean(latencies_b1) / np.mean(latencies_fp16), 2),
        "batch_scaling": {
            "b1_fps": round(1000 / np.mean(latencies_b1), 1),
            "b2_fps": round(2 * 1000 / np.mean(latencies_b2), 1),
            "b4_fps": round(4 * 1000 / np.mean(latencies_b4), 1),
        },
    }

    out_path = RESULTS_DIR / "bench_02_student_build.json"
    write_json_artifact(out_path, results)

    print(f"\nResults saved to {out_path}")
    print(f"  Total: {results['params']['total_m']}M ({results['params']['trainable_pct']}% trainable)")
    print(f"  FP32 b=1: {results['latency_fp32_b1']['p50_ms']}ms p50, {results['latency_fp32_b1']['fps']} fps")
    print(f"  FP16 b=1: {results['latency_fp16_b1']['p50_ms']}ms p50, {results['latency_fp16_b1']['fps']} fps")
    print(f"  GPU mem: {results['memory']['peak_gb']} GB")
    print("BENCH 02: DONE")


if __name__ == "__main__":
    main()
