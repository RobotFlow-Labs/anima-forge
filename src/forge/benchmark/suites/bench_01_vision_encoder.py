"""Benchmark 01: Vision Encoder (SigLIP2-SO400M) — latency, memory, output validation."""

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
from forge.benchmark.suites.real_data import data_provenance, load_real_dataset, real_batch
from forge.benchmark.suites.runtime import results_dir

RESULTS_DIR = results_dir()

MODEL_DIR = Path(os.environ.get("FORGE_MODEL_DIR", "./models"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_WARMUP = 10
N_ITERS = 100


def main():
    if DEVICE != "cuda":
        print("SKIP: No CUDA device")
        sys.exit(0)

    siglip_path = MODEL_DIR / "google--siglip2-so400m-patch14-384"
    if not siglip_path.exists():
        print(f"SKIP: SigLIP not found at {siglip_path}")
        sys.exit(0)

    from transformers import SiglipVisionModel

    dataset = load_real_dataset(MODEL_DIR, max_samples=8)
    image_b1, _ = real_batch(dataset, 1, DEVICE)
    image_b4, _ = real_batch(dataset, 4, DEVICE)
    image_b8, _ = real_batch(dataset, 8, DEVICE)

    print("Loading SigLIP2-SO400M...")
    t0 = time.time()
    encoder: Any = SiglipVisionModel.from_pretrained(str(siglip_path), local_files_only=True)
    load_time_cpu = time.time() - t0

    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()

    t0 = time.time()
    encoder = encoder.to(DEVICE)
    load_time_gpu = time.time() - t0

    torch.cuda.reset_peak_memory_stats()
    n_params = sum(p.numel() for p in encoder.parameters())
    gpu_mem_loaded = torch.cuda.max_memory_allocated() / 1e9

    # Warmup
    print(f"Warmup ({N_WARMUP} passes)...")
    with torch.no_grad():
        for _ in range(N_WARMUP):
            _ = encoder(image_b1)
    torch.cuda.synchronize()

    # Benchmark single image
    print(f"Benchmarking batch=1 ({N_ITERS} iterations)...")
    latencies_b1 = []
    with torch.no_grad():
        for _ in range(N_ITERS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = encoder(image_b1)
            torch.cuda.synchronize()
            latencies_b1.append((time.perf_counter() - t0) * 1000)

    # Verify output shape
    last_hidden = out.last_hidden_state
    output_shape = list(last_hidden.shape)

    # Benchmark batch=4
    print(f"Benchmarking batch=4 ({N_ITERS // 2} iterations)...")
    latencies_b4 = []
    with torch.no_grad():
        for _ in range(N_ITERS // 2):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = encoder(image_b4)
            torch.cuda.synchronize()
            latencies_b4.append((time.perf_counter() - t0) * 1000)

    # Benchmark batch=8
    print(f"Benchmarking batch=8 ({N_ITERS // 4} iterations)...")
    latencies_b8 = []
    with torch.no_grad():
        for _ in range(N_ITERS // 4):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = encoder(image_b8)
            torch.cuda.synchronize()
            latencies_b8.append((time.perf_counter() - t0) * 1000)

    # FP16 autocast benchmark
    print(f"Benchmarking FP16 autocast batch=1 ({N_ITERS} iterations)...")
    latencies_fp16 = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        for _ in range(N_ITERS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = encoder(image_b1)
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
        "benchmark": "vision_encoder",
        "timestamp": datetime.now(UTC).isoformat(),
        "model": "google/siglip2-so400m-patch14-384",
        "data_provenance": data_provenance(dataset),
        "device": torch.cuda.get_device_name(0),
        "params": n_params,
        "params_m": round(n_params / 1e6, 1),
        "load_time_cpu_s": round(load_time_cpu, 2),
        "load_time_gpu_s": round(load_time_gpu, 2),
        "gpu_mem_loaded_gb": round(gpu_mem_loaded, 2),
        "gpu_mem_peak_gb": round(gpu_mem_peak, 2),
        "output_shape": output_shape,
        "d_output": output_shape[2],
        "n_tokens": output_shape[1],
        "latency_fp32_b1": stats(latencies_b1),
        "latency_fp32_b4": stats(latencies_b4),
        "latency_fp32_b8": stats(latencies_b8),
        "latency_fp16_b1": stats(latencies_fp16),
        "fp16_speedup": round(np.mean(latencies_b1) / np.mean(latencies_fp16), 2),
    }

    out_path = RESULTS_DIR / "bench_01_vision_encoder.json"
    write_json_artifact(out_path, results)

    print(f"\nResults saved to {out_path}")
    print(f"  Params: {results['params_m']}M")
    print(f"  FP32 b=1: {results['latency_fp32_b1']['p50_ms']}ms p50, {results['latency_fp32_b1']['fps']} fps")
    print(f"  FP16 b=1: {results['latency_fp16_b1']['p50_ms']}ms p50, {results['latency_fp16_b1']['fps']} fps")
    print(f"  FP16 speedup: {results['fp16_speedup']}x")
    print(f"  GPU mem: {results['gpu_mem_peak_gb']} GB")
    print("BENCH 01: DONE")


if __name__ == "__main__":
    main()
