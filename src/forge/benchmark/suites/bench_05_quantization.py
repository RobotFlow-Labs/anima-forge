"""Benchmark 05: Chunk-Aware Quantization — INT4/INT8 compression, quality metrics."""

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
N_CALIB = 5
N_QUALITY = 10
N_LATENCY = 30


def main():
    if DEVICE != "cuda":
        print("SKIP: No CUDA device")
        sys.exit(0)

    from forge.config import ForgeConfig
    from forge.quantize_v2 import (
        calibrate_chunk_ranges,
        measure_chunk_quantization_quality,
        quantize_chunk_aware,
    )
    from forge.student import FORGEStudent

    dataset = load_real_dataset(MODEL_DIR, max_samples=max(N_CALIB, N_QUALITY))
    image_b1, _ = real_batch(dataset, 1, DEVICE)

    print("Building FORGEStudent...")
    student = FORGEStudent(ForgeConfig.default().student, model_dir=str(MODEL_DIR)).to(DEVICE)
    student.eval()

    n_params = sum(p.numel() for p in student.parameters())
    fp32_size_mb = n_params * 4 / 1e6

    # Calibrate
    print(f"Calibrating chunk ranges ({N_CALIB} samples)...")
    calib = [real_batch(dataset, 1, DEVICE, start=index)[0] for index in range(N_CALIB)]
    t0 = time.time()
    ranges = calibrate_chunk_ranges(student, calib)
    cal_time = time.time() - t0
    print(f"  Calibrated {len(ranges)} modules in {cal_time:.1f}s")

    # Test quality data
    quality_data = [real_batch(dataset, 1, DEVICE, start=index)[0] for index in range(N_QUALITY)]

    # Quantize at different bit widths
    quant_results: dict[str, Any] = {}
    for target_bits, ah_bits in [(8, 8), (4, 8), (4, 4), (3, 8)]:
        label = f"int{target_bits}_ah{ah_bits}"
        print(f"\nQuantizing: target={target_bits}bit, action_head={ah_bits}bit...")

        t0 = time.time()
        q_model = quantize_chunk_aware(
            student,
            target_bits=float(target_bits),
            chunk_calibration=ranges,
            action_head_bits=ah_bits,
        )
        quant_time = time.time() - t0

        q_model = q_model.to(DEVICE)
        q_model.eval()

        # Estimated size
        est_size_mb = fp32_size_mb * target_bits / 32.0
        compression = fp32_size_mb / est_size_mb

        # Quality metrics
        print("  Measuring quality...")
        quality = measure_chunk_quantization_quality(student, q_model, quality_data)

        # Latency
        print(f"  Benchmarking latency ({N_LATENCY} iters)...")
        torch.cuda.reset_peak_memory_stats()
        latencies = []
        with torch.no_grad():
            for _ in range(5):  # warmup
                _ = q_model(image_b1)
            torch.cuda.synchronize()
            for _ in range(N_LATENCY):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = q_model(image_b1)
                torch.cuda.synchronize()
                latencies.append((time.perf_counter() - t0) * 1000)

        la = np.array(latencies)
        gpu_mem = torch.cuda.max_memory_allocated() / 1e9

        quant_results[label] = {
            "target_bits": target_bits,
            "action_head_bits": ah_bits,
            "quantize_time_s": round(quant_time, 1),
            "fp32_size_mb": round(fp32_size_mb, 1),
            "estimated_packed_size_mb": round(est_size_mb, 1),
            "estimated_weight_compression_ratio": round(compression, 1),
            "quantization_storage": "fake-quantized-fp32-for-quality-analysis",
            "quality": {
                "action_mse": round(quality["action_mse"], 6),
                "temporal_coherence_delta": round(quality["temporal_coherence_delta"], 6),
                "max_step_drift": round(quality["max_step_drift"], 6),
                "per_step_error": [round(x, 6) for x in quality.get("per_step_error", [])],
            },
            "latency_p50_ms": round(float(np.percentile(la, 50)), 2),
            "latency_p95_ms": round(float(np.percentile(la, 95)), 2),
            "latency_mean_ms": round(float(np.mean(la)), 2),
            "fps": round(float(1000 / np.mean(la)), 1),
            "gpu_mem_gb": round(gpu_mem, 2),
        }

        del q_model
        torch.cuda.empty_cache()

    results = {
        "benchmark": "quantization",
        "timestamp": datetime.now(UTC).isoformat(),
        "device": torch.cuda.get_device_name(0),
        "model": "forge-nano",
        "data_provenance": data_provenance(dataset),
        "params": n_params,
        "params_m": round(n_params / 1e6, 1),
        "fp32_size_mb": round(fp32_size_mb, 1),
        "n_calibration_samples": N_CALIB,
        "n_quality_samples": N_QUALITY,
        "calibration_time_s": round(cal_time, 1),
        "calibrated_modules": len(ranges),
        "quantization_results": quant_results,
    }

    out_path = RESULTS_DIR / "bench_05_quantization.json"
    write_json_artifact(out_path, results)

    print(f"\nResults saved to {out_path}")
    for label, r in quant_results.items():
        print(
            f"  {label}: estimated {r['estimated_weight_compression_ratio']}x packed compression, "
            f"MSE={r['quality']['action_mse']:.4f}, {r['latency_p50_ms']}ms, {r['fps']} fps"
        )
    print("BENCH 05: DONE")


if __name__ == "__main__":
    main()
