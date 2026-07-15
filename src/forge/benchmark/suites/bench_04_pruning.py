"""Benchmark 04: Chunk-Aware Layer Pruning — importance scoring, pruned model validation."""

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
N_CALIB = 5
N_ITERS = 30


def main():
    if DEVICE != "cuda":
        print("SKIP: No CUDA device")
        sys.exit(0)

    from forge.config import ForgeConfig
    from forge.prune import _find_transformer_layers
    from forge.prune_v2 import compute_chunk_layer_importance, prune_chunk_aware
    from forge.student import FORGEStudent

    dataset = load_real_dataset(MODEL_DIR, max_samples=max(N_CALIB, 1))
    image_b1, _ = real_batch(dataset, 1, DEVICE)

    print("Building FORGEStudent...")
    student = FORGEStudent(ForgeConfig.default().student, model_dir=str(MODEL_DIR)).to(DEVICE)
    student.eval()

    n_params_orig = sum(p.numel() for p in student.parameters())
    layers = _find_transformer_layers(student)
    n_layers = len(layers)
    print(f"  {n_params_orig / 1e6:.1f}M params, {n_layers} transformer layers")

    if n_layers < 6:
        print("SKIP: Not enough layers for pruning")
        sys.exit(0)

    # Compute importance with multiple alpha values
    calib = [real_batch(dataset, 1, DEVICE, start=index)[0] for index in range(N_CALIB)]

    importance_results = {}
    for alpha in [0.4, 0.6, 0.8, 1.0]:
        print(f"\nComputing importance (alpha={alpha}, {N_CALIB} samples)...")
        t0 = time.time()
        importance = compute_chunk_layer_importance(student, calib, action_horizon=1, alpha=alpha)
        t_imp = time.time() - t0
        importance_results[f"alpha_{alpha}"] = {
            "alpha": alpha,
            "compute_time_s": round(t_imp, 2),
            "scores": {str(k): round(v, 6) for k, v in sorted(importance.items())},
            "top_5": [k for k, _ in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]],
            "bottom_5": [k for k, _ in sorted(importance.items(), key=lambda x: x[1])[:5]],
        }

    # Prune at different ratios
    importance_default = compute_chunk_layer_importance(student, calib, alpha=0.6)
    prune_results = {}

    for keep_ratio in [0.9, 0.75, 0.6, 0.5]:
        target = max(4, int(n_layers * keep_ratio))
        label = f"keep_{int(keep_ratio * 100)}pct"
        print(f"\nPruning: {n_layers} → {target} layers (keep {keep_ratio * 100:.0f}%)...")

        t0 = time.time()
        pruned, removed = prune_chunk_aware(student, importance_default, target_layers=target)
        t_prune = time.time() - t0

        pruned = pruned.to(DEVICE)
        pruned.eval()
        n_params_pruned = sum(p.numel() for p in pruned.parameters())

        # Benchmark pruned model
        torch.cuda.reset_peak_memory_stats()
        latencies = []
        with torch.no_grad():
            for _ in range(5):  # warmup
                _ = pruned(image_b1)
            torch.cuda.synchronize()
            for _ in range(N_ITERS):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                out = pruned(image_b1)
                torch.cuda.synchronize()
                latencies.append((time.perf_counter() - t0) * 1000)

        la = np.array(latencies)
        gpu_mem = torch.cuda.max_memory_allocated() / 1e9

        prune_results[label] = {
            "keep_ratio": keep_ratio,
            "layers_before": n_layers,
            "layers_after": n_layers - len(removed),
            "layers_removed": removed,
            "params_before": n_params_orig,
            "params_before_m": round(n_params_orig / 1e6, 1),
            "params_after": n_params_pruned,
            "params_after_m": round(n_params_pruned / 1e6, 1),
            "params_retained_pct": round(n_params_pruned / n_params_orig * 100, 1),
            "prune_time_s": round(t_prune, 2),
            "latency_p50_ms": round(float(np.percentile(la, 50)), 2),
            "latency_p95_ms": round(float(np.percentile(la, 95)), 2),
            "latency_mean_ms": round(float(np.mean(la)), 2),
            "fps": round(float(1000 / np.mean(la)), 1),
            "gpu_mem_gb": round(gpu_mem, 2),
            "output_shape": list(out["actions"].shape),
        }

        del pruned
        torch.cuda.empty_cache()

    results = {
        "benchmark": "pruning",
        "timestamp": datetime.now(UTC).isoformat(),
        "device": torch.cuda.get_device_name(0),
        "model": "forge-nano",
        "data_provenance": data_provenance(dataset),
        "original_layers": n_layers,
        "original_params_m": round(n_params_orig / 1e6, 1),
        "n_calibration_samples": N_CALIB,
        "importance_scoring": importance_results,
        "pruning_results": prune_results,
    }

    out_path = RESULTS_DIR / "bench_04_pruning.json"
    write_json_artifact(out_path, results)

    print(f"\nResults saved to {out_path}")
    for label, r in prune_results.items():
        print(
            f"  {label}: {r['layers_before']}→{r['layers_after']} layers, "
            f"{r['params_after_m']}M params, {r['latency_p50_ms']}ms, {r['fps']} fps"
        )
    print("BENCH 04: DONE")


if __name__ == "__main__":
    main()
