"""Benchmark 12: Full Pipeline Combinations — build → train → prune → quantize → benchmark.

Tests multiple configuration combinations end-to-end to find optimal settings.
This is the DD-ready benchmark that proves FORGE works across configurations.
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


def run_pipeline(dataset, name, student_overrides, prune_ratio=0.75, quant_bits=4, train_steps=30, train_batch=4):
    """Run full pipeline: build → train → prune → quantize → latency."""
    from forge.config import ForgeConfig, apply_student_variant
    from forge.student import FORGEStudent

    print(f"\n{'=' * 60}")
    print(f"Pipeline: {name}")
    print(f"{'=' * 60}")

    seed = reset_benchmark_rng()
    result: dict[str, Any] = {"name": name, "config": student_overrides, "random_seed": seed}
    timings = {}

    # BUILD
    config = ForgeConfig.default()
    apply_student_variant(config.student, str(student_overrides["variant"]))
    for key, val in student_overrides.items():
        setattr(config.student, key, val)
    config.student.action_dim = dataset.action_dim
    image_b1, _ = real_batch(dataset, 1, DEVICE)

    t0 = time.perf_counter()
    student: Any = FORGEStudent(config.student, model_dir=str(MODEL_DIR)).to(DEVICE)
    timings["build_s"] = round(time.perf_counter() - t0, 1)

    result["model"] = {
        "total_params_m": round(student.total_params / 1e6, 1),
        "trainable_params_m": round(student.trainable_params / 1e6, 1),
    }
    print(f"  Build: {timings['build_s']}s, {result['model']['total_params_m']}M params")

    # TRAIN
    student.train()
    optimizer = torch.optim.AdamW(student.trainable_parameters(), lr=2e-4)
    losses = []
    evaluation_loss_before = fixed_action_loss(student, dataset, DEVICE)

    t0 = time.perf_counter()
    for step in range(train_steps):
        img, gt = real_batch(dataset, train_batch, DEVICE, start=step * train_batch)
        out = student(img, gt_actions=gt)
        loss = out["loss"]
        optimizer.zero_grad()
        backward_with_finite_gradients(loss, student.trainable_parameters())
        optimizer.step()
        losses.append(loss.item())
    timings["train_s"] = round(time.perf_counter() - t0, 1)
    del optimizer
    evaluation_loss_after = fixed_action_loss(student, dataset, DEVICE)
    evaluation_loss_reduction_pct = (
        (evaluation_loss_before - evaluation_loss_after) / max(evaluation_loss_before, 1e-12) * 100
    )

    result["training"] = {
        "steps": train_steps,
        "loss_metric": "fixed-real-evaluation-mean",
        "evaluation_samples": 5,
        "evaluation_loss_before": round(evaluation_loss_before, 4),
        "evaluation_loss_after": round(evaluation_loss_after, 4),
        "loss_reduction_pct": round(evaluation_loss_reduction_pct, 1),
        "training_loss_first": round(losses[0], 4),
        "training_loss_last": round(losses[-1], 4),
        "speed_steps_per_s": round(train_steps / timings["train_s"], 1),
    }
    print(
        f"  Train: {timings['train_s']}s, fixed-set loss "
        f"{evaluation_loss_before:.3f}→{evaluation_loss_after:.3f} "
        f"({result['training']['loss_reduction_pct']}%)"
    )

    # PRUNE
    from forge.prune_v2 import compute_chunk_layer_importance, prune_chunk_aware

    student.eval()
    n_layers_before = None
    # Count transformer layers
    if hasattr(student.language, "model") and hasattr(student.language.model, "layers"):
        n_layers_before = len(student.language.model.layers)

    if n_layers_before and n_layers_before > 4:
        t0 = time.perf_counter()

        # Calibration data: list of image tensors
        calib_data = [real_batch(dataset, 2, DEVICE, start=index * 2)[0] for index in range(3)]

        scores = compute_chunk_layer_importance(student, calib_data, alpha=0.6)
        n_keep = max(4, int(n_layers_before * prune_ratio))
        pruned_module, removed = prune_chunk_aware(student, scores, target_layers=n_keep)
        pruned_student: Any = pruned_module
        timings["prune_s"] = round(time.perf_counter() - t0, 1)

        n_layers_after = n_layers_before - len(removed)

        result["pruning"] = {
            "ratio": prune_ratio,
            "layers_before": n_layers_before,
            "layers_after": n_layers_after,
            "layers_removed": removed,
            "params_after_m": round(pruned_student.total_params / 1e6, 1),
        }
        print(
            f"  Prune: {n_layers_before}→{n_layers_after} layers ({len(removed)} removed), "
            f"{result['pruning']['params_after_m']}M params"
        )
        student = pruned_student
    else:
        raise RuntimeError("Real language backbone exposes too few transformer layers for pruning")

    # QUANTIZE
    from forge.quantize_v2 import quantize_chunk_aware

    t0 = time.perf_counter()
    try:
        # quantize_chunk_aware returns an nn.Module, not a dict
        q_model = quantize_chunk_aware(student, target_bits=float(quant_bits), action_head_bits=8)
        timings["quantize_s"] = round(time.perf_counter() - t0, 1)

        # Compute size stats from model parameters
        fp32_size_mb = sum(p.numel() * 4 for p in student.parameters()) / (1024 * 1024)
        q_size_mb = fp32_size_mb * (quant_bits / 32.0)
        n_quantized = sum(
            1
            for n, m in q_model.named_modules()
            if isinstance(m, torch.nn.Linear) and "vision_encoder" not in n and "lora" not in n.lower()
        )

        result["quantization"] = {
            "target_bits": quant_bits,
            "fp32_size_mb": round(fp32_size_mb, 1),
            "estimated_packed_size_mb": round(q_size_mb, 1),
            "estimated_weight_compression_ratio": round(32.0 / quant_bits, 1),
            "quantization_storage": "fake-quantized-fp32-for-quality-analysis",
            "n_modules_quantized": n_quantized,
        }
        student = q_model  # Use quantized model for inference benchmark
        print(
            f"  Quantize: INT{quant_bits}, estimated {result['quantization']['estimated_weight_compression_ratio']}x "
            f"compression, {n_quantized} modules"
        )
    except Exception as e:
        result["quantization"] = {"error": str(e)[:100]}
        timings["quantize_s"] = round(time.perf_counter() - t0, 1)
        print(f"  Quantize: ERROR - {str(e)[:80]}")

    # INFERENCE BENCHMARK
    student.eval()
    torch.cuda.reset_peak_memory_stats()

    # FP32
    times = []
    with torch.no_grad():
        for _ in range(5):
            student(image_b1)
        for _ in range(20):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = student(image_b1)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    fp32_arr = np.array(times[3:])

    # FP16
    times16 = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        for _ in range(5):
            student(image_b1)
        for _ in range(20):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = student(image_b1)
            torch.cuda.synchronize()
            times16.append((time.perf_counter() - t0) * 1000)

    fp16_arr = np.array(times16[3:])

    result["inference"] = {
        "fp32_p50_ms": round(float(np.percentile(fp32_arr, 50)), 2),
        "fp32_fps": round(float(1000 / fp32_arr.mean()), 1),
        "fp16_p50_ms": round(float(np.percentile(fp16_arr, 50)), 2),
        "fp16_fps": round(float(1000 / fp16_arr.mean()), 1),
        "fp16_speedup": round(float(fp32_arr.mean() / fp16_arr.mean()), 2),
        "gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
    }
    print(
        f"  Infer: FP32={fp32_arr.mean():.1f}ms ({1000 / fp32_arr.mean():.1f}fps), "
        f"FP16={fp16_arr.mean():.1f}ms ({1000 / fp16_arr.mean():.1f}fps)"
    )

    result["timings"] = timings
    total_time = sum(timings.values())
    result["total_time_s"] = round(total_time, 1)
    print(f"  Total pipeline: {total_time:.1f}s")

    del student
    torch.cuda.empty_cache()

    return result


def main():
    if DEVICE == "cpu":
        print("SKIP: No CUDA")
        sys.exit(0)

    dataset = load_real_dataset(MODEL_DIR, max_samples=120)

    print("=== Full Pipeline Combinations ===")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # (name, overrides, prune_ratio, quant_bits, train_batch)
    pipelines = [
        # 1. Nano + diffusion, standard prune/quant (baseline)
        (
            "nano_diff_p75_q4",
            {
                "variant": "nano",
                "action_head_type": "diffusion",
            },
            0.75,
            4,
            4,
        ),
        # 2. Nano + flow, aggressive prune
        (
            "nano_flow_p50_q4",
            {"variant": "nano", "action_head_type": "flow"},
            0.50,
            4,
            4,
        ),
        # 3. Nano + high LoRA, light prune
        (
            "nano_lora64_p90_q4",
            {
                "variant": "nano",
                "lora_rank": 64,
                "action_head_type": "diffusion",
            },
            0.90,
            4,
            4,
        ),
        # 4. Nano INT8 (higher quality quant)
        (
            "nano_diff_p75_q8",
            {
                "variant": "nano",
                "action_head_type": "diffusion",
            },
            0.75,
            8,
            4,
        ),
        # 5. Nano flow + high LoRA + aggressive prune
        (
            "nano_flow_lora64_p60_q4",
            {"variant": "nano", "lora_rank": 64, "action_head_type": "flow"},
            0.60,
            4,
            4,
        ),
        # 6. Nano diffusion + no prune + INT8 (quality ceiling)
        (
            "nano_diff_noprune_q8",
            {
                "variant": "nano",
                "action_head_type": "diffusion",
            },
            1.0,
            8,
            4,
        ),
    ]

    all_results = {}
    for name, overrides, prune, quant, batch in pipelines:
        try:
            all_results[name] = run_pipeline(dataset, name, overrides, prune, quant, train_batch=batch)
        except Exception as e:
            print(f"  PIPELINE ERROR: {e}")
            all_results[name] = {"error": str(e)[:200]}
            torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 100)
    print(f"{'Pipeline':<25} {'Params':>7} {'FP32fps':>8} {'FP16fps':>8} {'Loss↓':>7} {'Compress':>9} {'Time':>6}")
    print("-" * 100)
    for name, r in all_results.items():
        if "error" in r:
            print(f"{name:<25} ERROR")
            continue
        comp = r.get("quantization", {}).get("estimated_weight_compression_ratio", "N/A")
        comp_str = f"{comp}x" if isinstance(comp, (int, float)) else comp
        print(
            f"{name:<25} {r['model']['total_params_m']:>5.1f}M "
            f"{r['inference']['fp32_fps']:>6.1f} "
            f"{r['inference']['fp16_fps']:>6.1f} "
            f"{r['training']['loss_reduction_pct']:>5.1f}% "
            f"{comp_str:>8} "
            f"{r['total_time_s']:>4.0f}s"
        )

    output = {
        "benchmark": "full_pipeline_combos",
        "timestamp": datetime.now(UTC).isoformat(),
        "device": DEVICE,
        "gpu": torch.cuda.get_device_name(0),
        "n_gpus": torch.cuda.device_count(),
        "random_seed": BENCHMARK_SEED,
        "data_provenance": data_provenance(dataset),
        "pipelines": all_results,
    }

    out_path = RESULTS_DIR / "bench_12_full_pipeline_combos.json"
    write_json_artifact(out_path, output)

    print(f"\nResults saved to {out_path}")
    print("BENCH 12: DONE")


if __name__ == "__main__":
    main()
