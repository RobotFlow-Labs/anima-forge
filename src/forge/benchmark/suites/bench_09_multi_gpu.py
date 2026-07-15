"""Benchmark 09: Multi-GPU — explicit per-GPU training + inference replicas."""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

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
N_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0
EXECUTION_BACKEND = "explicit-per-gpu-replicas-no-nccl"


def _synchronize(n_gpu: int) -> None:
    for device_id in range(n_gpu):
        torch.cuda.synchronize(device_id)


def _student_replicas(student_cls, config, model_dir: Path, n_gpu: int, *, train: bool) -> list[torch.nn.Module]:
    replicas: list[torch.nn.Module] = []
    for device_id in range(n_gpu):
        model = student_cls(config.student, model_dir=str(model_dir)).to(f"cuda:{device_id}")
        if train:
            model.train()
        else:
            model.eval()
        replicas.append(model)
    return replicas


def _batch_shards(dataset, batch_size: int, n_gpu: int, *, start: int = 0, action_dim: int | None = None):
    base = batch_size // n_gpu
    extra = batch_size % n_gpu
    shards = []
    offset = start
    for device_id in range(n_gpu):
        shard_size = base + (1 if device_id < extra else 0)
        if shard_size < 1:
            continue
        images, actions = real_batch(
            dataset,
            shard_size,
            f"cuda:{device_id}",
            start=offset,
            action_dim=action_dim,
        )
        shards.append((device_id, images, actions))
        offset += shard_size
    return shards


def _parallel_call(fn, items: list[Any]) -> list[Any]:
    if len(items) == 1:
        return [fn(items[0])]
    with ThreadPoolExecutor(max_workers=len(items)) as pool:
        futures = [pool.submit(fn, item) for item in items]
        return [future.result() for future in futures]


def _forward_replicas(models: list[torch.nn.Module], shards, *, gt_actions: bool = False):
    def run(item):
        device_id, images, actions = item
        if gt_actions:
            return models[device_id](images, gt_actions=actions)
        return models[device_id](images)

    return _parallel_call(run, list(shards))


def _mean_fixed_action_loss(
    models: list[torch.nn.Module],
    dataset,
    *,
    batch_size: int,
    action_dim: int | None = None,
) -> float:
    losses: list[float] = []
    was_training = [model.training for model in models]
    for model in models:
        model.eval()
    with torch.no_grad():
        for batch_index in range(5):
            shards = _batch_shards(
                dataset,
                batch_size,
                len(models),
                start=batch_index * batch_size,
                action_dim=action_dim,
            )
            outputs = _forward_replicas(models, shards, gt_actions=True)
            for out in outputs:
                loss = out["loss"] if isinstance(out["loss"], torch.Tensor) else torch.as_tensor(out["loss"])
                losses.append(float(loss.detach().mean().item()))
    for model, training in zip(models, was_training, strict=True):
        model.train(training)
    return sum(losses) / len(losses)


def bench_multi_gpu_inference(student_cls, config_cls, model_dir, dataset):
    """Benchmark single vs multi-GPU inference."""
    from forge.config import ForgeConfig

    results = {}

    for n_gpu in [1, 2, 4]:
        if n_gpu > N_GPUS:
            continue

        label = f"gpu_{n_gpu}"
        print(f"\n=== Inference: {n_gpu} GPU(s) ===")

        reset_benchmark_rng()
        config = ForgeConfig.default()
        models = _student_replicas(student_cls, config, model_dir, n_gpu, train=False)

        # Warmup
        with torch.no_grad():
            for _ in range(5):
                shards = _batch_shards(dataset, n_gpu, n_gpu)
                _forward_replicas(models, shards)

        _synchronize(n_gpu)

        # Benchmark different batch sizes
        batch_results = {}
        for batch_size in [1, 4, 8, 16]:
            times = []
            with torch.no_grad():
                for _ in range(30):
                    shards = _batch_shards(dataset, batch_size, n_gpu)
                    _synchronize(n_gpu)
                    t0 = time.perf_counter()
                    _forward_replicas(models, shards)
                    _synchronize(n_gpu)
                    times.append((time.perf_counter() - t0) * 1000)

            ta = np.array(times[5:])  # skip first 5 warmup
            batch_results[f"batch_{batch_size}"] = {
                "p50_ms": round(float(np.percentile(ta, 50)), 2),
                "p95_ms": round(float(np.percentile(ta, 95)), 2),
                "mean_ms": round(float(ta.mean()), 2),
                "fps": round(float(batch_size * 1000 / ta.mean()), 1),
                "per_sample_ms": round(float(ta.mean() / batch_size), 2),
            }
            print(f"  batch={batch_size}: {ta.mean():.1f}ms total, {batch_size * 1000 / ta.mean():.1f} fps")

        # GPU memory per device
        mem = {}
        for i in range(n_gpu):
            mem[f"gpu_{i}_allocated_gb"] = round(torch.cuda.memory_allocated(i) / 1024**3, 2)
            mem[f"gpu_{i}_reserved_gb"] = round(torch.cuda.memory_reserved(i) / 1024**3, 2)

        results[label] = {
            "n_gpus": n_gpu,
            "execution_backend": EXECUTION_BACKEND,
            "batch_results": batch_results,
            "memory": mem,
        }

        del models
        torch.cuda.empty_cache()
        for i in range(N_GPUS):
            torch.cuda.reset_peak_memory_stats(i)

    return results


def bench_multi_gpu_training(student_cls, config_cls, model_dir, dataset):
    """Benchmark single vs multi-GPU training throughput."""
    from forge.config import ForgeConfig

    results = {}

    for n_gpu in [1, 2, 4]:
        if n_gpu > N_GPUS:
            continue

        label = f"gpu_{n_gpu}"
        print(f"\n=== Training: {n_gpu} GPU(s) ===")

        reset_benchmark_rng()
        config = ForgeConfig.default()
        config.student.action_dim = dataset.action_dim
        models = _student_replicas(student_cls, config, model_dir, n_gpu, train=True)

        # Training loop
        n_steps = 30
        batch_size = n_gpu * 2  # scale batch with GPUs
        optimizers = [torch.optim.AdamW(cast(Any, model).trainable_parameters(), lr=2e-4) for model in models]
        if n_gpu == 1:
            evaluation_loss_before = fixed_action_loss(models[0], dataset, "cuda:0", batch_size=batch_size)
        else:
            evaluation_loss_before = _mean_fixed_action_loss(models, dataset, batch_size=batch_size)

        losses = []
        step_times = []

        def train_replica(item):
            model, optimizer, images, gt_actions = item
            out = model(images, gt_actions=gt_actions)
            loss = (
                out["loss"] if isinstance(out["loss"], torch.Tensor) and out["loss"].dim() == 0 else out["loss"].mean()
            )
            optimizer.zero_grad()
            backward_with_finite_gradients(loss, model.parameters())
            optimizer.step()
            return float(loss.detach().item())

        for step in range(n_steps):
            shards = _batch_shards(dataset, batch_size, n_gpu, start=step * batch_size)
            train_items = [
                (models[device_id], optimizers[device_id], images, actions) for device_id, images, actions in shards
            ]

            _synchronize(n_gpu)
            t0 = time.perf_counter()

            replica_losses = _parallel_call(train_replica, train_items)
            loss_value = sum(replica_losses) / len(replica_losses)

            _synchronize(n_gpu)
            step_time = (time.perf_counter() - t0) * 1000
            step_times.append(step_time)
            losses.append(loss_value)

            if step % 10 == 0:
                print(f"  Step {step}: loss={loss_value:.4f}, time={step_time:.0f}ms")

        if n_gpu == 1:
            evaluation_loss_after = fixed_action_loss(models[0], dataset, "cuda:0", batch_size=batch_size)
        else:
            evaluation_loss_after = _mean_fixed_action_loss(models, dataset, batch_size=batch_size)
        evaluation_loss_reduction_pct = (
            (evaluation_loss_before - evaluation_loss_after) / max(evaluation_loss_before, 1e-12) * 100
        )
        ta = np.array(step_times[5:])  # skip warmup
        la = np.array(losses)

        # Peak memory per device
        mem = {}
        for i in range(n_gpu):
            mem[f"gpu_{i}_peak_gb"] = round(torch.cuda.max_memory_allocated(i) / 1024**3, 2)

        results[label] = {
            "n_gpus": n_gpu,
            "execution_backend": EXECUTION_BACKEND,
            "batch_size": batch_size,
            "n_steps": n_steps,
            "step_time_mean_ms": round(float(ta.mean()), 1),
            "steps_per_sec": round(float(1000 / ta.mean()), 2),
            "samples_per_sec": round(float(batch_size * 1000 / ta.mean()), 2),
            "loss_metric": "fixed-real-evaluation-mean",
            "evaluation_batches": 5,
            "evaluation_loss_before": round(float(evaluation_loss_before), 4),
            "evaluation_loss_after": round(float(evaluation_loss_after), 4),
            "loss_reduction_pct": round(float(evaluation_loss_reduction_pct), 1),
            "training_loss_first": round(float(la[0]), 4),
            "training_loss_last": round(float(la[-1]), 4),
            "memory": mem,
        }

        del models, optimizers
        torch.cuda.empty_cache()
        for i in range(N_GPUS):
            torch.cuda.reset_peak_memory_stats(i)

    return results


def bench_fp16_multi_gpu(student_cls, config_cls, model_dir, dataset):
    """FP16 autocast inference across multiple GPUs."""
    from forge.config import ForgeConfig

    results = {}

    for n_gpu in [1, 4]:
        if n_gpu > N_GPUS:
            continue

        label = f"fp16_gpu_{n_gpu}"
        print(f"\n=== FP16 Inference: {n_gpu} GPU(s) ===")

        reset_benchmark_rng()
        config = ForgeConfig.default()
        models = _student_replicas(student_cls, config, model_dir, n_gpu, train=False)

        batch_results = {}
        for batch_size in [4, 8, 16, 32]:
            times = []
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
                # Warmup
                for _ in range(5):
                    shards = _batch_shards(dataset, min(4, batch_size), n_gpu)
                    _forward_replicas(models, shards)

                for _ in range(25):
                    shards = _batch_shards(dataset, batch_size, n_gpu)
                    _synchronize(n_gpu)
                    t0 = time.perf_counter()
                    _forward_replicas(models, shards)
                    _synchronize(n_gpu)
                    times.append((time.perf_counter() - t0) * 1000)

            ta = np.array(times[3:])
            batch_results[f"batch_{batch_size}"] = {
                "p50_ms": round(float(np.percentile(ta, 50)), 2),
                "fps": round(float(batch_size * 1000 / ta.mean()), 1),
                "per_sample_ms": round(float(ta.mean() / batch_size), 2),
            }
            print(f"  FP16 batch={batch_size}: {ta.mean():.1f}ms, {batch_size * 1000 / ta.mean():.1f} fps")

        results[label] = {
            "n_gpus": n_gpu,
            "execution_backend": EXECUTION_BACKEND,
            "precision": "fp16",
            "batch_results": batch_results,
        }

        del models
        torch.cuda.empty_cache()

    return results


def main():
    from forge.config import ForgeConfig
    from forge.student import FORGEStudent

    if N_GPUS < 4:
        print(f"SKIP: Multi-GPU acceptance requires 4 CUDA GPUs; found {N_GPUS}")
        sys.exit(0)

    dataset = load_real_dataset(MODEL_DIR, max_samples=32)

    print(f"=== Multi-GPU Benchmark: {N_GPUS} GPUs ===")
    for i in range(N_GPUS):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    inference_results = bench_multi_gpu_inference(FORGEStudent, ForgeConfig, MODEL_DIR, dataset)
    training_results = bench_multi_gpu_training(FORGEStudent, ForgeConfig, MODEL_DIR, dataset)
    fp16_results = bench_fp16_multi_gpu(FORGEStudent, ForgeConfig, MODEL_DIR, dataset)

    results = {
        "benchmark": "multi_gpu",
        "timestamp": datetime.now(UTC).isoformat(),
        "n_gpus_available": N_GPUS,
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(N_GPUS)],
        "random_seed": BENCHMARK_SEED,
        "data_provenance": data_provenance(dataset),
        "execution_backend": EXECUTION_BACKEND,
        "collectives": "none",
        "inference": inference_results,
        "training": training_results,
        "fp16": fp16_results,
    }

    out_path = RESULTS_DIR / "bench_09_multi_gpu.json"
    write_json_artifact(out_path, results)

    print(f"\nResults saved to {out_path}")
    print("BENCH 09: DONE")


if __name__ == "__main__":
    main()
