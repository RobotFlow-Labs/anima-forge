"""Benchmark 15: 400-Trial Auto Hyperparameter Search (4x L4 GPUs).

Runs 100 trials per GPU, each with a different objective:
  GPU 0: balanced  (fps + loss + compression + params)
  GPU 1: speed     (maximize FPS)
  GPU 2: quality   (maximize loss reduction)
  GPU 3: size      (maximize compression + minimize params)

Total: 400 trials, ~40 min wall clock (4 GPUs parallel).
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from forge.benchmark.artifacts import write_json_artifact
from forge.benchmark.suites.real_data import BENCHMARK_SEED, reset_benchmark_rng
from forge.benchmark.suites.runtime import export_dir, results_dir

RESULTS_DIR = results_dir()

MODEL_DIR = Path(os.environ.get("FORGE_MODEL_DIR", "./models"))
DATASET_DIR = Path(os.environ.get("FORGE_BENCHMARK_DATA_DIR", MODEL_DIR / "datasets" / "lerobot--pusht"))


def run_search_on_gpu(
    gpu_id: int,
    objective: str,
    n_trials: int,
    train_steps: int,
    output_dir: str,
) -> None:
    """Run Optuna search on a specific GPU."""
    import torch

    random_seed = reset_benchmark_rng(BENCHMARK_SEED + gpu_id)
    torch.cuda.set_device(gpu_id)

    from forge.auto_hyperparam import run_auto_search

    print(f"[GPU {gpu_id}] Starting {n_trials} trials | objective={objective} | steps={train_steps}")

    result = run_auto_search(
        objective=objective,
        n_trials=n_trials,
        train_steps=train_steps,
        device=f"cuda:{gpu_id}",
        model_dir=str(MODEL_DIR),
        output_dir=output_dir,
        pruner="median",
        data_dir=DATASET_DIR,
        random_seed=random_seed,
    )

    print(
        f"[GPU {gpu_id}] Done: {result['completed']} completed, "
        f"{result['pruned']} pruned ({result['gpu_time_saved_pct']}% saved), "
        f"{result['total_time_s']:.0f}s"
    )

    if result.get("best_trial"):
        best = result["best_trial"]
        m = best.get("metrics", {})
        print(
            f"[GPU {gpu_id}] Best #{best['number']}: score={best['score']}, "
            f"fps={m.get('fps')}, loss↓={m.get('loss_reduction_pct')}%"
        )


def main():
    import torch

    n_gpus = torch.cuda.device_count()
    if n_gpus < 4:
        print(f"SKIP: 400-trial benchmark requires 4 CUDA GPUs; found {n_gpus}")
        sys.exit(0)
    if not DATASET_DIR.is_dir():
        raise FileNotFoundError(f"Required real LeRobot dataset not found: {DATASET_DIR}")

    n_trials = 100
    train_steps = 50

    print("=== 400-Trial Auto HP Search ===")
    print(f"GPUs: {n_gpus}x {torch.cuda.get_device_name(0)}")
    print(f"Trials per GPU: {n_trials}")
    print(f"Steps per trial: {train_steps}")
    print(f"Total trials: {n_gpus * n_trials}")
    print()

    objectives = ["balanced", "speed", "quality", "size"]
    gpu_assignments = list(range(4))

    t0 = time.time()
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    scratch_root = export_dir() / "auto_hp_400" / run_id
    scratch_root.mkdir(parents=True, exist_ok=False)

    # Launch parallel processes
    ctx = mp.get_context("spawn")
    processes = []
    for gpu_id, objective in zip(gpu_assignments, objectives):
        objective_dir = scratch_root / objective
        p = ctx.Process(
            target=run_search_on_gpu,
            args=(gpu_id, objective, n_trials, train_steps, str(objective_dir)),
        )
        p.start()
        processes.append((p, gpu_id, objective))
        print(f"Launched GPU {gpu_id} → {objective}")

    # Wait for all
    results = {}
    for p, gpu_id, objective in processes:
        p.join()
        print(f"GPU {gpu_id} ({objective}) exited with code {p.exitcode}")

        result_path = scratch_root / objective / "auto_hp_results.json"
        if p.exitcode != 0:
            raise RuntimeError(f"GPU {gpu_id} ({objective}) search failed with exit code {p.exitcode}")
        if not result_path.is_file():
            raise RuntimeError(f"GPU {gpu_id} ({objective}) wrote no fresh result: {result_path}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        observed_trials = int(result.get("completed", 0)) + int(result.get("pruned", 0)) + int(result.get("failed", 0))
        if observed_trials != n_trials:
            raise RuntimeError(f"GPU {gpu_id} ({objective}) produced {observed_trials}/{n_trials} trials")
        if int(result.get("failed", 0)):
            raise RuntimeError(f"GPU {gpu_id} ({objective}) recorded {result['failed']} failed trials")
        if result.get("data_provenance", {}).get("kind") != "real":
            raise RuntimeError(f"GPU {gpu_id} ({objective}) did not use real training data")
        results[objective] = result

    total_time = time.time() - t0

    # Aggregate results
    print()
    print("=" * 80)
    print(f"400-TRIAL SEARCH COMPLETE — {total_time:.0f}s ({total_time / 60:.1f} min)")
    print("=" * 80)

    combined = {
        "benchmark": "auto_hp_400",
        "timestamp": datetime.now(UTC).isoformat(),
        "n_gpus": n_gpus,
        "gpu": torch.cuda.get_device_name(0),
        "random_seed_base": BENCHMARK_SEED,
        "total_trials": sum(r.get("completed", 0) + r.get("pruned", 0) + r.get("failed", 0) for r in results.values()),
        "total_completed": sum(r.get("completed", 0) for r in results.values()),
        "total_pruned": sum(r.get("pruned", 0) for r in results.values()),
        "total_failed": sum(r.get("failed", 0) for r in results.values()),
        "total_time_s": round(total_time, 1),
        "dataset": DATASET_DIR.name.replace("--", "/"),
        "data_provenance": {"kind": "real", "format": "lerobot-v3-video"},
        "objectives": {},
    }

    for objective in objectives:
        r = results.get(objective, {})
        best = r.get("best_trial", {})
        combined["objectives"][objective] = {
            "completed": r.get("completed", 0),
            "pruned": r.get("pruned", 0),
            "gpu_time_saved_pct": r.get("gpu_time_saved_pct", 0),
            "best_score": best.get("score"),
            "best_params": best.get("params"),
            "best_metrics": best.get("metrics"),
        }

        if best:
            m = best.get("metrics", {})
            p = best.get("params", {})
            print(f"\n  [{objective.upper()}] Best trial #{best.get('number', '?')}:")
            print(f"    Score: {best.get('score')}")
            print(
                f"    LoRA: {p.get('lora_rank')} | Head: {p.get('action_head_type')} | "
                f"LR: {p.get('learning_rate', 0):.6f}"
            )
            print(
                f"    FPS: {m.get('fps')} | Loss↓: {m.get('loss_reduction_pct')}% | "
                f"Estimated weight compression: {m.get('estimated_weight_compression_ratio')}x | "
                f"Params: {m.get('total_params_m')}M"
            )

    # Save combined results
    if combined["total_trials"] != 400 or combined["total_failed"] != 0:
        raise RuntimeError(
            f"400-trial acceptance failed: total={combined['total_trials']}, failed={combined['total_failed']}"
        )
    out_path = RESULTS_DIR / "bench_15_auto_hp_400.json"
    write_json_artifact(out_path, combined)

    print(f"\nResults saved to {out_path}")
    print("BENCH 15: DONE")


if __name__ == "__main__":
    main()
