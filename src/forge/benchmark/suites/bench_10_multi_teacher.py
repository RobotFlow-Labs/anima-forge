"""Benchmark 10: real multi-teacher inference, routing, and distillation."""

from __future__ import annotations

import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from forge.benchmark.artifacts import write_json_artifact
from forge.benchmark.suites.real_data import (
    BENCHMARK_SEED,
    data_provenance,
    load_real_dataset,
    real_batch,
    reset_benchmark_rng,
)
from forge.benchmark.suites.runtime import results_dir
from forge.training_safety import backward_with_finite_gradients

RESULTS_DIR = results_dir()
MODEL_DIR = Path(os.environ.get("FORGE_MODEL_DIR", "./models"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0
TEACHERS = [
    "molmoact2-libero",
    "openvla-7b",
    "rdt2-fm",
    "smolvla-base",
    "vla-jepa-3b",
]
ACTION_DIM = 7


def _fit_action_width(value: object, *, prediction_index: int, batch_size: int) -> torch.Tensor:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 3 or not len(array):
        raise ValueError(f"Teacher predictions must have shape (N,H,D), got {array.shape}")
    action = torch.from_numpy(array[prediction_index % len(array), 0])
    if action.shape[-1] > ACTION_DIM:
        action = action[:ACTION_DIM]
    elif action.shape[-1] < ACTION_DIM:
        action = torch.nn.functional.pad(action, (0, ACTION_DIM - action.shape[-1]))
    return action.unsqueeze(0).expand(batch_size, -1).to(DEVICE)


def _teacher_tensors(
    records: list[dict[str, Any]],
    *,
    prediction_index: int,
    batch_size: int,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    predictions = [
        _fit_action_width(record["prediction_actions"], prediction_index=prediction_index, batch_size=batch_size)
        for record in records
    ]
    confidences = [
        _fit_action_width(record["prediction_confidences"], prediction_index=prediction_index, batch_size=batch_size)
        for record in records
    ]
    return predictions, torch.stack(confidences, dim=1)


def collect_real_teacher_evidence() -> dict[str, Any]:
    """Run every required teacher on real robot frames and retain its outputs."""
    from forge.teacher_fleet import build_isolated_fleet_report

    benchmark_data_dir = os.environ.get("FORGE_BENCHMARK_DATA_DIR")
    teacher_dataset_root = os.environ.get("FORGE_TEACHER_DATASET_ROOT")
    if benchmark_data_dir:
        dataset_root = Path(benchmark_data_dir).expanduser().resolve().parent
    elif teacher_dataset_root:
        dataset_root = Path(teacher_dataset_root).expanduser().resolve()
    else:
        dataset_root = (MODEL_DIR / "datasets").expanduser().resolve()
    required = (dataset_root / "lerobot--pusht", dataset_root / "lerobot--aloha_sim_transfer_cube_human")
    missing = [str(path) for path in required if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"Required teacher verification datasets are missing: {', '.join(missing)}")
    report = build_isolated_fleet_report(
        teacher_names=TEACHERS,
        model_dir=MODEL_DIR,
        dataset_dir=dataset_root,
        gpu_ids=list(range(4)),
        predictions=4,
        include_predictions=True,
    )
    if not report.get("all_real") or report.get("teachers_verified") != len(TEACHERS):
        failures = [record for record in report["results"] if record.get("status") != "ok"]
        raise RuntimeError(f"Real teacher fleet verification failed: {failures}")
    return report


def teacher_evidence_summary(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Remove local paths while preserving measurable real-inference evidence."""
    metadata_keys = ("teacher", "architecture", "inference", "uncertainty")
    return [
        {
            "status": record["status"],
            "teacher": record["teacher"],
            "device": record["device"],
            "model_bytes": record["model_bytes"],
            "predictions": record["predictions"],
            "latency_ms": record["latency_ms"],
            "cuda_memory_bytes": record["cuda_memory_bytes"],
            "actions": record["actions"],
            "prediction_metadata": [
                {key: metadata[key] for key in metadata_keys if key in metadata}
                for metadata in record["prediction_metadata"]
            ],
        }
        for record in report["results"]
    ]


def evaluate_fixed_router_loss(
    student: torch.nn.Module,
    loss_fn: torch.nn.Module,
    dataset,
    records: list[dict[str, Any]],
    *,
    universal: bool = False,
    n_batches: int = 5,
) -> float:
    """Evaluate router loss on fixed real inputs and teacher predictions."""
    student_was_training = student.training
    loss_was_training = loss_fn.training
    student.eval()
    loss_fn.eval()
    losses: list[float] = []
    cuda_devices = [torch.cuda.current_device()] if DEVICE.startswith("cuda") else []
    with torch.random.fork_rng(devices=cuda_devices), torch.no_grad():
        torch.manual_seed(0)
        for batch_index in range(n_batches):
            images, ground_truth = real_batch(
                dataset,
                4,
                DEVICE,
                start=batch_index * 4,
                action_dim=ACTION_DIM,
            )
            teacher_actions, confidences = _teacher_tensors(
                records,
                prediction_index=batch_index,
                batch_size=4,
            )
            output = student(images, gt_actions=ground_truth)
            arguments = (
                output["actions"],
                teacher_actions,
                ground_truth,
                output["vision_features"].mean(dim=1),
            )
            loss = loss_fn(*arguments, confidences) if universal else loss_fn(*arguments)
            losses.append(float(loss["total"].item()))
    student.train(student_was_training)
    loss_fn.train(loss_was_training)
    return float(np.mean(losses))


def bench_multi_teacher_training(dataset, records: list[dict[str, Any]]) -> dict[str, Any]:
    """Train routing against genuine outputs from increasing teacher counts."""
    from forge.config import ForgeConfig
    from forge.multi_teacher import MultiTeacherDistillationLoss
    from forge.student import FORGEStudent

    results = {}
    for n_teachers in (1, 3, 5):
        reset_benchmark_rng()
        selected = records[:n_teachers]
        config = ForgeConfig.default()
        config.student.action_dim = ACTION_DIM
        student = FORGEStudent(config.student, model_dir=str(MODEL_DIR)).to(DEVICE)
        student.train()
        loss_fn = MultiTeacherDistillationLoss(
            n_teachers=n_teachers,
            d_student=config.student.bridge_d_model,
            temperature=4.0,
            alpha_task=0.3,
        ).to(DEVICE)
        optimized_parameters = [*student.trainable_parameters(), *loss_fn.parameters()]
        optimizer = torch.optim.AdamW(optimized_parameters, lr=2e-4)
        evaluation_loss_before = evaluate_fixed_router_loss(student, loss_fn, dataset, selected)
        losses: list[float] = []
        entropy: list[float] = []
        step_times: list[float] = []
        for step in range(30):
            images, ground_truth = real_batch(dataset, 4, DEVICE, start=step * 4, action_dim=ACTION_DIM)
            teacher_actions, _ = _teacher_tensors(selected, prediction_index=step, batch_size=4)
            torch.cuda.synchronize()
            started = time.perf_counter()
            output = student(images, gt_actions=ground_truth)
            features = output["vision_features"].mean(dim=1)
            loss = loss_fn(output["actions"], teacher_actions, ground_truth, features)
            optimizer.zero_grad()
            backward_with_finite_gradients(loss["total"], optimized_parameters)
            optimizer.step()
            torch.cuda.synchronize()
            step_times.append((time.perf_counter() - started) * 1_000)
            losses.append(loss["total"].item())
            weights = loss["router_weights"]
            entropy.append((-(weights * (weights + 1e-8).log()).sum(dim=-1).mean()).item())

        evaluation_loss_after = evaluate_fixed_router_loss(student, loss_fn, dataset, selected)
        evaluation_loss_reduction_pct = (
            (evaluation_loss_before - evaluation_loss_after) / max(evaluation_loss_before, 1e-12) * 100
        )
        timed = np.asarray(step_times[5:])
        curve = np.asarray(losses)
        results[f"{n_teachers}_teachers"] = {
            "teachers": [record["teacher"] for record in selected],
            "n_steps": len(losses),
            "training_loss_curve": [round(value, 6) for value in losses],
            "loss_metric": "fixed-real-router-evaluation-mean",
            "evaluation_batches": 5,
            "evaluation_loss_before": round(float(evaluation_loss_before), 6),
            "evaluation_loss_after": round(float(evaluation_loss_after), 6),
            "loss_reduction_pct": round(float(evaluation_loss_reduction_pct), 2),
            "training_loss_first": round(float(curve[0]), 6),
            "training_loss_last": round(float(curve[-1]), 6),
            "router_entropy_final": round(entropy[-1], 6),
            "step_time_mean_ms": round(float(timed.mean()), 2),
            "steps_per_second": round(float(1_000 / timed.mean()), 3),
        }
        del student, loss_fn, optimizer
        torch.cuda.empty_cache()
    return results


def bench_universal_distillation(dataset, records: list[dict[str, Any]]) -> dict[str, Any]:
    """Exercise confidence/diversity routing with genuine teacher outputs."""
    from forge.config import ForgeConfig
    from forge.student import FORGEStudent
    from forge.universal_distill import UniversalDistillationLoss

    results = {}
    configurations = [
        {"name": "balanced", "alpha_task": 0.3, "alpha_diversity": 0.05, "alpha_consistency": 0.1},
        {"name": "kd_heavy", "alpha_task": 0.1, "alpha_diversity": 0.05, "alpha_consistency": 0.05},
        {"name": "diverse", "alpha_task": 0.2, "alpha_diversity": 0.15, "alpha_consistency": 0.1},
    ]
    for parameters in configurations:
        reset_benchmark_rng()
        config = ForgeConfig.default()
        config.student.action_dim = ACTION_DIM
        student = FORGEStudent(config.student, model_dir=str(MODEL_DIR)).to(DEVICE)
        student.train()
        loss_fn = UniversalDistillationLoss(
            n_teachers=len(records),
            d_student=config.student.bridge_d_model,
            confidence_dim=ACTION_DIM,
            alpha_task=cast(float, parameters["alpha_task"]),
            alpha_diversity=cast(float, parameters["alpha_diversity"]),
            alpha_consistency=cast(float, parameters["alpha_consistency"]),
        ).to(DEVICE)
        optimized_parameters = [*student.trainable_parameters(), *loss_fn.parameters()]
        optimizer = torch.optim.AdamW(optimized_parameters, lr=2e-4)
        evaluation_loss_before = evaluate_fixed_router_loss(
            student,
            loss_fn,
            dataset,
            records,
            universal=True,
        )
        curves: dict[str, list[float]] = {key: [] for key in ("total", "kd", "task", "diversity", "consistency")}
        router_weights: list[float] = []
        for step in range(30):
            images, ground_truth = real_batch(dataset, 4, DEVICE, start=step * 4, action_dim=ACTION_DIM)
            teacher_actions, confidences = _teacher_tensors(records, prediction_index=step, batch_size=4)
            output = student(images, gt_actions=ground_truth)
            loss = loss_fn(
                output["actions"],
                teacher_actions,
                ground_truth,
                output["vision_features"].mean(dim=1),
                confidences,
            )
            optimizer.zero_grad()
            backward_with_finite_gradients(loss["total"], optimized_parameters)
            optimizer.step()
            for key in curves:
                curves[key].append(loss[key].item())
            router_weights = loss["router_weights"][0].detach().cpu().tolist()
        evaluation_loss_after = evaluate_fixed_router_loss(
            student,
            loss_fn,
            dataset,
            records,
            universal=True,
        )
        evaluation_loss_reduction_pct = (
            (evaluation_loss_before - evaluation_loss_after) / max(evaluation_loss_before, 1e-12) * 100
        )
        total = curves["total"]
        results[str(parameters["name"])] = {
            "config": parameters,
            "teachers": [record["teacher"] for record in records],
            "training_loss_curves": {key: [round(value, 6) for value in values] for key, values in curves.items()},
            "loss_metric": "fixed-real-router-evaluation-mean",
            "evaluation_batches": 5,
            "evaluation_loss_before": round(float(evaluation_loss_before), 6),
            "evaluation_loss_after": round(float(evaluation_loss_after), 6),
            "loss_reduction_pct": round(float(evaluation_loss_reduction_pct), 2),
            "training_loss_first": round(total[0], 6),
            "training_loss_last": round(total[-1], 6),
            "final_router_weights": [round(value, 6) for value in router_weights],
        }
        del student, loss_fn, optimizer
        torch.cuda.empty_cache()
    return results


def main() -> None:
    if N_GPUS < 4:
        print(f"SKIP: Real multi-teacher acceptance requires 4 CUDA GPUs; found {N_GPUS}")
        sys.exit(0)
    reset_benchmark_rng()
    dataset = load_real_dataset(MODEL_DIR, max_samples=120)
    report = collect_real_teacher_evidence()
    records = report["results"]
    results = {
        "benchmark": "multi_teacher",
        "timestamp": datetime.now(UTC).isoformat(),
        "device": DEVICE,
        "n_gpus": N_GPUS,
        "gpu_names": [torch.cuda.get_device_name(index) for index in range(N_GPUS)],
        "random_seed": BENCHMARK_SEED,
        "data_provenance": data_provenance(dataset),
        "all_teachers_real": True,
        "teacher_evidence": teacher_evidence_summary(report),
        "multi_teacher_training": bench_multi_teacher_training(dataset, records),
        "universal_distillation": bench_universal_distillation(dataset, records),
    }
    out_path = RESULTS_DIR / "bench_10_multi_teacher.json"
    write_json_artifact(out_path, results)
    print(f"Results saved to {out_path}")
    print("BENCH 10: DONE")


if __name__ == "__main__":
    main()
