"""PRD-31: Automated Hyperparameter Optimization via Optuna.

Replaces manual 6-config testing with intelligent automated search.
Uses Bayesian optimization (TPE) to explore FORGE's hyperparameter space
and MedianPruner to kill bad trials early, saving ~50% GPU time.

Usage:
    from forge.auto_hyperparam import run_auto_search

    results = run_auto_search(
        objective="balanced",
        n_trials=30,
        train_steps=100,
        device="cuda",
    )

CLI:
    forge hyperparam auto --trials 30 --objective balanced --steps 100
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Sized
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch

from forge.benchmark.suites.real_data import BENCHMARK_SEED, fixed_action_loss, reset_benchmark_rng
from forge.training_safety import backward_with_finite_gradients

if TYPE_CHECKING:
    import optuna
    from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# ── Scoring Functions ─────────────────────────────────────────


def score_balanced(metrics: dict[str, float]) -> float:
    """Balanced score: fps * 3 + loss_reduction * 2 + compression * 5 + (1000/params) * 1."""
    fps = metrics.get("fps", 0)
    loss_pct = metrics.get("loss_reduction_pct", 0)
    compression = metrics.get("compression_ratio", 1)
    params_m = metrics.get("total_params_m", 1000)
    return fps * 3.0 + loss_pct * 2.0 + compression * 5.0 + (1000 / max(params_m, 1)) * 1.0


def score_speed(metrics: dict[str, float]) -> float:
    """Speed-focused score: fps * 10 + loss_reduction * 0.1."""
    return metrics.get("fps", 0) * 10.0 + metrics.get("loss_reduction_pct", 0) * 0.1


def score_quality(metrics: dict[str, float]) -> float:
    """Quality-focused score: loss_reduction * 10 + fps * 0.5."""
    return metrics.get("loss_reduction_pct", 0) * 10.0 + metrics.get("fps", 0) * 0.5


def score_size(metrics: dict[str, float]) -> float:
    """Size-focused score: compression * 20 + (1000/params) * 10."""
    compression = metrics.get("compression_ratio", 1)
    params_m = metrics.get("total_params_m", 1000)
    return compression * 20.0 + (1000 / max(params_m, 1)) * 10.0


SCORE_FUNCTIONS = {
    "balanced": score_balanced,
    "speed": score_speed,
    "quality": score_quality,
    "size": score_size,
}


# ── Trial Result ──────────────────────────────────────────────


@dataclass
class AutoTrialResult:
    """Result from one automated trial."""

    trial_number: int
    params: dict[str, Any]
    score: float
    metrics: dict[str, float]
    status: str  # "completed", "pruned", "failed"
    duration_s: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Study Creation ────────────────────────────────────────────


def create_forge_study(
    objective: str = "balanced",
    pruner: str = "median",
    storage: str | None = None,
    study_name: str = "forge_auto_hp",
    random_seed: int = BENCHMARK_SEED,
) -> optuna.Study:
    """Create an Optuna study pre-configured for FORGE.

    Args:
        objective: Scoring objective (balanced, speed, quality, size)
        pruner: Pruning strategy ("median" or "hyperband")
        storage: SQLite URL for persistence (e.g., "sqlite:///study.db")
        study_name: Name for the study

    Returns:
        Configured Optuna study (maximize direction)
    """
    import optuna

    # Suppress Optuna's verbose logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Select pruner
    optuna_pruner: Any
    if pruner == "hyperband":
        optuna_pruner = optuna.pruners.HyperbandPruner(
            min_resource=10,
            max_resource=200,
            reduction_factor=3,
        )
    else:
        optuna_pruner = optuna.pruners.MedianPruner(
            n_startup_trials=3,
            n_warmup_steps=10,
        )

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",  # All score functions: higher = better
        pruner=optuna_pruner,
        sampler=optuna.samplers.TPESampler(seed=random_seed),
        storage=storage,
        load_if_exists=True,
    )

    return study


def suggest_forge_params(trial: optuna.Trial) -> dict[str, Any]:
    """Suggest hyperparameters from FORGE's search space.

    Uses Optuna's trial API for Bayesian-guided suggestions.
    """
    return {
        "lora_rank": trial.suggest_categorical("lora_rank", [16, 32, 64, 128]),
        "action_head_type": trial.suggest_categorical("action_head_type", ["diffusion", "flow"]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
        "prune_keep_ratio": trial.suggest_float("prune_keep_ratio", 0.5, 1.0),
        "quant_bits": trial.suggest_categorical("quant_bits", [4, 8]),
        "batch_size": trial.suggest_categorical("batch_size", [4, 8, 16]),
        "bridge_n_queries": trial.suggest_categorical("bridge_n_queries", [32, 64, 128]),
        "bridge_n_layers": trial.suggest_int("bridge_n_layers", 1, 4),
        "flow_inference_steps": trial.suggest_int("flow_inference_steps", 1, 8),
    }


def _build_trial_config(params: dict[str, Any], *, action_dim: int | None = None):
    """Apply search parameters on top of the canonical v3 nano preset."""
    from forge.config import ForgeConfig, apply_student_variant

    config = ForgeConfig.default()
    apply_student_variant(config.student, "nano")
    config.student.lora_rank = params["lora_rank"]
    config.student.action_head_type = params["action_head_type"]
    config.student.flow_inference_steps = params["flow_inference_steps"]
    config.student.bridge_n_queries = params["bridge_n_queries"]
    config.student.bridge_n_layers = params["bridge_n_layers"]
    config.distill.learning_rate = params["learning_rate"]
    config.distill.batch_size = params["batch_size"]
    if action_dim is not None:
        config.student.action_dim = action_dim
    return config


# ── Objective Function ────────────────────────────────────────


def forge_objective(
    trial: optuna.Trial,
    objective: str = "balanced",
    model_dir: str | Path | None = None,
    device: str = "cuda",
    train_steps: int = 100,
    report_every: int = 10,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    dataset: Dataset[dict[str, Any]] | None = None,
    allow_mock: bool = False,
    random_seed: int = BENCHMARK_SEED,
) -> float:
    """Optuna objective wrapping FORGE build → train → eval pipeline.

    This is the function Optuna calls for each trial. It:
    1. Suggests hyperparameters via TPE sampler
    2. Builds a FORGEStudent with those params
    3. Trains for `train_steps` steps with intermediate pruning
    4. Evaluates FPS and loss reduction
    5. Returns composite score

    Args:
        trial: Optuna trial object
        objective: Scoring objective
        model_dir: Path to model weights
        device: CUDA device
        train_steps: Training steps per trial
        report_every: Report intermediate values every N steps (for pruning)

    Returns:
        Composite score (higher = better)
    """
    import optuna

    from forge.student import FORGEStudent

    trial_seed = reset_benchmark_rng(random_seed + trial.number)

    # 1. Suggest params
    params = suggest_forge_params(trial)
    logger.info(f"Trial {trial.number}: {params}")

    # W&B per-trial run (additive — local JSON logging still happens)
    _wandb_run = None
    if wandb_project:
        try:
            import wandb

            _wandb_run = wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                group=f"auto_hp_{objective}",
                name=f"trial_{trial.number}",
                config=params,
                reinit=True,
            )
        except Exception as e:
            logger.warning(f"W&B init failed for trial {trial.number}: {e}")

    if dataset is None and not allow_mock:
        raise ValueError("Real training data is required for auto-HP; pass data_dir or explicitly enable allow_mock")

    action_dim: int | None = None
    if dataset is not None:
        action_dim = getattr(dataset, "action_dim", None)
        if action_dim is None:
            sample = dataset[0]
            action_dim = int(sample["ground_truth_actions"].shape[-1])

    # 2. Configure
    config = _build_trial_config(params, action_dim=action_dim)

    # Resolve model_dir
    model_dir_path = Path(model_dir or os.environ.get("FORGE_MODEL_DIR", "./models"))

    # 3. Build model
    try:
        student = FORGEStudent(config.student, model_dir=str(model_dir_path)).to(device)
    except Exception as e:
        logger.warning(f"Trial {trial.number} build failed: {e}")
        if _wandb_run is not None:
            _wandb_run.finish()
        raise optuna.TrialPruned(f"Build failed: {e}")

    total_params_m = student.total_params / 1e6

    try:
        # 4. Train with intermediate pruning
        student.train()
        optimizer = torch.optim.AdamW(
            student.trainable_parameters(),
            lr=params["learning_rate"],
            weight_decay=0.01,
        )

        from torch.utils.data import DataLoader

        image_size = 384
        batch_size = min(params["batch_size"], 4)  # Cap at 4 for memory
        action_dim = config.student.action_dim
        data_iterator = None
        if dataset is not None:
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
            data_iterator = iter(loader)

        initial_loss = None
        last_loss = None
        evaluation_loss_before = (
            fixed_action_loss(
                student,
                dataset,
                device,
                n_batches=3,
                action_dim=action_dim,
            )
            if dataset is not None
            else None
        )

        for step in range(1, train_steps + 1):
            if data_iterator is None:
                images = torch.randn(batch_size, 3, image_size, image_size, device=device)
                gt_actions = torch.randn(batch_size, action_dim, device=device)
            else:
                try:
                    batch = next(data_iterator)
                except StopIteration:
                    data_iterator = iter(loader)
                    batch = next(data_iterator)
                images = batch["image"].to(device)
                gt_actions = batch["ground_truth_actions"].to(device)

            out = student(images, gt_actions=gt_actions)
            loss = out["loss"]
            backward_with_finite_gradients(loss, student.trainable_parameters())
            optimizer.step()
            optimizer.zero_grad()

            loss_val = loss.item()
            if initial_loss is None:
                initial_loss = loss_val
            last_loss = loss_val

            # Report intermediate value for pruning
            if step % report_every == 0:
                current_evaluation_loss = (
                    fixed_action_loss(
                        student,
                        dataset,
                        device,
                        n_batches=3,
                        action_dim=action_dim,
                    )
                    if dataset is not None
                    else last_loss
                )
                baseline_loss = evaluation_loss_before if evaluation_loss_before is not None else initial_loss
                reduction_pct = (1 - current_evaluation_loss / max(baseline_loss, 1e-8)) * 100
                trial.report(reduction_pct, step)

                if _wandb_run is not None:
                    import wandb

                    wandb.log(
                        {
                            "training_loss": loss_val,
                            "evaluation_loss": current_evaluation_loss,
                            "loss_reduction_pct": reduction_pct,
                        },
                        step=step,
                    )

                if trial.should_prune():
                    logger.info(f"Trial {trial.number} pruned at step {step}")
                    if _wandb_run is not None:
                        import wandb

                        wandb.log({"pruned": True, "pruned_at_step": step}, step=step)
                    raise optuna.TrialPruned()

        # 5. Evaluate
        student.eval()
        if initial_loss is None or last_loss is None:
            raise RuntimeError("Auto-HP trial completed no training steps")
        evaluation_loss_after = (
            fixed_action_loss(
                student,
                dataset,
                device,
                n_batches=3,
                action_dim=action_dim,
            )
            if dataset is not None
            else last_loss
        )
        baseline_loss = evaluation_loss_before if evaluation_loss_before is not None else initial_loss
        loss_reduction_pct = (1 - evaluation_loss_after / max(baseline_loss, 1e-8)) * 100

        # FPS benchmark (5 warmup + 20 timed) on a real observation when available.
        times = []
        if dataset is not None:
            dataset_size = len(cast(Sized, dataset))
            eval_image = dataset[trial.number % dataset_size]["image"].unsqueeze(0).to(device)
        else:
            eval_image = torch.randn(1, 3, image_size, image_size, device=device)
        with torch.no_grad():
            for _ in range(5):
                student(eval_image)

            for _ in range(20):
                if device.startswith("cuda"):
                    torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                student(eval_image)
                if device.startswith("cuda"):
                    torch.cuda.synchronize(device)
                times.append((time.perf_counter() - t0) * 1000)

        import numpy as np

        fps = float(1000 / np.mean(times)) if times else 0

        # Compression estimate
        estimated_compression = 32.0 / params["quant_bits"]

        # 6. Score
        metrics: dict[str, Any] = {
            "fps": round(fps, 1),
            "loss_reduction_pct": round(loss_reduction_pct, 1),
            "loss_metric": ("fixed-real-evaluation-mean" if dataset is not None else "synthetic-training-endpoints"),
            "evaluation_batches": 3 if dataset is not None else 0,
            "evaluation_loss_before": (
                round(evaluation_loss_before, 6) if evaluation_loss_before is not None else None
            ),
            "evaluation_loss_after": round(evaluation_loss_after, 6),
            "training_loss_first": round(initial_loss, 6),
            "training_loss_last": round(last_loss, 6),
            "estimated_weight_compression_ratio": estimated_compression,
            "total_params_m": round(total_params_m, 1),
            "real_training_samples": len(cast(Sized, dataset)) if dataset is not None else 0,
            "random_seed": trial_seed,
        }

        score_fn = SCORE_FUNCTIONS.get(objective, score_balanced)
        scoring_metrics = {
            "fps": float(metrics["fps"]),
            "loss_reduction_pct": float(metrics["loss_reduction_pct"]),
            "compression_ratio": estimated_compression,
            "total_params_m": float(metrics["total_params_m"]),
        }
        score = score_fn(scoring_metrics)

        # Store metrics on trial
        trial.set_user_attr("metrics", metrics)
        trial.set_user_attr("params", params)

        logger.info(
            f"Trial {trial.number}: score={score:.1f}, fps={fps:.1f}, "
            f"loss↓={loss_reduction_pct:.1f}%, estimated weight compression={estimated_compression}x"
        )

        # W&B: log final metrics
        if _wandb_run is not None:
            import wandb

            wandb.log({"score": score, **metrics}, step=train_steps)

        return score

    finally:
        # Always finish W&B run and cleanup GPU
        if _wandb_run is not None:
            _wandb_run.finish()
        del student
        torch.cuda.empty_cache()


# ── Run Full Search ───────────────────────────────────────────


def run_auto_search(
    objective: str = "balanced",
    n_trials: int = 30,
    train_steps: int = 100,
    device: str = "cuda",
    model_dir: str | None = None,
    output_dir: str = "./outputs/auto_hp",
    pruner: str = "median",
    storage: str | None = None,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    data_dir: str | Path | None = None,
    allow_mock: bool = False,
    random_seed: int = BENCHMARK_SEED,
) -> dict[str, Any]:
    """Run full automated hyperparameter search.

    Args:
        objective: Scoring objective (balanced, speed, quality, size)
        n_trials: Number of trials to run
        train_steps: Training steps per trial
        device: Device (cuda/cpu)
        model_dir: Model weights directory
        output_dir: Output directory for results
        pruner: Pruning strategy
        storage: SQLite URL for persistence

    Returns:
        Summary dict with best params, all trials, and statistics.
    """
    import optuna

    dataset = None
    data_provenance: dict[str, Any]
    if data_dir is not None:
        from forge.data.lerobot_video_dataset import LeRobotVideoActionDataset

        dataset = LeRobotVideoActionDataset(data_dir, max_samples=max(2_000, train_steps * 4))
        data_provenance = dict(dataset.provenance)
    elif allow_mock:
        data_provenance = {"kind": "mock", "format": "random-tensors", "explicit_opt_in": True}
    else:
        raise ValueError(
            "Auto-HP requires a real LeRobot dataset. Pass --data-dir PATH, or use "
            "--allow-mock only for an explicit test workflow."
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Auto-configure SQLite storage if not provided
    if storage is None:
        db_path = output_path / "study.db"
        storage = f"sqlite:///{db_path}"

    study = create_forge_study(
        objective=objective,
        pruner=pruner,
        storage=storage,
        random_seed=random_seed,
    )

    t0 = time.time()

    # W&B sweep-level run (groups all per-trial runs)
    _sweep_run = None
    if wandb_project:
        try:
            import wandb

            _sweep_run = wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=f"auto_hp_{objective}",
                config={
                    "objective": objective,
                    "n_trials": n_trials,
                    "train_steps": train_steps,
                    "device": device,
                    "pruner": pruner,
                },
                job_type="sweep",
            )
            _sweep_run.finish()  # Finish sweep-level run before trial runs start
        except Exception as e:
            logger.warning(f"W&B sweep init failed: {e}")

    # Run optimization
    study.optimize(
        lambda trial: forge_objective(
            trial,
            objective=objective,
            model_dir=model_dir,
            device=device,
            train_steps=train_steps,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            dataset=dataset,
            allow_mock=allow_mock,
            random_seed=random_seed,
        ),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    total_time = time.time() - t0

    # Collect results
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]

    # Build result
    result: dict[str, Any] = {
        "objective": objective,
        "n_trials": n_trials,
        "train_steps": train_steps,
        "device": device,
        "total_time_s": round(total_time, 1),
        "completed": len(completed),
        "pruned": len(pruned),
        "failed": len(failed),
        "gpu_time_saved_pct": round(len(pruned) / max(len(study.trials), 1) * 100, 1),
        "data_provenance": data_provenance,
        "random_seed": random_seed,
    }

    if completed:
        best = study.best_trial
        if best.value is None:
            raise RuntimeError("Optuna best trial has no objective value")
        result["best_trial"] = {
            "number": best.number,
            "score": round(best.value, 2),
            "params": best.params,
            "metrics": best.user_attrs.get("metrics", {}),
        }

    # All trials
    trials_data = []
    for t in study.trials:
        trial_info: dict[str, Any] = {
            "number": t.number,
            "state": t.state.name,
            "params": t.params,
        }
        if t.value is not None:
            trial_info["score"] = round(t.value, 2)
        if t.user_attrs.get("metrics"):
            trial_info["metrics"] = t.user_attrs["metrics"]
        trials_data.append(trial_info)

    result["trials"] = trials_data

    # W&B project URL (if used)
    if wandb_project:
        result["wandb_project"] = wandb_project
        result["wandb_entity"] = wandb_entity

    # Save results
    results_path = output_path / "auto_hp_results.json"
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    logger.info(f"Auto HP search complete: {len(completed)} completed, {len(pruned)} pruned")
    logger.info(f"Results saved to {results_path}")

    # Connect to HyperparamSearch (PRD-27) format
    _sync_to_hyperparam_search(study, output_path / "trials.json", objective)

    return result


def _sync_to_hyperparam_search(
    study: optuna.Study,
    trials_path: Path,
    objective: str,
) -> None:
    """Sync Optuna results to HyperparamSearch trial format for compatibility."""
    import optuna

    from forge.hyperparam import Trial as ForgeTrial

    trials = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        forge_trial = ForgeTrial(
            trial_id=f"auto_{t.number:04d}",
            params=t.params,
            status="completed",
            objective_value=t.value,
            metrics=t.user_attrs.get("metrics", {}),
        )
        trials.append(forge_trial.to_dict())

    data = {
        "source": "auto_hyperparam",
        "objective": objective,
        "trials": trials,
    }

    trials_path.parent.mkdir(parents=True, exist_ok=True)
    trials_path.write_text(json.dumps(data, indent=2, default=str))


def export_best_yaml(study: optuna.Study, output_path: str | Path) -> Path:
    """Export best trial config as YAML file."""
    import yaml

    output_path = Path(output_path)

    if not study.best_trial:
        raise ValueError("No completed trials in study")

    best = study.best_trial
    config = {
        "student": {
            "variant": "nano",
            "lora_rank": best.params.get("lora_rank", 64),
            "action_head_type": best.params.get("action_head_type", "flow"),
            "flow_inference_steps": best.params.get("flow_inference_steps", 4),
            "bridge_n_queries": best.params.get("bridge_n_queries", 64),
            "bridge_n_layers": best.params.get("bridge_n_layers", 4),
        },
        "distill": {
            "learning_rate": best.params.get("learning_rate", 2e-4),
            "batch_size": best.params.get("batch_size", 8),
        },
        "quant": {
            "target_avg_bits": float(best.params.get("quant_bits", 4)),
        },
        "pruning": {
            "keep_ratio": best.params.get("prune_keep_ratio", 0.6),
        },
        "auto_hp": {
            "score": round(best.value, 2) if best.value else None,
            "trial_number": best.number,
            "metrics": best.user_attrs.get("metrics", {}),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return output_path


def get_search_summary(output_dir: str | Path = "./outputs/auto_hp") -> dict[str, Any] | None:
    """Load and return summary from a previous auto HP search."""
    results_path = Path(output_dir) / "auto_hp_results.json"
    if not results_path.exists():
        return None
    return json.loads(results_path.read_text())
