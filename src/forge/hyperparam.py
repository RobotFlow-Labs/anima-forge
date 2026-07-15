"""PRD-27: Hyperparameter Search & Optimization.

Systematic exploration of FORGE training hyperparameters with grid search,
random search, and early stopping. Integrates with ModelRegistry (PRD-26)
and TrainingMonitor (PRD-24).

Usage:
    from forge.hyperparam import HyperparamSearch, SearchSpace, Trial

    space = SearchSpace()
    space.add_categorical("variant", ["nano", "small"])
    space.add_choice("lora_rank", [16, 32, 64])
    space.add_range("learning_rate", 1e-5, 1e-3, log_scale=True)

    search = HyperparamSearch(space, objective="best_loss")
    trials = search.grid_search()  # or .random_search(n_trials=20)
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

logger = logging.getLogger(__name__)


# ── Search Space Definition ───────────────────────────────────


@dataclass
class ParamSpec:
    """Specification for a single hyperparameter."""

    name: str
    param_type: str  # "categorical", "choice", "range"
    values: list[Any] = field(default_factory=list)  # For categorical/choice
    low: float = 0.0  # For range
    high: float = 1.0  # For range
    log_scale: bool = False  # For range: sample in log space
    step: float | None = None  # For range: discrete steps


class SearchSpace:
    """Define the hyperparameter search space."""

    def __init__(self) -> None:
        self._params: dict[str, ParamSpec] = {}

    def add_categorical(self, name: str, values: list[Any]) -> SearchSpace:
        """Add a categorical parameter (tries all values in grid)."""
        self._params[name] = ParamSpec(name=name, param_type="categorical", values=values)
        return self

    def add_choice(self, name: str, values: list[Any]) -> SearchSpace:
        """Add a choice parameter (discrete set of values)."""
        self._params[name] = ParamSpec(name=name, param_type="choice", values=values)
        return self

    def add_range(
        self,
        name: str,
        low: float,
        high: float,
        log_scale: bool = False,
        step: float | None = None,
    ) -> SearchSpace:
        """Add a continuous range parameter."""
        self._params[name] = ParamSpec(
            name=name,
            param_type="range",
            low=low,
            high=high,
            log_scale=log_scale,
            step=step,
        )
        return self

    @property
    def params(self) -> dict[str, ParamSpec]:
        return dict(self._params)

    @property
    def param_names(self) -> list[str]:
        return list(self._params.keys())

    def grid_size(self) -> int:
        """Total number of grid combinations."""
        size = 1
        for spec in self._params.values():
            if spec.param_type in ("categorical", "choice"):
                size *= len(spec.values)
            elif spec.param_type == "range" and spec.step:
                n = int((spec.high - spec.low) / spec.step) + 1
                size *= n
            else:
                raise ValueError(f"Cannot compute grid size for range param '{spec.name}' without step")
        return size

    def sample_random(self, rng: random.Random | None = None) -> dict[str, Any]:
        """Sample a random point from the space."""
        rng = rng or random.Random()
        sample: dict[str, Any] = {}
        for name, spec in self._params.items():
            if spec.param_type in ("categorical", "choice"):
                sample[name] = rng.choice(spec.values)
            elif spec.param_type == "range":
                if spec.log_scale:
                    log_low = math.log(spec.low)
                    log_high = math.log(spec.high)
                    val = math.exp(rng.uniform(log_low, log_high))
                else:
                    val = rng.uniform(spec.low, spec.high)
                if spec.step:
                    val = round(val / spec.step) * spec.step
                sample[name] = val
        return sample

    def enumerate_grid(self) -> list[dict[str, Any]]:
        """Enumerate all grid points."""
        param_lists: list[list[tuple[str, Any]]] = []
        for name, spec in self._params.items():
            if spec.param_type in ("categorical", "choice"):
                param_lists.append([(name, v) for v in spec.values])
            elif spec.param_type == "range" and spec.step:
                vals = []
                v = spec.low
                while v <= spec.high + 1e-10:
                    vals.append((name, round(v, 10)))
                    v += spec.step
                param_lists.append(vals)
            else:
                raise ValueError(f"Grid search requires step for range param '{name}'")

        # Cartesian product
        from itertools import product as cart_product

        return [dict(combo) for combo in cart_product(*param_lists)]

    def to_dict(self) -> list[dict[str, Any]]:
        return [asdict(spec) for spec in self._params.values()]

    @classmethod
    def from_dict(cls, data: list[dict[str, Any]]) -> SearchSpace:
        space = cls()
        for spec_data in data:
            spec = ParamSpec(**spec_data)
            space._params[spec.name] = spec
        return space


# ── Trial ─────────────────────────────────────────────────────


@dataclass
class Trial:
    """Result of a single hyperparameter trial."""

    trial_id: str
    params: dict[str, Any]
    status: str = "pending"  # pending, running, completed, failed, pruned
    objective_value: float | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    start_time: float = 0.0
    end_time: float = 0.0
    error: str | None = None

    @property
    def duration_seconds(self) -> float:
        if self.end_time > 0 and self.start_time > 0:
            return self.end_time - self.start_time
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Trial:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def _trial_id(params: dict[str, Any]) -> str:
    raw = json.dumps(params, sort_keys=True, default=str)
    return hashlib.md5(raw.encode()).hexdigest()[:12]


# ── Hyperparameter Search ─────────────────────────────────────


class HyperparamSearch:
    """Systematic hyperparameter search with trial tracking.

    Supports grid search, random search, and manual trials.
    Results are persisted to JSON for analysis.
    """

    def __init__(
        self,
        space: SearchSpace,
        objective: str = "best_loss",
        lower_is_better: bool = True,
        results_dir: str | Path = "./outputs/hyperparam",
    ):
        self.space = space
        self.objective = objective
        self.lower_is_better = lower_is_better
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.trials: list[Trial] = []
        self._load()

    def _load(self) -> None:
        """Load existing trials from disk."""
        results_file = self.results_dir / "trials.json"
        if results_file.exists():
            try:
                data = json.loads(results_file.read_text())
                self.trials = [Trial.from_dict(t) for t in data.get("trials", [])]
                logger.debug(f"Loaded {len(self.trials)} existing trials")
            except (json.JSONDecodeError, OSError):
                pass

    def _save(self) -> None:
        """Persist trials to disk."""
        data = {
            "space": self.space.to_dict(),
            "objective": self.objective,
            "lower_is_better": self.lower_is_better,
            "trials": [t.to_dict() for t in self.trials],
        }
        (self.results_dir / "trials.json").write_text(json.dumps(data, indent=2))

    def create_trial(self, params: dict[str, Any]) -> Trial:
        """Create a new trial with given parameters."""
        trial = Trial(
            trial_id=_trial_id(params),
            params=params,
            status="pending",
        )
        self.trials.append(trial)
        self._save()
        return trial

    def start_trial(self, trial: Trial) -> Trial:
        """Mark trial as running."""
        trial.status = "running"
        trial.start_time = time.time()
        self._save()
        return trial

    def complete_trial(
        self,
        trial: Trial,
        objective_value: float,
        metrics: dict[str, float] | None = None,
    ) -> Trial:
        """Mark trial as completed with results."""
        trial.status = "completed"
        trial.objective_value = objective_value
        trial.metrics = metrics or {}
        trial.end_time = time.time()
        self._save()
        logger.info(
            f"Trial [{trial.trial_id}] completed: {self.objective}={objective_value:.6f}, params={trial.params}"
        )
        return trial

    def fail_trial(self, trial: Trial, error: str) -> Trial:
        """Mark trial as failed."""
        trial.status = "failed"
        trial.error = error
        trial.end_time = time.time()
        self._save()
        logger.warning(f"Trial [{trial.trial_id}] failed: {error}")
        return trial

    def grid_search(self) -> list[Trial]:
        """Generate all grid search trials."""
        grid = self.space.enumerate_grid()
        trials: list[Trial] = []
        for params in grid:
            trial = self.create_trial(params)
            trials.append(trial)
        logger.info(f"Grid search: {len(trials)} trials generated")
        return trials

    def random_search(self, n_trials: int = 20, seed: int | None = None) -> list[Trial]:
        """Generate random search trials."""
        rng = random.Random(seed)
        trials: list[Trial] = []
        seen: set[str] = set()
        attempts = 0
        max_attempts = n_trials * 10

        while len(trials) < n_trials and attempts < max_attempts:
            params = self.space.sample_random(rng)
            tid = _trial_id(params)
            if tid not in seen:
                seen.add(tid)
                trial = self.create_trial(params)
                trials.append(trial)
            attempts += 1

        logger.info(f"Random search: {len(trials)} trials generated")
        return trials

    def best_trial(self) -> Trial | None:
        """Find the best completed trial."""
        completed = [t for t in self.trials if t.status == "completed" and t.objective_value is not None]
        if not completed:
            return None
        if self.lower_is_better:
            return min(completed, key=lambda trial: cast(float, trial.objective_value))
        return max(completed, key=lambda trial: cast(float, trial.objective_value))

    def top_trials(self, n: int = 5) -> list[Trial]:
        """Get top N trials by objective value."""
        completed = [t for t in self.trials if t.status == "completed" and t.objective_value is not None]
        completed.sort(key=lambda trial: cast(float, trial.objective_value), reverse=not self.lower_is_better)
        return completed[:n]

    def summary(self) -> dict[str, Any]:
        """Summary statistics of the search."""
        completed = [t for t in self.trials if t.status == "completed"]
        failed = [t for t in self.trials if t.status == "failed"]
        pending = [t for t in self.trials if t.status == "pending"]

        result: dict[str, Any] = {
            "total_trials": len(self.trials),
            "completed": len(completed),
            "failed": len(failed),
            "pending": len(pending),
            "objective": self.objective,
            "lower_is_better": self.lower_is_better,
        }

        completed_with_values = [trial for trial in completed if trial.objective_value is not None]
        if completed_with_values:
            values = [cast(float, trial.objective_value) for trial in completed_with_values]
            result["best_value"] = min(values) if self.lower_is_better else max(values)
            result["worst_value"] = max(values) if self.lower_is_better else min(values)
            result["mean_value"] = sum(values) / len(values)

            best = self.best_trial()
            if best:
                result["best_params"] = best.params

        return result

    def apply_to_config(self, params: dict[str, Any], config: Any) -> Any:
        """Apply trial parameters to a ForgeConfig.

        Supports dotted keys like "distill.learning_rate" and
        flat keys that map to StudentConfig.
        """
        student_fields = {
            "variant",
            "vision_encoder",
            "language_model",
            "bridge_d_vision",
            "bridge_d_model",
            "bridge_n_queries",
            "bridge_n_heads",
            "bridge_n_layers",
            "action_dim",
            "action_head_type",
            "action_horizon",
            "lora_rank",
            "lora_alpha",
            "flow_inference_steps",
        }
        distill_fields = {
            "learning_rate",
            "weight_decay",
            "warmup_steps",
            "max_steps",
            "batch_size",
            "gradient_accumulation_steps",
            "temperature",
            "alpha_kd",
            "alpha_task",
            "alpha_feat",
            "alpha_action",
        }

        for key, value in params.items():
            if "." in key:
                # Dotted path: "distill.learning_rate"
                parts = key.split(".", 1)
                sub_config = getattr(config, parts[0], None)
                if sub_config and hasattr(sub_config, parts[1]):
                    setattr(sub_config, parts[1], value)
            elif key in student_fields and hasattr(config, "student"):
                setattr(config.student, key, value)
            elif key in distill_fields and hasattr(config, "distill"):
                setattr(config.distill, key, value)
            elif hasattr(config, key):
                setattr(config, key, value)

        return config

    @property
    def count(self) -> int:
        return len(self.trials)


# ── Benchmark-Driven Recommendations ──────────────────────────


def _load_bench_results(results_dir: Path) -> dict[str, Any]:
    """Load all benchmark JSON results from a directory."""
    results = {}
    for f in sorted(results_dir.glob("bench_*.json")):
        try:
            results[f.stem] = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
    return results


def _score_pipeline(pipeline: dict[str, Any], objective: str = "balanced") -> float:
    """Score a pipeline result. Higher = better.

    Objectives:
        balanced: weighted mix of speed, quality, compression
        speed: maximize FPS
        quality: maximize loss reduction
        size: minimize model size
    """
    training = pipeline.get("training", {})
    inference = pipeline.get("inference", {})
    quant = pipeline.get("quantization", {})
    model = pipeline.get("model", {})

    def metric(section: object, key: str, default: float) -> float:
        if not isinstance(section, dict):
            return default
        value = section.get(key, default)
        return float(value) if isinstance(value, int | float) else default

    loss_pct = (
        metric(training, "loss_reduction_pct", 0.0)
        if isinstance(training, dict) and training.get("loss_metric") == "fixed-real-evaluation-mean"
        else 0.0
    )
    fp16_fps = metric(inference, "fp16_fps", 0.0)
    compression = metric(quant, "compression_ratio", 1.0)
    params_m = metric(model, "total_params_m", 1000.0)

    if "error" in quant:
        compression = 1

    if objective == "speed":
        return fp16_fps * 10 + loss_pct * 0.1
    elif objective == "quality":
        return loss_pct * 10 + fp16_fps * 0.5
    elif objective == "size":
        return compression * 20 + (1000 / max(params_m, 1)) * 10
    else:  # balanced
        return fp16_fps * 3.0 + loss_pct * 2.0 + compression * 5.0 + (1000 / max(params_m, 1)) * 1.0


def recommend_config(
    results_dir: str | Path,
    objective: str = "balanced",
    top_n: int = 3,
) -> list[dict[str, Any]]:
    """Recommend hyperparameter configs based on benchmark results.

    Reads bench_12 (full pipeline combos) and bench_03 (training) results
    to rank configurations by the given objective.

    Args:
        results_dir: Path to the flat benchmark JSON artifact directory.
        objective: One of "balanced", "speed", "quality", "size"
        top_n: Number of recommendations to return

    Returns:
        List of recommendation dicts sorted best-first.
    """
    results_dir = Path(results_dir)
    bench_data = _load_bench_results(results_dir)

    recommendations = []

    # Primary: bench_12 full pipeline combos
    bench_12 = bench_data.get("bench_12_full_pipeline_combos", {})
    pipelines = bench_12.get("pipelines", {})

    for name, pipeline in pipelines.items():
        if "error" in pipeline and not isinstance(pipeline.get("error"), str):
            continue
        if not pipeline.get("training"):
            continue

        score = _score_pipeline(pipeline, objective)
        config = pipeline.get("config", {})
        training = pipeline.get("training", {})
        if not isinstance(training, dict) or training.get("loss_metric") != "fixed-real-evaluation-mean":
            continue
        inference = pipeline.get("inference", {})
        quant = pipeline.get("quantization", {})
        model = pipeline.get("model", {})
        pruning = pipeline.get("pruning", {})

        rec = {
            "rank": 0,
            "name": name,
            "score": round(score, 2),
            "config": config,
            "metrics": {
                "total_params_m": model.get("total_params_m"),
                "fp16_fps": inference.get("fp16_fps"),
                "fp32_fps": inference.get("fp32_fps"),
                "loss_reduction_pct": training.get("loss_reduction_pct"),
                "compression_ratio": quant.get("compression_ratio", "N/A"),
                "gpu_mem_gb": inference.get("gpu_mem_gb"),
            },
            "pruning": {
                "ratio": pruning.get("ratio"),
                "layers_removed": len(pruning.get("layers_removed", [])),
            }
            if not pruning.get("skipped")
            else {"skipped": True},
            "recommendation": "",
        }
        recommendations.append(rec)

    # Sort by score descending
    recommendations.sort(key=lambda r: r["score"], reverse=True)

    # Assign ranks and generate recommendation text
    for i, rec in enumerate(recommendations[:top_n]):
        rec["rank"] = i + 1
        cfg = rec["config"]
        m = rec["metrics"]

        if i == 0:
            rec["recommendation"] = (
                f"BEST {objective.upper()}: {cfg.get('action_head_type', '?')} head, "
                f"LoRA-{cfg.get('lora_rank', '?')}, "
                f"{m.get('fp16_fps', '?')} FPS FP16, "
                f"{m.get('loss_reduction_pct', '?')}% loss reduction"
            )
        else:
            rec["recommendation"] = (
                f"#{i + 1}: {cfg.get('action_head_type', '?')} head, "
                f"LoRA-{cfg.get('lora_rank', '?')}, "
                f"{m.get('fp16_fps', '?')} FPS"
            )

    # Also pull in training-specific insights from bench_03
    bench_03 = bench_data.get("bench_03_training", {})
    comparable_runs = [
        (key, run)
        for key, run in bench_03.items()
        if key.startswith("run_") and isinstance(run, dict) and run.get("loss_metric") == "fixed-real-evaluation-mean"
    ]
    comparable_runs.sort(key=lambda item: float(item[1].get("loss_reduction_pct", float("-inf"))), reverse=True)
    training_insight = None
    if comparable_runs and float(comparable_runs[0][1].get("loss_reduction_pct", 0)) > 85:
        key, run = comparable_runs[0]
        lr = next((part for part in key.split("_") if part.startswith("lr")), "unknown")
        training_insight = {
            "source": "bench_03",
            "best_lr": lr,
            "loss_reduction_pct": run["loss_reduction_pct"],
            "steps_per_sec": run.get("steps_per_sec"),
            "note": (
                f"LR {lr} achieved {run['loss_reduction_pct']}% fixed-real loss reduction "
                f"in {run.get('n_steps', '?')} steps"
            ),
        }

    result = recommendations[:top_n]
    if training_insight:
        for rec in result:
            rec["training_insight"] = training_insight

    return result
