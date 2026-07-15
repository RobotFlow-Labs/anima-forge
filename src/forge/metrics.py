"""PRD-24: Training Metrics & Monitoring.

Structured metrics collection, logging backends, and training visualization
for the FORGE production training pipeline.

Components:
1. MetricsCollector — collects and aggregates training metrics
2. MetricsLogger — abstract base for logging backends
3. JSONLogger — file-based JSON lines logging
4. TensorBoardLogger — TensorBoard integration (optional)
5. ConsoleLogger — rich terminal output
6. MetricsAggregator — windowed statistics (mean, min, max, std)
7. TrainingMonitor — combines collector + loggers for ProductionTrainer
"""

from __future__ import annotations

import json
import logging
import math
import time
from abc import ABC, abstractmethod
from collections import deque
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── Metrics Aggregator ───────────────────────────────────────────


class MetricsAggregator:
    """Windowed statistics over a stream of values.

    Tracks running mean, min, max, and standard deviation over
    a configurable window of recent values.
    """

    def __init__(self, window: int = 100):
        self.window = window
        self._values: deque[float] = deque(maxlen=window)
        self._total_count = 0
        self._running_sum = 0.0

    def update(self, value: float) -> None:
        self._values.append(value)
        self._total_count += 1
        self._running_sum += value

    @property
    def count(self) -> int:
        return len(self._values)

    @property
    def total_count(self) -> int:
        return self._total_count

    @property
    def mean(self) -> float:
        if not self._values:
            return 0.0
        return sum(self._values) / len(self._values)

    @property
    def global_mean(self) -> float:
        if self._total_count == 0:
            return 0.0
        return self._running_sum / self._total_count

    @property
    def min(self) -> float:
        if not self._values:
            return 0.0
        return min(self._values)

    @property
    def max(self) -> float:
        if not self._values:
            return 0.0
        return max(self._values)

    @property
    def std(self) -> float:
        if len(self._values) < 2:
            return 0.0
        m = self.mean
        variance = sum((v - m) ** 2 for v in self._values) / (len(self._values) - 1)
        return math.sqrt(variance)

    @property
    def last(self) -> float:
        if not self._values:
            return 0.0
        return self._values[-1]

    def summary(self) -> dict[str, float]:
        return {
            "mean": round(self.mean, 6),
            "min": round(self.min, 6),
            "max": round(self.max, 6),
            "std": round(self.std, 6),
            "last": round(self.last, 6),
            "count": self.count,
        }


# ── Metrics Collector ────────────────────────────────────────────


class MetricsCollector:
    """Collects and aggregates training metrics by name.

    Each metric name gets its own MetricsAggregator for windowed stats.
    """

    def __init__(self, window: int = 100):
        self.window = window
        self._aggregators: dict[str, MetricsAggregator] = {}
        self._step = 0
        self._start_time = time.time()

    def record(self, name: str, value: float, step: int | None = None) -> None:
        """Record a metric value."""
        if name not in self._aggregators:
            self._aggregators[name] = MetricsAggregator(window=self.window)
        self._aggregators[name].update(value)
        if step is not None:
            self._step = step

    def record_dict(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Record multiple metrics at once."""
        for name, value in metrics.items():
            self.record(name, value, step)

    def get(self, name: str) -> MetricsAggregator | None:
        return self._aggregators.get(name)

    def get_summary(self, name: str) -> dict[str, float]:
        agg = self._aggregators.get(name)
        if agg is None:
            return {}
        return agg.summary()

    def get_all_summaries(self) -> dict[str, dict[str, float]]:
        return {name: agg.summary() for name, agg in self._aggregators.items()}

    def get_snapshot(self, step: int | None = None) -> dict[str, Any]:
        """Get a point-in-time snapshot of all metrics."""
        elapsed = time.time() - self._start_time
        snapshot: dict[str, Any] = {
            "step": step or self._step,
            "elapsed_seconds": round(elapsed, 1),
        }
        for name, agg in self._aggregators.items():
            snapshot[name] = agg.last
        return snapshot

    @property
    def metric_names(self) -> list[str]:
        return list(self._aggregators.keys())

    def reset(self) -> None:
        self._aggregators.clear()
        self._step = 0
        self._start_time = time.time()


# ── Logger Backends ──────────────────────────────────────────────


class MetricsLogger(ABC):
    """Abstract base for metrics logging backends."""

    @abstractmethod
    def log(self, metrics: dict[str, Any], step: int) -> None:
        """Log metrics at a given step."""

    @abstractmethod
    def close(self) -> None:
        """Clean up resources."""


class JSONLogger(MetricsLogger):
    """Logs metrics as JSON lines to a file."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "a")

    def log(self, metrics: dict[str, Any], step: int) -> None:
        record = {"step": step, **metrics}
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    @staticmethod
    def load(path: str | Path) -> list[dict[str, Any]]:
        """Load all records from a JSON lines file."""
        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records


class ConsoleLogger(MetricsLogger):
    """Logs metrics to console via Python logging."""

    def __init__(self, prefix: str = "FORGE"):
        self.prefix = prefix

    def log(self, metrics: dict[str, Any], step: int) -> None:
        parts = [f"[{self.prefix}] Step {step}"]
        for k, v in metrics.items():
            if isinstance(v, float):
                parts.append(f"{k}={v:.4f}")
            else:
                parts.append(f"{k}={v}")
        logger.info(" | ".join(parts))

    def close(self) -> None:
        pass


class TensorBoardLogger(MetricsLogger):
    """TensorBoard logging backend (optional dependency)."""

    def __init__(self, log_dir: str | Path):
        self.log_dir = str(log_dir)
        self._writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter

            self._writer = SummaryWriter(log_dir=self.log_dir)
        except ImportError:
            logger.warning("TensorBoard not available. Install with: pip install tensorboard")

    def log(self, metrics: dict[str, Any], step: int) -> None:
        if self._writer is None:
            return
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                self._writer.add_scalar(k, v, step)
        self._writer.flush()

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()

    @property
    def available(self) -> bool:
        return self._writer is not None


class WandBLogger(MetricsLogger):
    """Weights & Biases logging backend (optional dependency).

    Adds W&B as an *additional* logger alongside JSON, Console, and
    TensorBoard — existing backends are never replaced.
    """

    def __init__(
        self,
        project: str = "forge",
        entity: str | None = None,
        run_name: str | None = None,
        config: dict | None = None,
        tags: list[str] | None = None,
        run: Any | None = None,
    ):
        self._run = None
        self._owns_run = False
        try:
            import wandb

            if run is not None:
                self._run = run
                self._owns_run = False
            else:
                self._run = wandb.init(
                    project=project,
                    entity=entity,
                    name=run_name,
                    config=config,
                    tags=tags,
                    reinit=True,
                )
                self._owns_run = True
        except ImportError:
            logger.warning("wandb not available. Install with: uv add wandb")
        except Exception:
            logger.warning("wandb init failed", exc_info=True)

    def log(self, metrics: dict[str, Any], step: int) -> None:
        if self._run is None:
            return
        import wandb

        wandb.log(metrics, step=step)

    def close(self) -> None:
        if self._run is not None and self._owns_run:
            self._run.finish()

    @property
    def available(self) -> bool:
        return self._run is not None

    @property
    def url(self) -> str | None:
        if self._run is not None:
            return self._run.get_url()
        return None


# ── Training Monitor ─────────────────────────────────────────────


class TrainingMonitor:
    """Combines MetricsCollector with multiple logging backends.

    Designed to plug into ProductionTrainer for seamless monitoring.
    """

    def __init__(
        self,
        log_dir: str | Path = "./logs",
        window: int = 100,
        log_every: int = 100,
        use_tensorboard: bool = False,
        use_json: bool = True,
        use_console: bool = True,
        use_wandb: bool = False,
        wandb_project: str = "forge",
        wandb_entity: str | None = None,
        wandb_config: dict | None = None,
        wandb_tags: list[str] | None = None,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_every = log_every
        self.collector = MetricsCollector(window=window)

        self._loggers: list[MetricsLogger] = []
        if use_console:
            self._loggers.append(ConsoleLogger())
        if use_json:
            self._loggers.append(JSONLogger(self.log_dir / "metrics.jsonl"))
        if use_tensorboard:
            self._loggers.append(TensorBoardLogger(self.log_dir / "tensorboard"))
        if use_wandb:
            wb = WandBLogger(
                project=wandb_project,
                entity=wandb_entity,
                config=wandb_config,
                tags=wandb_tags,
            )
            if wb.available:
                self._loggers.append(wb)

    def record(self, metrics: dict[str, float], step: int) -> None:
        """Record metrics and optionally log them."""
        self.collector.record_dict(metrics, step=step)

        if step % self.log_every == 0:
            # Build log payload with latest values
            snapshot = self.collector.get_snapshot(step)
            for lgr in self._loggers:
                lgr.log(snapshot, step)

    def record_scalar(self, name: str, value: float, step: int) -> None:
        """Record a single scalar metric."""
        self.collector.record(name, value, step=step)

    def get_summary(self) -> dict[str, dict[str, float]]:
        """Get windowed summaries for all metrics."""
        return self.collector.get_all_summaries()

    def get_training_curves(self) -> dict[str, Any]:
        """Get data suitable for plotting training curves."""
        curves: dict[str, Any] = {}
        for name in self.collector.metric_names:
            agg = self.collector.get(name)
            if agg is not None:
                curves[name] = {
                    "last": agg.last,
                    "mean": agg.mean,
                    "global_mean": agg.global_mean,
                    "min": agg.min,
                    "max": agg.max,
                    "total_count": agg.total_count,
                }
        return curves

    @property
    def wandb_url(self) -> str | None:
        """Return W&B run URL if available."""
        for lgr in self._loggers:
            if isinstance(lgr, WandBLogger):
                return lgr.url
        return None

    def close(self) -> None:
        """Clean up all loggers."""
        for lgr in self._loggers:
            lgr.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
