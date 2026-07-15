"""PRD-29: Inference Telemetry & Real-time Monitoring.

Lightweight telemetry for FORGE inference pipelines.
Tracks latency percentiles, throughput, buffer health, and anomalies
without adding overhead to the critical path.

Usage:
    from forge.telemetry import InferenceTelemetry

    telemetry = InferenceTelemetry(window_size=1000)
    telemetry.record_inference(latency_ms=12.3)
    telemetry.record_action(action=np.array([0.1, 0.2, ...]))
    summary = telemetry.summary()
    # {'p50_ms': 12.3, 'p95_ms': 15.1, 'fps': 81.3, ...}
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────


@dataclass
class TelemetryConfig:
    """Configuration for inference telemetry."""

    window_size: int = 1000  # Rolling window for stats
    anomaly_threshold: float = 3.0  # Std devs for anomaly detection
    log_interval: int = 100  # Log summary every N inferences
    export_path: str = ""  # Path for JSON export (empty = no export)
    track_actions: bool = True  # Track action statistics
    track_latency: bool = True  # Track latency percentiles


# ── Latency Tracker ───────────────────────────────────────


class LatencyTracker:
    """Rolling window latency statistics.

    Tracks p50, p95, p99 latencies with O(1) append and O(n log n) query.
    """

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self._latencies: deque[float] = deque(maxlen=window_size)
        self._total_count: int = 0
        self._total_sum: float = 0.0

    def record(self, latency_ms: float) -> None:
        """Record a latency measurement in milliseconds."""
        self._latencies.append(latency_ms)
        self._total_count += 1
        self._total_sum += latency_ms

    def percentile(self, p: float) -> float:
        """Compute percentile (0-100) from current window."""
        if not self._latencies:
            return 0.0
        sorted_vals = sorted(self._latencies)
        idx = int(len(sorted_vals) * p / 100.0)
        idx = min(idx, len(sorted_vals) - 1)
        return sorted_vals[idx]

    @property
    def p50(self) -> float:
        return self.percentile(50)

    @property
    def p95(self) -> float:
        return self.percentile(95)

    @property
    def p99(self) -> float:
        return self.percentile(99)

    @property
    def mean(self) -> float:
        if not self._latencies:
            return 0.0
        return sum(self._latencies) / len(self._latencies)

    @property
    def count(self) -> int:
        return self._total_count

    @property
    def window_count(self) -> int:
        return len(self._latencies)

    def stats(self) -> dict[str, float]:
        """Full latency statistics."""
        return {
            "p50_ms": round(self.p50, 3),
            "p95_ms": round(self.p95, 3),
            "p99_ms": round(self.p99, 3),
            "mean_ms": round(self.mean, 3),
            "count": self._total_count,
            "window_count": self.window_count,
        }


# ── Throughput Tracker ────────────────────────────────────


class ThroughputTracker:
    """Measures inference throughput (fps) over rolling window."""

    def __init__(self, window_seconds: float = 10.0):
        self.window_seconds = window_seconds
        self._timestamps: deque[float] = deque()
        self._total_count: int = 0

    def tick(self) -> None:
        """Record an inference event."""
        now = time.monotonic()
        self._timestamps.append(now)
        self._total_count += 1
        # Prune old timestamps
        cutoff = now - self.window_seconds
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()

    @property
    def fps(self) -> float:
        """Current throughput in frames per second."""
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed

    @property
    def total_count(self) -> int:
        return self._total_count

    def stats(self) -> dict[str, float]:
        return {
            "fps": round(self.fps, 1),
            "total_inferences": self._total_count,
        }


# ── Action Monitor ────────────────────────────────────────


class ActionMonitor:
    """Tracks action statistics and detects anomalies.

    Monitors action magnitude, rate of change, and out-of-range values.
    """

    def __init__(self, window_size: int = 100, anomaly_threshold: float = 3.0):
        self.window_size = window_size
        self.anomaly_threshold = anomaly_threshold
        self._magnitudes: deque[float] = deque(maxlen=window_size)
        self._deltas: deque[float] = deque(maxlen=window_size)
        self._last_action: np.ndarray | None = None
        self._anomaly_count: int = 0
        self._total_count: int = 0

    def record(self, action: np.ndarray) -> dict[str, Any]:
        """Record an action and return anomaly info.

        Returns:
            Dict with 'is_anomaly', 'magnitude', 'delta' keys.
        """
        self._total_count += 1
        mag = float(np.linalg.norm(action))
        self._magnitudes.append(mag)

        delta = 0.0
        if self._last_action is not None:
            delta = float(np.linalg.norm(action - self._last_action))
            self._deltas.append(delta)
        self._last_action = action.copy()

        # Anomaly detection: magnitude exceeds threshold * std from mean
        is_anomaly = False
        if len(self._magnitudes) >= 10:
            mag_arr = np.array(self._magnitudes)
            mean = mag_arr.mean()
            std = mag_arr.std()
            if std > 0 and abs(mag - mean) > self.anomaly_threshold * std:
                is_anomaly = True
                self._anomaly_count += 1

        return {
            "is_anomaly": is_anomaly,
            "magnitude": round(mag, 4),
            "delta": round(delta, 4),
        }

    @property
    def anomaly_rate(self) -> float:
        if self._total_count == 0:
            return 0.0
        return self._anomaly_count / self._total_count

    def stats(self) -> dict[str, Any]:
        mags = np.array(self._magnitudes) if self._magnitudes else np.array([0.0])
        deltas = np.array(self._deltas) if self._deltas else np.array([0.0])
        return {
            "action_count": self._total_count,
            "anomaly_count": self._anomaly_count,
            "anomaly_rate": round(self.anomaly_rate, 4),
            "mean_magnitude": round(float(mags.mean()), 4),
            "std_magnitude": round(float(mags.std()), 4),
            "mean_delta": round(float(deltas.mean()), 4),
            "max_delta": round(float(deltas.max()), 4),
        }


# ── Buffer Health ─────────────────────────────────────────


class BufferHealthMonitor:
    """Monitors action buffer fill level and starvation events."""

    def __init__(self, max_buffer_size: int = 4):
        self.max_buffer_size = max_buffer_size
        self._fill_levels: deque[int] = deque(maxlen=1000)
        self._starvation_count: int = 0
        self._overflow_count: int = 0

    def record(self, current_fill: int) -> None:
        """Record buffer fill level."""
        self._fill_levels.append(current_fill)
        if current_fill == 0:
            self._starvation_count += 1
        if current_fill >= self.max_buffer_size:
            self._overflow_count += 1

    @property
    def mean_fill(self) -> float:
        if not self._fill_levels:
            return 0.0
        return sum(self._fill_levels) / len(self._fill_levels)

    def stats(self) -> dict[str, Any]:
        levels = list(self._fill_levels)
        return {
            "mean_fill": round(self.mean_fill, 2),
            "min_fill": min(levels) if levels else 0,
            "max_fill": max(levels) if levels else 0,
            "starvation_count": self._starvation_count,
            "overflow_count": self._overflow_count,
        }


# ── Main Telemetry ────────────────────────────────────────


class InferenceTelemetry:
    """Central telemetry collector for FORGE inference.

    Aggregates latency, throughput, action, and buffer metrics.
    Thread-safe for use with AsyncInferenceEngine.
    """

    def __init__(self, config: TelemetryConfig | None = None):
        self.config = config or TelemetryConfig()
        self.latency = LatencyTracker(self.config.window_size)
        self.throughput = ThroughputTracker()
        self.actions = ActionMonitor(
            window_size=self.config.window_size,
            anomaly_threshold=self.config.anomaly_threshold,
        )
        self.buffer = BufferHealthMonitor()
        self._start_time = time.monotonic()

    def record_inference(self, latency_ms: float) -> None:
        """Record a completed inference with its latency."""
        if self.config.track_latency:
            self.latency.record(latency_ms)
        self.throughput.tick()

        # Periodic logging
        if self.config.log_interval > 0 and self.throughput.total_count % self.config.log_interval == 0:
            logger.info(
                f"Telemetry: {self.throughput.total_count} inferences | "
                f"p50={self.latency.p50:.1f}ms p95={self.latency.p95:.1f}ms | "
                f"fps={self.throughput.fps:.1f}"
            )

    def record_action(self, action: np.ndarray) -> dict[str, Any]:
        """Record a produced action and return anomaly info."""
        if self.config.track_actions:
            return self.actions.record(action)
        return {"is_anomaly": False, "magnitude": 0.0, "delta": 0.0}

    def record_buffer(self, fill_level: int) -> None:
        """Record buffer fill level."""
        self.buffer.record(fill_level)

    @property
    def uptime_seconds(self) -> float:
        return time.monotonic() - self._start_time

    def summary(self) -> dict[str, Any]:
        """Full telemetry summary."""
        result: dict[str, Any] = {
            "uptime_seconds": round(self.uptime_seconds, 1),
        }
        result.update(self.latency.stats())
        result.update(self.throughput.stats())
        if self.config.track_actions:
            result["actions"] = self.actions.stats()
        result["buffer"] = self.buffer.stats()
        return result

    def export_json(self, path: str | Path | None = None) -> str:
        """Export telemetry to JSON file."""
        import json

        export_path = path or self.config.export_path
        if not export_path:
            return json.dumps(self.summary(), indent=2)
        p = Path(export_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(self.summary(), indent=2, default=str)
        p.write_text(data)
        logger.info(f"Telemetry exported to {p}")
        return data

    def reset(self) -> None:
        """Reset all counters."""
        self.latency = LatencyTracker(self.config.window_size)
        self.throughput = ThroughputTracker()
        self.actions = ActionMonitor(
            window_size=self.config.window_size,
            anomaly_threshold=self.config.anomaly_threshold,
        )
        self.buffer = BufferHealthMonitor()
        self._start_time = time.monotonic()
