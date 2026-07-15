"""Tests for PRD-29: Inference Telemetry & Real-time Monitoring."""

from __future__ import annotations

import json

import numpy as np
import pytest

from forge.telemetry import (
    ActionMonitor,
    BufferHealthMonitor,
    InferenceTelemetry,
    LatencyTracker,
    TelemetryConfig,
    ThroughputTracker,
)

# ── LatencyTracker ───────────────────────────────────────


class TestLatencyTracker:
    def test_empty(self):
        t = LatencyTracker()
        assert t.p50 == 0.0
        assert t.mean == 0.0
        assert t.count == 0

    def test_single_value(self):
        t = LatencyTracker()
        t.record(10.0)
        assert t.p50 == 10.0
        assert t.count == 1

    def test_percentiles(self):
        t = LatencyTracker(window_size=100)
        for i in range(100):
            t.record(float(i))
        assert t.p50 == pytest.approx(50.0, abs=1)
        assert t.p95 >= 90.0
        assert t.p99 >= 95.0

    def test_window_size_respected(self):
        t = LatencyTracker(window_size=10)
        for i in range(100):
            t.record(float(i))
        assert t.window_count == 10
        assert t.count == 100  # Total still tracked

    def test_mean(self):
        t = LatencyTracker()
        t.record(10.0)
        t.record(20.0)
        assert t.mean == 15.0

    def test_stats_dict(self):
        t = LatencyTracker()
        t.record(5.0)
        s = t.stats()
        assert "p50_ms" in s
        assert "p95_ms" in s
        assert "mean_ms" in s
        assert s["count"] == 1


# ── ThroughputTracker ────────────────────────────────────


class TestThroughputTracker:
    def test_empty(self):
        t = ThroughputTracker()
        assert t.fps == 0.0
        assert t.total_count == 0

    def test_single_tick(self):
        t = ThroughputTracker()
        t.tick()
        assert t.fps == 0.0  # Need at least 2 ticks
        assert t.total_count == 1

    def test_fps_calculation(self):
        t = ThroughputTracker(window_seconds=10.0)
        # Simulate rapid ticks
        for _ in range(10):
            t.tick()
        assert t.fps > 0
        assert t.total_count == 10

    def test_stats_dict(self):
        t = ThroughputTracker()
        t.tick()
        s = t.stats()
        assert "fps" in s
        assert "total_inferences" in s


# ── ActionMonitor ────────────────────────────────────────


class TestActionMonitor:
    def test_first_action(self):
        m = ActionMonitor()
        info = m.record(np.array([1.0, 0.0, 0.0]))
        assert info["magnitude"] > 0
        assert info["delta"] == 0.0
        assert info["is_anomaly"] is False

    def test_delta_tracking(self):
        m = ActionMonitor()
        m.record(np.array([1.0, 0.0, 0.0]))
        info = m.record(np.array([2.0, 0.0, 0.0]))
        assert info["delta"] == pytest.approx(1.0, abs=0.01)

    def test_anomaly_detection(self):
        m = ActionMonitor(window_size=50, anomaly_threshold=2.0)
        # Record normal actions
        for _ in range(50):
            m.record(np.random.randn(7) * 0.1)
        # Record extreme action
        info = m.record(np.ones(7) * 100.0)
        assert info["is_anomaly"] is True

    def test_no_anomaly_for_consistent(self):
        m = ActionMonitor(window_size=50, anomaly_threshold=3.0)
        for _ in range(50):
            m.record(np.ones(7) * 1.0)
        info = m.record(np.ones(7) * 1.0)
        assert info["is_anomaly"] is False

    def test_anomaly_rate(self):
        m = ActionMonitor(window_size=100, anomaly_threshold=2.0)
        for _ in range(100):
            m.record(np.random.randn(7) * 0.1)
        m.record(np.ones(7) * 100.0)
        assert m.anomaly_rate > 0

    def test_empty_anomaly_rate(self):
        m = ActionMonitor()
        assert m.anomaly_rate == 0.0

    def test_stats_dict(self):
        m = ActionMonitor()
        m.record(np.array([1.0, 0.0]))
        s = m.stats()
        assert "action_count" in s
        assert "anomaly_count" in s
        assert "mean_magnitude" in s


# ── BufferHealthMonitor ──────────────────────────────────


class TestBufferHealth:
    def test_empty(self):
        b = BufferHealthMonitor(max_buffer_size=4)
        assert b.mean_fill == 0.0

    def test_fill_tracking(self):
        b = BufferHealthMonitor(max_buffer_size=4)
        b.record(2)
        b.record(3)
        assert b.mean_fill == 2.5

    def test_starvation_detection(self):
        b = BufferHealthMonitor(max_buffer_size=4)
        b.record(0)
        b.record(2)
        b.record(0)
        s = b.stats()
        assert s["starvation_count"] == 2

    def test_overflow_detection(self):
        b = BufferHealthMonitor(max_buffer_size=4)
        b.record(4)
        b.record(2)
        b.record(4)
        s = b.stats()
        assert s["overflow_count"] == 2

    def test_stats_dict(self):
        b = BufferHealthMonitor()
        b.record(2)
        s = b.stats()
        assert "mean_fill" in s
        assert "min_fill" in s
        assert "max_fill" in s


# ── InferenceTelemetry ───────────────────────────────────


class TestInferenceTelemetry:
    def test_defaults(self):
        t = InferenceTelemetry()
        assert t.config.window_size == 1000
        assert t.uptime_seconds >= 0

    def test_record_inference(self):
        t = InferenceTelemetry()
        t.record_inference(12.5)
        assert t.latency.count == 1
        assert t.throughput.total_count == 1

    def test_record_action(self):
        t = InferenceTelemetry()
        info = t.record_action(np.array([0.1, 0.2, 0.3]))
        assert "is_anomaly" in info

    def test_record_buffer(self):
        t = InferenceTelemetry()
        t.record_buffer(3)
        assert t.buffer.mean_fill == 3.0

    def test_summary(self):
        t = InferenceTelemetry()
        for i in range(10):
            t.record_inference(float(i))
            t.record_action(np.random.randn(7))
            t.record_buffer(2)
        s = t.summary()
        assert "p50_ms" in s
        assert "fps" in s
        assert "actions" in s
        assert "buffer" in s
        assert "uptime_seconds" in s

    def test_export_json_string(self):
        t = InferenceTelemetry()
        t.record_inference(10.0)
        data = t.export_json()
        parsed = json.loads(data)
        assert parsed["count"] == 1

    def test_export_json_file(self, tmp_path):
        t = InferenceTelemetry()
        t.record_inference(10.0)
        path = tmp_path / "telemetry.json"
        t.export_json(path)
        assert path.exists()
        parsed = json.loads(path.read_text())
        assert parsed["count"] == 1

    def test_reset(self):
        t = InferenceTelemetry()
        t.record_inference(10.0)
        t.record_action(np.array([1.0]))
        t.reset()
        assert t.latency.count == 0
        assert t.throughput.total_count == 0

    def test_action_tracking_disabled(self):
        config = TelemetryConfig(track_actions=False)
        t = InferenceTelemetry(config)
        info = t.record_action(np.array([1.0]))
        assert info["is_anomaly"] is False

    def test_custom_config(self):
        config = TelemetryConfig(
            window_size=50,
            anomaly_threshold=2.0,
            log_interval=10,
        )
        t = InferenceTelemetry(config)
        assert t.config.window_size == 50


# ── Strict Edge Cases ────────────────────────────────────


class TestLatencyStrict:
    def test_window_evicts_old(self):
        t = LatencyTracker(window_size=5)
        for v in [100, 100, 100, 100, 100]:
            t.record(float(v))
        assert t.mean == 100.0
        for v in [1, 1, 1, 1, 1]:
            t.record(float(v))
        assert t.mean == 1.0  # Old values evicted

    def test_percentile_boundary(self):
        t = LatencyTracker()
        t.record(1.0)
        assert t.percentile(0) == 1.0
        assert t.percentile(100) == 1.0


class TestActionMonitorStrict:
    def test_high_dimensional_action(self):
        m = ActionMonitor()
        action = np.random.randn(128)
        info = m.record(action)
        assert info["magnitude"] > 0

    def test_zero_action(self):
        m = ActionMonitor()
        info = m.record(np.zeros(7))
        assert info["magnitude"] == 0.0

    def test_large_burst_anomaly_rate(self):
        m = ActionMonitor(window_size=100, anomaly_threshold=2.0)
        for _ in range(100):
            m.record(np.random.randn(7) * 0.1)
        # 10 anomalous actions
        for _ in range(10):
            m.record(np.ones(7) * 100.0)
        assert m.anomaly_rate > 0.05


class TestTelemetryIntegration:
    def test_full_inference_loop(self):
        """Simulate a realistic inference loop."""
        t = InferenceTelemetry(
            TelemetryConfig(
                window_size=100,
                log_interval=0,
            )
        )
        for step in range(50):
            latency = 10.0 + np.random.randn() * 2.0
            t.record_inference(latency)
            action = np.random.randn(7) * 0.1
            t.record_action(action)
            t.record_buffer(max(0, 4 - step % 5))

        s = t.summary()
        assert s["total_inferences"] == 50
        assert s["p50_ms"] > 0
        assert s["actions"]["action_count"] == 50
        assert s["buffer"]["starvation_count"] >= 0

    def test_export_roundtrip(self, tmp_path):
        """Export and re-read telemetry JSON."""
        t = InferenceTelemetry()
        for _ in range(20):
            t.record_inference(np.random.uniform(5, 15))
        path = tmp_path / "telem.json"
        t.export_json(path)
        data = json.loads(path.read_text())
        assert data["count"] == 20
        assert data["p50_ms"] > 0
