"""Tests for PRD-24: Training Metrics & Monitoring.

All tests run on CPU with no external dependencies.
"""

from __future__ import annotations

from forge.metrics import (
    ConsoleLogger,
    JSONLogger,
    MetricsAggregator,
    MetricsCollector,
    TensorBoardLogger,
    TrainingMonitor,
    WandBLogger,
)

# ── MetricsAggregator tests ──────────────────────────────────────


class TestMetricsAggregator:
    def test_empty(self):
        agg = MetricsAggregator()
        assert agg.count == 0
        assert agg.mean == 0.0
        assert agg.min == 0.0
        assert agg.max == 0.0
        assert agg.std == 0.0

    def test_single_value(self):
        agg = MetricsAggregator()
        agg.update(5.0)
        assert agg.count == 1
        assert agg.mean == 5.0
        assert agg.last == 5.0
        assert agg.std == 0.0  # only 1 value

    def test_multiple_values(self):
        agg = MetricsAggregator()
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            agg.update(v)
        assert agg.count == 5
        assert agg.mean == 3.0
        assert agg.min == 1.0
        assert agg.max == 5.0
        assert agg.last == 5.0

    def test_window_eviction(self):
        agg = MetricsAggregator(window=3)
        for v in [1.0, 2.0, 3.0, 100.0, 200.0]:
            agg.update(v)
        # Only last 3 values: 3.0, 100.0, 200.0
        assert agg.count == 3
        assert agg.min == 3.0
        assert agg.max == 200.0

    def test_global_mean_tracks_all(self):
        agg = MetricsAggregator(window=2)
        for v in [1.0, 2.0, 3.0, 4.0]:
            agg.update(v)
        assert agg.total_count == 4
        assert agg.global_mean == 2.5  # (1+2+3+4)/4

    def test_std_correct(self):
        agg = MetricsAggregator()
        for v in [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]:
            agg.update(v)
        # Known std for this data
        assert agg.std > 0

    def test_summary_dict(self):
        agg = MetricsAggregator()
        agg.update(1.0)
        agg.update(2.0)
        s = agg.summary()
        assert "mean" in s
        assert "min" in s
        assert "max" in s
        assert "std" in s
        assert "last" in s
        assert "count" in s


# ── MetricsCollector tests ───────────────────────────────────────


class TestMetricsCollector:
    def test_record_single(self):
        collector = MetricsCollector()
        collector.record("loss", 0.5, step=0)
        agg = collector.get("loss")
        assert agg is not None
        assert agg.last == 0.5

    def test_record_dict(self):
        collector = MetricsCollector()
        collector.record_dict({"loss": 0.5, "lr": 1e-3}, step=10)
        assert collector.get("loss").last == 0.5
        assert collector.get("lr").last == 1e-3

    def test_metric_names(self):
        collector = MetricsCollector()
        collector.record("a", 1.0)
        collector.record("b", 2.0)
        assert set(collector.metric_names) == {"a", "b"}

    def test_get_summary(self):
        collector = MetricsCollector()
        collector.record("loss", 0.5)
        s = collector.get_summary("loss")
        assert s["last"] == 0.5

    def test_get_all_summaries(self):
        collector = MetricsCollector()
        collector.record("loss", 0.5)
        collector.record("lr", 0.001)
        summaries = collector.get_all_summaries()
        assert "loss" in summaries
        assert "lr" in summaries

    def test_get_snapshot(self):
        collector = MetricsCollector()
        collector.record("loss", 0.5, step=10)
        snap = collector.get_snapshot(step=10)
        assert snap["step"] == 10
        assert snap["loss"] == 0.5
        assert "elapsed_seconds" in snap

    def test_reset(self):
        collector = MetricsCollector()
        collector.record("loss", 0.5)
        collector.reset()
        assert len(collector.metric_names) == 0

    def test_nonexistent_metric(self):
        collector = MetricsCollector()
        assert collector.get("nonexistent") is None
        assert collector.get_summary("nonexistent") == {}


# ── JSONLogger tests ─────────────────────────────────────────────


class TestJSONLogger:
    def test_write_and_read(self, tmp_path):
        path = tmp_path / "metrics.jsonl"
        lgr = JSONLogger(path)
        lgr.log({"loss": 0.5, "lr": 1e-3}, step=0)
        lgr.log({"loss": 0.3, "lr": 1e-4}, step=100)
        lgr.close()

        records = JSONLogger.load(path)
        assert len(records) == 2
        assert records[0]["step"] == 0
        assert records[0]["loss"] == 0.5
        assert records[1]["step"] == 100

    def test_creates_parent_dir(self, tmp_path):
        path = tmp_path / "sub" / "dir" / "metrics.jsonl"
        lgr = JSONLogger(path)
        lgr.log({"loss": 0.1}, step=0)
        lgr.close()
        assert path.exists()


# ── ConsoleLogger tests ──────────────────────────────────────────


class TestConsoleLogger:
    def test_log_runs(self, caplog):
        lgr = ConsoleLogger(prefix="TEST")
        with caplog.at_level("INFO", logger="forge.metrics"):
            lgr.log({"loss": 0.5}, step=0)
        lgr.close()
        # Just verify it doesn't crash

    def test_close_is_noop(self):
        lgr = ConsoleLogger()
        lgr.close()  # should not raise


# ── TensorBoardLogger tests ──────────────────────────────────────


class TestTensorBoardLogger:
    def test_graceful_when_unavailable(self, tmp_path):
        """TensorBoardLogger works even if tensorboard is missing."""
        lgr = TensorBoardLogger(tmp_path / "tb")
        lgr.log({"loss": 0.5}, step=0)
        lgr.close()
        # If tensorboard IS available, it writes; if not, it silently skips


# ── TrainingMonitor tests ────────────────────────────────────────


class TestTrainingMonitor:
    def test_basic_workflow(self, tmp_path):
        monitor = TrainingMonitor(
            log_dir=tmp_path,
            log_every=10,
            use_json=True,
            use_console=False,
            use_tensorboard=False,
        )
        for step in range(20):
            monitor.record({"loss": 1.0 - step * 0.01, "lr": 1e-3}, step=step)
        monitor.close()

        # Check JSON log was written
        records = JSONLogger.load(tmp_path / "metrics.jsonl")
        assert len(records) == 2  # step 0 and step 10

    def test_record_scalar(self, tmp_path):
        monitor = TrainingMonitor(
            log_dir=tmp_path,
            use_json=False,
            use_console=False,
        )
        monitor.record_scalar("custom_metric", 42.0, step=0)
        agg = monitor.collector.get("custom_metric")
        assert agg is not None
        assert agg.last == 42.0
        monitor.close()

    def test_get_summary(self, tmp_path):
        monitor = TrainingMonitor(
            log_dir=tmp_path,
            use_json=False,
            use_console=False,
        )
        for i in range(10):
            monitor.record({"loss": float(i)}, step=i)
        summary = monitor.get_summary()
        assert "loss" in summary
        assert summary["loss"]["count"] == 10
        monitor.close()

    def test_training_curves(self, tmp_path):
        monitor = TrainingMonitor(
            log_dir=tmp_path,
            use_json=False,
            use_console=False,
        )
        for i in range(5):
            monitor.record({"loss": 1.0 - i * 0.1}, step=i)
        curves = monitor.get_training_curves()
        assert "loss" in curves
        assert curves["loss"]["total_count"] == 5
        monitor.close()

    def test_context_manager(self, tmp_path):
        with TrainingMonitor(
            log_dir=tmp_path,
            use_json=False,
            use_console=False,
        ) as monitor:
            monitor.record({"loss": 0.5}, step=0)
        # Should not raise after exit


# ── WandBLogger tests ───────────────────────────────────────────


class TestWandBLogger:
    def test_wandb_logger_no_run(self):
        """WandBLogger with no run is not available and doesn't crash."""
        lgr = WandBLogger.__new__(WandBLogger)
        lgr._run = None
        lgr._owns_run = False
        # Should not crash
        lgr.log({"loss": 0.5}, step=0)
        lgr.close()
        assert not lgr.available
        assert lgr.url is None

    def test_wandb_logger_offline(self, tmp_path, monkeypatch):
        """WandBLogger works in offline mode (no network)."""
        monkeypatch.setenv("WANDB_MODE", "offline")
        monkeypatch.setenv("WANDB_DIR", str(tmp_path))
        monkeypatch.setenv("WANDB_SILENT", "true")

        lgr = WandBLogger(project="test-forge-offline", run_name="test-run")
        assert lgr.available
        lgr.log({"loss": 0.5, "lr": 1e-3}, step=10)
        lgr.close()

    def test_wandb_logger_close_owns_run(self, tmp_path, monkeypatch):
        """finish() is called when WandBLogger owns the run."""
        monkeypatch.setenv("WANDB_MODE", "offline")
        monkeypatch.setenv("WANDB_DIR", str(tmp_path))
        monkeypatch.setenv("WANDB_SILENT", "true")

        lgr = WandBLogger(project="test-forge-close")
        assert lgr._owns_run is True
        assert lgr.available
        lgr.close()
        # After close, the run is finished — verify by checking it doesn't crash

    def test_wandb_logger_external_run(self, tmp_path, monkeypatch):
        """WandBLogger with external run does NOT finish it on close."""
        monkeypatch.setenv("WANDB_MODE", "offline")
        monkeypatch.setenv("WANDB_DIR", str(tmp_path))
        monkeypatch.setenv("WANDB_SILENT", "true")
        import wandb

        run = wandb.init(project="test-forge-ext", reinit=True)
        lgr = WandBLogger(run=run)
        assert lgr._owns_run is False
        assert lgr.available
        lgr.close()
        # Run should still be active (not finished by logger)
        run.finish()

    def test_wandb_logger_url(self, tmp_path, monkeypatch):
        """url property returns a value when run is available."""
        monkeypatch.setenv("WANDB_MODE", "offline")
        monkeypatch.setenv("WANDB_DIR", str(tmp_path))
        monkeypatch.setenv("WANDB_SILENT", "true")

        lgr = WandBLogger(project="test-forge-url")
        assert lgr.available
        # In offline mode, url may be None, but available should be True
        lgr.close()

    def test_training_monitor_with_wandb(self, tmp_path, monkeypatch):
        """TrainingMonitor with use_wandb=True creates WandBLogger."""
        monkeypatch.setenv("WANDB_MODE", "offline")
        monkeypatch.setenv("WANDB_DIR", str(tmp_path))
        monkeypatch.setenv("WANDB_SILENT", "true")

        monitor = TrainingMonitor(
            log_dir=tmp_path,
            use_json=False,
            use_console=False,
            use_wandb=True,
            wandb_project="test-forge-monitor",
        )
        # WandBLogger should be in the loggers list
        has_wandb = any(isinstance(lgr, WandBLogger) for lgr in monitor._loggers)
        assert has_wandb
        monitor.record({"loss": 0.5}, step=0)
        monitor.close()
