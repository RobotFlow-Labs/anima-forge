"""Tests for PRD-27: Hyperparameter Search & Optimization."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from forge.config import ForgeConfig
from forge.hyperparam import (
    HyperparamSearch,
    SearchSpace,
    Trial,
    _trial_id,
    recommend_config,
)


@pytest.fixture
def results_dir(tmp_path: Path) -> Path:
    return tmp_path / "hyperparam"


# ── SearchSpace ───────────────────────────────────────────


class TestSearchSpace:
    def test_add_categorical(self):
        space = SearchSpace()
        space.add_categorical("variant", ["nano", "small", "micro"])
        assert "variant" in space.param_names
        assert space.params["variant"].values == ["nano", "small", "micro"]

    def test_add_choice(self):
        space = SearchSpace()
        space.add_choice("lora_rank", [16, 32, 64])
        assert space.params["lora_rank"].values == [16, 32, 64]

    def test_add_range(self):
        space = SearchSpace()
        space.add_range("learning_rate", 1e-5, 1e-3, log_scale=True)
        spec = space.params["learning_rate"]
        assert spec.low == 1e-5
        assert spec.high == 1e-3
        assert spec.log_scale is True

    def test_chaining(self):
        space = SearchSpace().add_categorical("variant", ["nano", "small"]).add_choice("lora_rank", [16, 32])
        assert len(space.param_names) == 2

    def test_grid_size(self):
        space = SearchSpace()
        space.add_categorical("variant", ["nano", "small"])
        space.add_choice("lora_rank", [16, 32, 64])
        assert space.grid_size() == 6  # 2 * 3

    def test_enumerate_grid(self):
        space = SearchSpace()
        space.add_categorical("variant", ["nano", "small"])
        space.add_choice("lora_rank", [16, 32])
        grid = space.enumerate_grid()
        assert len(grid) == 4
        assert {"variant": "nano", "lora_rank": 16} in grid
        assert {"variant": "small", "lora_rank": 32} in grid

    def test_enumerate_grid_with_range_step(self):
        space = SearchSpace()
        space.add_range("lr", 0.001, 0.003, step=0.001)
        grid = space.enumerate_grid()
        assert len(grid) == 3
        assert grid[0]["lr"] == pytest.approx(0.001)
        assert grid[2]["lr"] == pytest.approx(0.003)

    def test_sample_random_categorical(self):
        space = SearchSpace()
        space.add_categorical("variant", ["nano", "small"])
        import random

        rng = random.Random(42)
        sample = space.sample_random(rng)
        assert sample["variant"] in ["nano", "small"]

    def test_sample_random_range(self):
        space = SearchSpace()
        space.add_range("lr", 1e-5, 1e-3)
        import random

        rng = random.Random(42)
        sample = space.sample_random(rng)
        assert 1e-5 <= sample["lr"] <= 1e-3

    def test_sample_random_log_scale(self):
        space = SearchSpace()
        space.add_range("lr", 1e-5, 1e-3, log_scale=True)
        import random

        rng = random.Random(42)
        samples = [space.sample_random(rng)["lr"] for _ in range(100)]
        # Log-scale sampling should produce more values near the low end
        median = sorted(samples)[50]
        assert median < 5e-4  # Below linear midpoint

    def test_to_dict_from_dict_roundtrip(self):
        space = SearchSpace()
        space.add_categorical("variant", ["nano", "small"])
        space.add_range("lr", 1e-5, 1e-3, log_scale=True)
        data = space.to_dict()
        restored = SearchSpace.from_dict(data)
        assert restored.param_names == space.param_names


# ── Trial ─────────────────────────────────────────────────


class TestTrial:
    def test_create_trial(self):
        trial = Trial(trial_id="abc", params={"lr": 0.001})
        assert trial.status == "pending"
        assert trial.objective_value is None

    def test_trial_roundtrip(self):
        trial = Trial(
            trial_id="abc",
            params={"lr": 0.001},
            status="completed",
            objective_value=0.023,
            metrics={"latency_ms": 45.0},
        )
        d = trial.to_dict()
        restored = Trial.from_dict(d)
        assert restored.trial_id == "abc"
        assert restored.objective_value == 0.023

    def test_trial_id_deterministic(self):
        params = {"lr": 0.001, "rank": 32}
        assert _trial_id(params) == _trial_id(params)

    def test_trial_id_different_for_different_params(self):
        assert _trial_id({"lr": 0.001}) != _trial_id({"lr": 0.002})

    def test_duration(self):
        trial = Trial(trial_id="abc", params={}, start_time=100.0, end_time=150.0)
        assert trial.duration_seconds == 50.0


# ── HyperparamSearch ──────────────────────────────────────


class TestHyperparamSearch:
    def test_create_search(self, results_dir: Path):
        space = SearchSpace().add_choice("lora_rank", [16, 32])
        search = HyperparamSearch(space, results_dir=results_dir)
        assert search.count == 0

    def test_grid_search(self, results_dir: Path):
        space = SearchSpace().add_categorical("variant", ["nano", "small"]).add_choice("lora_rank", [16, 32])
        search = HyperparamSearch(space, results_dir=results_dir)
        trials = search.grid_search()
        assert len(trials) == 4
        assert all(t.status == "pending" for t in trials)

    def test_random_search(self, results_dir: Path):
        space = SearchSpace().add_range("lr", 1e-5, 1e-3, log_scale=True).add_choice("rank", [16, 32, 64])
        search = HyperparamSearch(space, results_dir=results_dir)
        trials = search.random_search(n_trials=10, seed=42)
        assert len(trials) == 10

    def test_trial_lifecycle(self, results_dir: Path):
        space = SearchSpace().add_choice("rank", [16])
        search = HyperparamSearch(space, results_dir=results_dir)

        trial = search.create_trial({"rank": 16})
        assert trial.status == "pending"

        search.start_trial(trial)
        assert trial.status == "running"
        assert trial.start_time > 0

        search.complete_trial(trial, objective_value=0.023, metrics={"loss": 0.023})
        assert trial.status == "completed"
        assert trial.objective_value == 0.023

    def test_fail_trial(self, results_dir: Path):
        space = SearchSpace()
        search = HyperparamSearch(space, results_dir=results_dir)
        trial = search.create_trial({"rank": 16})
        search.fail_trial(trial, error="OOM")
        assert trial.status == "failed"
        assert trial.error == "OOM"

    def test_best_trial(self, results_dir: Path):
        space = SearchSpace()
        search = HyperparamSearch(space, objective="loss", results_dir=results_dir)

        t1 = search.create_trial({"rank": 16})
        search.complete_trial(t1, objective_value=0.05)

        t2 = search.create_trial({"rank": 32})
        search.complete_trial(t2, objective_value=0.02)

        best = search.best_trial()
        assert best.trial_id == t2.trial_id
        assert best.objective_value == 0.02

    def test_best_trial_higher_is_better(self, results_dir: Path):
        space = SearchSpace()
        search = HyperparamSearch(
            space,
            objective="throughput",
            lower_is_better=False,
            results_dir=results_dir,
        )

        t1 = search.create_trial({"rank": 16})
        search.complete_trial(t1, objective_value=10.0)

        t2 = search.create_trial({"rank": 32})
        search.complete_trial(t2, objective_value=25.0)

        best = search.best_trial()
        assert best.objective_value == 25.0

    def test_best_trial_empty(self, results_dir: Path):
        search = HyperparamSearch(SearchSpace(), results_dir=results_dir)
        assert search.best_trial() is None

    def test_top_trials(self, results_dir: Path):
        space = SearchSpace()
        search = HyperparamSearch(space, results_dir=results_dir)

        for i, val in enumerate([0.05, 0.02, 0.08, 0.01, 0.03]):
            t = search.create_trial({"rank": i})
            search.complete_trial(t, objective_value=val)

        top3 = search.top_trials(n=3)
        assert len(top3) == 3
        assert top3[0].objective_value == 0.01
        assert top3[1].objective_value == 0.02
        assert top3[2].objective_value == 0.03

    def test_summary(self, results_dir: Path):
        space = SearchSpace()
        search = HyperparamSearch(space, objective="loss", results_dir=results_dir)

        t1 = search.create_trial({"rank": 16})
        search.complete_trial(t1, objective_value=0.05)

        t2 = search.create_trial({"rank": 32})
        search.complete_trial(t2, objective_value=0.02)

        t3 = search.create_trial({"rank": 64})
        search.fail_trial(t3, error="OOM")

        summary = search.summary()
        assert summary["total_trials"] == 3
        assert summary["completed"] == 2
        assert summary["failed"] == 1
        assert summary["best_value"] == 0.02
        assert summary["best_params"] == {"rank": 32}


# ── Persistence ───────────────────────────────────────────


class TestPersistence:
    def test_persists_trials(self, results_dir: Path):
        space = SearchSpace().add_choice("rank", [16, 32])
        search1 = HyperparamSearch(space, results_dir=results_dir)
        t = search1.create_trial({"rank": 16})
        search1.complete_trial(t, objective_value=0.05)

        # Reload
        search2 = HyperparamSearch(SearchSpace(), results_dir=results_dir)
        assert search2.count == 1
        assert search2.trials[0].objective_value == 0.05

    def test_trials_json_format(self, results_dir: Path):
        space = SearchSpace().add_choice("rank", [16])
        search = HyperparamSearch(space, results_dir=results_dir)
        t = search.create_trial({"rank": 16})
        search.complete_trial(t, objective_value=0.05)

        data = json.loads((results_dir / "trials.json").read_text())
        assert "space" in data
        assert "trials" in data
        assert len(data["trials"]) == 1


class TestBenchmarkRecommendations:
    def test_recommend_ignores_noncomparable_loss_metrics(self, tmp_path: Path) -> None:
        (tmp_path / "bench_12_full_pipeline_combos.json").write_text(
            json.dumps(
                {
                    "pipelines": {
                        "stale": {
                            "config": {"action_head_type": "flow"},
                            "training": {"loss_reduction_pct": 99.0},
                            "inference": {"fp16_fps": 100.0},
                        },
                        "fixed": {
                            "config": {"action_head_type": "diffusion"},
                            "training": {
                                "loss_metric": "fixed-real-evaluation-mean",
                                "loss_reduction_pct": 10.0,
                            },
                            "inference": {"fp16_fps": 5.0},
                        },
                    }
                }
            ),
            encoding="utf-8",
        )

        recommendations = recommend_config(tmp_path)

        assert [item["name"] for item in recommendations] == ["fixed"]

    def test_recommend_uses_actual_fixed_suite03_run_names(self, tmp_path: Path) -> None:
        (tmp_path / "bench_12_full_pipeline_combos.json").write_text(
            json.dumps(
                {
                    "pipelines": {
                        "fixed": {
                            "config": {"action_head_type": "flow"},
                            "training": {
                                "loss_metric": "fixed-real-evaluation-mean",
                                "loss_reduction_pct": 20.0,
                            },
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        (tmp_path / "bench_03_training.json").write_text(
            json.dumps(
                {
                    "run_3_lr2e4_100steps": {
                        "n_steps": 100,
                        "loss_metric": "fixed-real-evaluation-mean",
                        "loss_reduction_pct": 90.0,
                        "steps_per_sec": 3.0,
                    },
                    "run_legacy": {"loss_reduction_pct": 100.0},
                }
            ),
            encoding="utf-8",
        )

        recommendation = recommend_config(tmp_path)[0]

        assert recommendation["training_insight"]["best_lr"] == "lr2e4"
        assert "100 steps" in recommendation["training_insight"]["note"]


# ── apply_to_config ───────────────────────────────────────


class TestApplyToConfig:
    def test_apply_student_params(self, results_dir: Path):
        space = SearchSpace()
        search = HyperparamSearch(space, results_dir=results_dir)
        config = ForgeConfig.default()

        search.apply_to_config({"lora_rank": 64, "variant": "small"}, config)
        assert config.student.lora_rank == 64
        assert config.student.variant == "small"

    def test_apply_distill_params(self, results_dir: Path):
        space = SearchSpace()
        search = HyperparamSearch(space, results_dir=results_dir)
        config = ForgeConfig.default()

        search.apply_to_config({"learning_rate": 1e-3, "batch_size": 32}, config)
        assert config.distill.learning_rate == 1e-3
        assert config.distill.batch_size == 32

    def test_apply_dotted_params(self, results_dir: Path):
        space = SearchSpace()
        search = HyperparamSearch(space, results_dir=results_dir)
        config = ForgeConfig.default()

        search.apply_to_config({"distill.temperature": 8.0}, config)
        assert config.distill.temperature == 8.0


# ── CLI ───────────────────────────────────────────────────


class TestCLI:
    def test_status_json(self, results_dir: Path):
        from typer.testing import CliRunner

        from forge.cli import app

        # Create some trials
        space = SearchSpace()
        search = HyperparamSearch(space, results_dir=results_dir)
        t = search.create_trial({"rank": 16})
        search.complete_trial(t, objective_value=0.05)

        runner = CliRunner()
        result = runner.invoke(app, ["hyperparam", "status", "--json", "--results-dir", str(results_dir)])
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert data["completed"] == 1

    def test_top_json(self, results_dir: Path):
        from typer.testing import CliRunner

        from forge.cli import app

        space = SearchSpace()
        search = HyperparamSearch(space, results_dir=results_dir)
        for i in range(3):
            t = search.create_trial({"rank": i * 16})
            search.complete_trial(t, objective_value=0.01 * (i + 1))

        runner = CliRunner()
        result = runner.invoke(app, ["hyperparam", "top", "--n", "2", "--json", "--results-dir", str(results_dir)])
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert len(data) == 2
