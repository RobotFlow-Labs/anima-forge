"""Tests for PRD-31: Automated Hyperparameter Optimization."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import optuna

from forge.auto_hyperparam import (
    SCORE_FUNCTIONS,
    AutoTrialResult,
    _build_trial_config,
    _sync_to_hyperparam_search,
    create_forge_study,
    export_best_yaml,
    get_search_summary,
    score_balanced,
    score_quality,
    score_size,
    score_speed,
    suggest_forge_params,
)

# ── Score Functions ──────────────────────────────────────────


class TestScoreFunctions:
    def test_score_balanced(self):
        metrics = {"fps": 10, "loss_reduction_pct": 80, "compression_ratio": 8, "total_params_m": 1000}
        score = score_balanced(metrics)
        expected = 10 * 3 + 80 * 2 + 8 * 5 + (1000 / 1000) * 1
        assert score == pytest.approx(expected)

    def test_score_speed(self):
        metrics = {"fps": 15, "loss_reduction_pct": 50}
        score = score_speed(metrics)
        assert score == pytest.approx(15 * 10 + 50 * 0.1)

    def test_score_quality(self):
        metrics = {"fps": 10, "loss_reduction_pct": 90}
        score = score_quality(metrics)
        assert score == pytest.approx(90 * 10 + 10 * 0.5)

    def test_score_size(self):
        metrics = {"compression_ratio": 8, "total_params_m": 500}
        score = score_size(metrics)
        assert score == pytest.approx(8 * 20 + (1000 / 500) * 10)

    def test_all_objectives_registered(self):
        assert set(SCORE_FUNCTIONS.keys()) == {"balanced", "speed", "quality", "size"}

    def test_score_functions_handle_zero_params(self):
        """Ensure division by zero is handled."""
        metrics = {"fps": 10, "loss_reduction_pct": 50, "compression_ratio": 4, "total_params_m": 0}
        # Should not raise — we use max(params, 1)
        score = score_balanced(metrics)
        assert score > 0

    def test_score_functions_handle_missing_keys(self):
        """Missing keys default to 0."""
        score = score_balanced({})
        assert score >= 0

    def test_speed_prioritizes_fps(self):
        """Speed objective should heavily weight FPS."""
        slow = {"fps": 5, "loss_reduction_pct": 90}
        fast = {"fps": 15, "loss_reduction_pct": 30}
        assert score_speed(fast) > score_speed(slow)

    def test_quality_prioritizes_loss(self):
        """Quality objective should heavily weight loss reduction."""
        low_quality = {"fps": 20, "loss_reduction_pct": 30}
        high_quality = {"fps": 5, "loss_reduction_pct": 95}
        assert score_quality(high_quality) > score_quality(low_quality)


# ── Study Creation ───────────────────────────────────────────


class TestStudyCreation:
    def test_create_study_default(self):
        study = create_forge_study()
        assert study.study_name == "forge_auto_hp"
        assert study.direction.name == "MAXIMIZE"

    def test_create_study_custom_name(self):
        study = create_forge_study(study_name="test_study")
        assert study.study_name == "test_study"

    def test_create_study_median_pruner(self):
        import optuna

        study = create_forge_study(pruner="median")
        assert isinstance(study.pruner, optuna.pruners.MedianPruner)

    def test_create_study_hyperband_pruner(self):
        import optuna

        study = create_forge_study(pruner="hyperband")
        assert isinstance(study.pruner, optuna.pruners.HyperbandPruner)

    def test_create_study_with_sqlite(self, tmp_path):
        db_path = tmp_path / "test.db"
        study = create_forge_study(
            storage=f"sqlite:///{db_path}",
            study_name="sqlite_test",
        )
        assert study is not None
        # Verify DB was created
        assert db_path.exists()

    def test_create_study_load_existing(self, tmp_path):
        """Loading existing study preserves trials."""
        db_path = tmp_path / "persist.db"
        storage = f"sqlite:///{db_path}"

        study1 = create_forge_study(storage=storage, study_name="persist_test")
        # Add a dummy trial
        study1.add_trial(_create_dummy_trial(number=0, value=42.0, params={"lora_rank": 64}))

        # Reload
        study2 = create_forge_study(storage=storage, study_name="persist_test")
        assert len(study2.trials) == 1
        assert study2.best_value == pytest.approx(42.0)

    def test_tpe_parameter_sequence_is_seeded(self):
        first_study = create_forge_study(study_name="seeded_first", random_seed=7)
        second_study = create_forge_study(study_name="seeded_second", random_seed=7)

        first = suggest_forge_params(first_study.ask())
        second = suggest_forge_params(second_study.ask())

        assert first == second


# ── Parameter Suggestion ─────────────────────────────────────


class TestParamSuggestion:
    def test_suggest_all_params(self):
        import optuna

        study = optuna.create_study()

        trial = study.ask()
        params = suggest_forge_params(trial)

        assert "lora_rank" in params
        assert "action_head_type" in params
        assert "learning_rate" in params
        assert "prune_keep_ratio" in params
        assert "quant_bits" in params
        assert "batch_size" in params
        assert "bridge_n_queries" in params
        assert "bridge_n_layers" in params
        assert "flow_inference_steps" in params

    def test_param_types(self):
        import optuna

        study = optuna.create_study()
        trial = study.ask()
        params = suggest_forge_params(trial)

        assert params["lora_rank"] in [16, 32, 64, 128]
        assert params["action_head_type"] in ["diffusion", "flow"]
        assert 1e-4 <= params["learning_rate"] <= 5e-3
        assert 0.5 <= params["prune_keep_ratio"] <= 1.0
        assert params["quant_bits"] in [4, 8]
        assert params["batch_size"] in [4, 8, 16]
        assert params["bridge_n_queries"] in [32, 64, 128]
        assert 1 <= params["bridge_n_layers"] <= 4
        assert 1 <= params["flow_inference_steps"] <= 8

    def test_trial_config_uses_canonical_v3_nano_backbones(self):
        config = _build_trial_config(
            {
                "lora_rank": 64,
                "action_head_type": "flow",
                "flow_inference_steps": 4,
                "bridge_n_queries": 64,
                "bridge_n_layers": 3,
                "learning_rate": 2e-4,
                "batch_size": 8,
            }
        )
        assert config.student.vision_encoder == "google/siglip2-so400m-patch14-384"
        assert config.student.language_model == "Qwen/Qwen3-0.6B"
        assert config.student.bridge_d_model == 1024


# ── AutoTrialResult ──────────────────────────────────────────


class TestAutoTrialResult:
    def test_to_dict(self):
        result = AutoTrialResult(
            trial_number=0,
            params={"lora_rank": 64},
            score=42.0,
            metrics={"fps": 10},
            status="completed",
            duration_s=30.0,
        )
        d = result.to_dict()
        assert d["trial_number"] == 0
        assert d["score"] == 42.0
        assert d["status"] == "completed"

    def test_failed_trial(self):
        result = AutoTrialResult(
            trial_number=1,
            params={},
            score=0,
            metrics={},
            status="failed",
            error="OOM",
        )
        assert result.error == "OOM"


# ── Sync to HyperparamSearch ─────────────────────────────────


class TestSync:
    def test_sync_creates_file(self, tmp_path):
        import optuna

        study = optuna.create_study(direction="maximize")
        study.add_trial(
            _create_dummy_trial(
                number=0,
                value=50.0,
                params={"lora_rank": 64, "action_head_type": "flow"},
                user_attrs={"metrics": {"fps": 10, "loss_reduction_pct": 80}},
            )
        )

        trials_path = tmp_path / "trials.json"
        _sync_to_hyperparam_search(study, trials_path, "balanced")

        assert trials_path.exists()
        data = json.loads(trials_path.read_text())
        assert data["source"] == "auto_hyperparam"
        assert len(data["trials"]) == 1
        assert data["trials"][0]["trial_id"] == "auto_0000"

    def test_sync_skips_pruned(self, tmp_path):
        import optuna

        study = optuna.create_study(direction="maximize")
        # Add completed trial
        study.add_trial(_create_dummy_trial(number=0, value=50.0, params={"lora_rank": 64}))
        # Add pruned trial
        pruned = _create_dummy_trial(
            number=1,
            value=None,
            params={"lora_rank": 32},
            state=optuna.trial.TrialState.PRUNED,
        )
        study.add_trial(pruned)

        trials_path = tmp_path / "trials.json"
        _sync_to_hyperparam_search(study, trials_path, "balanced")

        data = json.loads(trials_path.read_text())
        assert len(data["trials"]) == 1


# ── YAML Export ──────────────────────────────────────────────


class TestYAMLExport:
    def test_export_best_yaml(self, tmp_path):
        import optuna

        study = optuna.create_study(direction="maximize")
        study.add_trial(
            _create_dummy_trial(
                number=0,
                value=100.0,
                params={
                    "lora_rank": 64,
                    "action_head_type": "flow",
                    "learning_rate": 2e-4,
                    "batch_size": 8,
                    "flow_inference_steps": 4,
                    "bridge_n_queries": 64,
                    "bridge_n_layers": 4,
                    "quant_bits": 4,
                    "prune_keep_ratio": 0.6,
                },
                user_attrs={"metrics": {"fps": 14, "loss_reduction_pct": 85}},
            )
        )

        yaml_path = tmp_path / "best_config.yaml"
        result_path = export_best_yaml(study, yaml_path)

        assert result_path.exists()
        import yaml

        config = yaml.safe_load(result_path.read_text())
        assert config["student"]["lora_rank"] == 64
        assert config["student"]["action_head_type"] == "flow"
        assert config["distill"]["learning_rate"] == pytest.approx(2e-4)

    def test_export_no_trials_raises(self, tmp_path):
        import optuna

        study = optuna.create_study(direction="maximize")
        with pytest.raises(ValueError):
            export_best_yaml(study, tmp_path / "fail.yaml")


# ── Search Summary ───────────────────────────────────────────


class TestSearchSummary:
    def test_get_summary_missing(self, tmp_path):
        result = get_search_summary(tmp_path / "nonexistent")
        assert result is None

    def test_get_summary_exists(self, tmp_path):
        results_path = tmp_path / "auto_hp_results.json"
        data = {"objective": "balanced", "completed": 5}
        results_path.write_text(json.dumps(data))

        result = get_search_summary(tmp_path)
        assert result["objective"] == "balanced"
        assert result["completed"] == 5


# ── Helper ───────────────────────────────────────────────────


def _create_dummy_trial(
    number: int,
    value: float | None = None,
    params: dict | None = None,
    user_attrs: dict | None = None,
    state=None,
) -> optuna.trial.FrozenTrial:
    """Create a FrozenTrial for testing."""
    from datetime import datetime

    import optuna

    if state is None:
        state = optuna.trial.TrialState.COMPLETE if value is not None else optuna.trial.TrialState.PRUNED

    distributions = {}
    params = params or {}
    for k, v in params.items():
        if isinstance(v, int):
            distributions[k] = optuna.distributions.CategoricalDistribution([v])
        elif isinstance(v, float):
            distributions[k] = optuna.distributions.FloatDistribution(v * 0.5, v * 2.0)
        elif isinstance(v, str):
            distributions[k] = optuna.distributions.CategoricalDistribution([v])
        else:
            distributions[k] = optuna.distributions.CategoricalDistribution([v])

    return optuna.trial.FrozenTrial(
        number=number,
        state=state,
        value=value,
        datetime_start=datetime.now(),
        datetime_complete=datetime.now(),
        params=params,
        distributions=distributions,
        user_attrs=user_attrs or {},
        system_attrs={},
        intermediate_values={},
        trial_id=number,
        values=None,
    )
