"""Tests for PRD-22: Curriculum Learning & Adaptive Training.

All tests run on CPU with synthetic data — no real models required.
"""

from __future__ import annotations

import json

import pytest
import torch

from forge.config import CurriculumConfig, ForgeConfig
from forge.curriculum import (
    CurriculumSampler,
    CurriculumScheduler,
    DifficultyScorer,
    HardExampleMiner,
    PlateauDetector,
    TeacherDropout,
)

B, D_ACTION, N_TEACHERS = 8, 7, 3


# ── DifficultyScorer tests ───────────────────────────────────────


class TestDifficultyScorer:
    def test_loss_metric(self):
        """Loss-based scoring returns losses as-is."""
        scorer = DifficultyScorer(metric="loss")
        losses = torch.tensor([0.5, 1.2, 0.1, 0.8])
        scores = scorer.score_batch(losses=losses)
        assert torch.allclose(scores, losses)

    def test_confidence_metric(self):
        """Confidence-based: low confidence = high difficulty."""
        scorer = DifficultyScorer(metric="confidence")
        conf = torch.tensor([[0.9, 0.9], [0.1, 0.1]])  # easy, hard
        scores = scorer.score_batch(confidences=conf)
        assert scores[1] > scores[0]  # low confidence is harder

    def test_teacher_disagreement_metric(self):
        """Teacher disagreement: high variance = high difficulty."""
        scorer = DifficultyScorer(metric="teacher_disagreement")
        # Agreeing teachers
        t1 = torch.ones(B, D_ACTION)
        t2 = torch.ones(B, D_ACTION) * 1.01
        # Disagreeing teachers
        t3 = torch.ones(B, D_ACTION) * 5.0
        scores_agree = scorer.score_batch(teacher_actions=[t1, t2])
        scores_disagree = scorer.score_batch(teacher_actions=[t1, t3])
        assert scores_disagree.mean() > scores_agree.mean()

    def test_rank_indices_ascending(self):
        """Ranking easiest-first produces correct order."""
        scorer = DifficultyScorer(metric="loss")
        scores = torch.tensor([0.5, 0.1, 0.9, 0.3])
        ranked = scorer.rank_indices(scores, ascending=True)
        # Expect: 1 (0.1), 3 (0.3), 0 (0.5), 2 (0.9)
        assert ranked[0].item() == 1
        assert ranked[-1].item() == 2

    def test_single_teacher_disagreement_raises(self):
        """Disagreement with 1 teacher raises ValueError."""
        scorer = DifficultyScorer(metric="teacher_disagreement")
        with pytest.raises(ValueError, match="N>=2"):
            scorer.score_batch(teacher_actions=[torch.randn(B, D_ACTION)])

    def test_invalid_metric_raises(self):
        """Unknown metric raises ValueError."""
        with pytest.raises(ValueError, match="Unknown"):
            DifficultyScorer(metric="invalid")

    def test_missing_data_raises(self):
        """Missing required data raises ValueError."""
        scorer = DifficultyScorer(metric="loss")
        with pytest.raises(ValueError, match="losses required"):
            scorer.score_batch()


# ── CurriculumScheduler tests ────────────────────────────────────


class TestCurriculumScheduler:
    def test_linear_schedule_start(self):
        """Linear schedule starts at initial difficulty."""
        sched = CurriculumScheduler(initial_difficulty=0.3, final_difficulty=1.0, ramp_steps=100)
        assert abs(sched.get_difficulty(0) - 0.3) < 1e-5

    def test_linear_schedule_end(self):
        """Linear schedule ends at final difficulty."""
        sched = CurriculumScheduler(initial_difficulty=0.3, final_difficulty=1.0, ramp_steps=100)
        assert abs(sched.get_difficulty(100) - 1.0) < 1e-5

    def test_linear_schedule_midpoint(self):
        """Linear schedule is at midpoint halfway."""
        sched = CurriculumScheduler(initial_difficulty=0.0, final_difficulty=1.0, ramp_steps=100)
        assert abs(sched.get_difficulty(50) - 0.5) < 1e-5

    def test_cosine_schedule_bounds(self):
        """Cosine schedule respects start/end bounds."""
        sched = CurriculumScheduler(
            initial_difficulty=0.2,
            final_difficulty=0.8,
            ramp_steps=100,
            schedule="cosine",
        )
        assert abs(sched.get_difficulty(0) - 0.2) < 1e-5
        assert abs(sched.get_difficulty(100) - 0.8) < 1e-5

    def test_step_schedule_jumps(self):
        """Step schedule has discrete jumps."""
        sched = CurriculumScheduler(
            initial_difficulty=0.3,
            final_difficulty=1.0,
            ramp_steps=300,
            schedule="step",
        )
        d_early = sched.get_difficulty(10)
        d_mid = sched.get_difficulty(150)
        d_late = sched.get_difficulty(250)
        assert d_early < d_mid < d_late

    def test_beyond_ramp_steps_is_final(self):
        """After ramp_steps, difficulty is final."""
        sched = CurriculumScheduler(initial_difficulty=0.3, final_difficulty=1.0, ramp_steps=100)
        assert abs(sched.get_difficulty(999) - 1.0) < 1e-5

    def test_monotonically_increasing(self):
        """Linear schedule is monotonically non-decreasing."""
        sched = CurriculumScheduler(initial_difficulty=0.2, final_difficulty=1.0, ramp_steps=1000)
        prev = 0.0
        for step in range(0, 1100, 50):
            d = sched.get_difficulty(step)
            assert d >= prev - 1e-7
            prev = d


# ── PlateauDetector tests ────────────────────────────────────────


class TestPlateauDetector:
    def test_no_plateau_when_improving(self):
        """No plateau detected when loss is decreasing."""
        detector = PlateauDetector(window=20, threshold=0.01)
        for i in range(30):
            detector.update(1.0 - i * 0.01)
        assert not detector.check_plateau(step=30)

    def test_plateau_detected_when_flat(self):
        """Plateau detected when loss is flat."""
        detector = PlateauDetector(window=20, threshold=0.01)
        for i in range(20):
            detector.update(1.0)
        assert detector.check_plateau(step=20)

    def test_lr_multiplier_accumulates(self):
        """LR multiplier accumulates with each plateau."""
        detector = PlateauDetector(window=10, threshold=0.01, lr_factor=0.5, max_plateaus=3)
        assert abs(detector.get_lr_multiplier() - 1.0) < 1e-7

        # Trigger plateau
        for i in range(10):
            detector.update(1.0)
        detector.check_plateau(step=10)

        assert abs(detector.get_lr_multiplier() - 0.5) < 1e-7

    def test_max_plateaus_respected(self):
        """No more reductions after max_plateaus."""
        detector = PlateauDetector(window=10, threshold=0.01, lr_factor=0.5, max_plateaus=1)
        for i in range(10):
            detector.update(1.0)
        detector.check_plateau(step=10)

        # Try to trigger another
        for i in range(10):
            detector.update(1.0)
        result = detector.check_plateau(step=30)
        assert not result  # max reached

    def test_not_enough_history(self):
        """No plateau when history is too short."""
        detector = PlateauDetector(window=100)
        detector.update(1.0)
        assert not detector.check_plateau(step=1)


# ── TeacherDropout tests ──────────────────────────────────────────


class TestTeacherDropout:
    def test_no_dropout_at_start(self):
        """No dropout when rate is 0."""
        td = TeacherDropout(n_teachers=5, dropout_start=0.0, dropout_end=0.3)
        mask = td.get_active_mask(step=0)
        assert all(mask)

    def test_dropout_increases_over_time(self):
        """Dropout rate increases."""
        td = TeacherDropout(n_teachers=5, dropout_start=0.0, dropout_end=0.4, ramp_steps=100)
        r0 = td.get_dropout_rate(0)
        r50 = td.get_dropout_rate(50)
        r100 = td.get_dropout_rate(100)
        assert r0 < r50 < r100

    def test_always_at_least_one_active(self):
        """Even at max dropout, at least 1 teacher is active."""
        td = TeacherDropout(n_teachers=3, dropout_start=0.0, dropout_end=0.9, ramp_steps=10)
        for _ in range(100):
            mask = td.get_active_mask(step=100)
            assert sum(mask) >= 1

    def test_mask_length_matches_n_teachers(self):
        """Mask length equals number of teachers."""
        td = TeacherDropout(n_teachers=7, dropout_start=0.0, dropout_end=0.3)
        mask = td.get_active_mask(step=50)
        assert len(mask) == 7


# ── HardExampleMiner tests ───────────────────────────────────────


class TestHardExampleMiner:
    def test_update_and_retrieve(self):
        """Can update losses and retrieve difficulty scores."""
        miner = HardExampleMiner(dataset_size=100, hard_ratio=0.3)
        miner.update_losses([0, 1, 2], torch.tensor([0.5, 1.0, 0.1]))
        scores = miner.get_difficulty_scores()
        assert scores[1] > scores[0] > scores[2]

    def test_sample_indices_correct_size(self):
        """Sample returns correct batch size."""
        miner = HardExampleMiner(dataset_size=100, hard_ratio=0.3)
        miner.update_losses(list(range(50)), torch.randn(50).abs())
        indices = miner.sample_indices(batch_size=16)
        assert len(indices) == 16

    def test_hard_examples_have_high_loss(self):
        """Hard examples come from high-loss region."""
        miner = HardExampleMiner(dataset_size=100, hard_ratio=1.0)
        # Set clear pattern: indices 90-99 have highest loss
        losses = torch.zeros(100)
        losses[90:] = 10.0
        miner.update_losses(list(range(100)), losses)
        indices = miner.sample_indices(batch_size=10)
        # Most should be from high-loss region
        high_loss_count = sum(1 for i in indices if i >= 90)
        assert high_loss_count >= 5  # at least half from hard region

    def test_unseen_get_median_score(self):
        """Unseen samples get median difficulty."""
        miner = HardExampleMiner(dataset_size=10)
        miner.update_losses([0, 1], torch.tensor([0.2, 0.8]))
        scores = miner.get_difficulty_scores()
        median = torch.tensor([0.2, 0.8]).median()
        assert abs(scores[5].item() - median.item()) < 1e-5  # unseen sample


# ── CurriculumSampler tests ──────────────────────────────────────


class TestCurriculumSampler:
    def test_initial_subset_smaller(self):
        """At step 0, sampler produces smaller subset."""
        config = CurriculumConfig(initial_difficulty=0.3, final_difficulty=1.0, ramp_steps=100)
        sampler = CurriculumSampler(dataset_size=100, config=config)
        sampler.set_step(0)
        assert len(sampler) == 30

    def test_final_full_dataset(self):
        """At final step, sampler uses full dataset."""
        config = CurriculumConfig(initial_difficulty=0.3, final_difficulty=1.0, ramp_steps=100)
        sampler = CurriculumSampler(dataset_size=100, config=config)
        sampler.set_step(100)
        assert len(sampler) == 100

    def test_with_difficulty_scores(self):
        """Sampler uses difficulty scores for ordering."""
        config = CurriculumConfig(
            initial_difficulty=0.5,
            final_difficulty=1.0,
            ramp_steps=100,
            hard_example_mining=False,
        )
        # Scores: index 0 is easiest, index 99 is hardest
        scores = torch.arange(100, dtype=torch.float32)
        sampler = CurriculumSampler(dataset_size=100, config=config, difficulty_scores=scores)
        sampler.set_step(0)

        indices = list(sampler)
        # At 50% difficulty, only indices 0-49 should be available
        assert all(i < 50 for i in indices)

    def test_iterator_yields_indices(self):
        """Sampler iterator yields valid indices."""
        config = CurriculumConfig(initial_difficulty=1.0, hard_example_mining=False)
        sampler = CurriculumSampler(dataset_size=50, config=config)
        indices = list(sampler)
        assert all(0 <= i < 50 for i in indices)
        assert len(indices) == 50


# ── CLI tests ─────────────────────────────────────────────────────


class TestCLI:
    def test_status_json(self, tmp_path):
        """CLI status projects real curriculum state from a run heartbeat."""
        from typer.testing import CliRunner

        from forge.cli_v2 import curriculum_app
        from forge.training_runtime import atomic_write_json

        run_dir = tmp_path / "train-runs" / "run-1"
        atomic_write_json(
            run_dir / "train_state.json",
            {
                "status": "completed",
                "step": 12,
                "curriculum": {
                    "enabled": True,
                    "difficulty": 0.42,
                    "difficulty_metric": "loss",
                    "initial_difficulty": 0.3,
                    "final_difficulty": 1.0,
                    "ramp_schedule": "linear",
                    "ramp_steps": 100,
                    "hard_example_mining": True,
                    "hard_examples_seen": 7,
                    "plateau_detection": True,
                    "plateaus": 1,
                    "teacher_dropout": False,
                    "teacher_dropout_rate": None,
                },
            },
        )

        result = CliRunner().invoke(
            curriculum_app,
            ["status", "--run-dir", str(run_dir), "--json"],
        )
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert data["step"] == 12
        assert data["difficulty"] == 0.42
        assert data["hard_examples_seen"] == 7
        assert data["plateaus"] == 1

    def test_simulate_json(self):
        """CLI simulate command produces valid JSON."""
        from typer.testing import CliRunner

        from forge.cli_v2 import curriculum_app

        runner = CliRunner()
        result = runner.invoke(curriculum_app, ["simulate", "--steps", "1000", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data) == 5
        assert data[0]["step"] == 0
        assert data[-1]["difficulty"] == 1.0


# ── Config tests ──────────────────────────────────────────────────


class TestConfig:
    def test_defaults_valid(self):
        """Default CurriculumConfig is valid."""
        config = CurriculumConfig()
        assert config.enabled is True
        assert 0 < config.initial_difficulty < config.final_difficulty <= 1.0
        assert config.ramp_steps > 0
        assert config.plateau_window > 0
        assert 0.0 <= config.hard_example_ratio <= 1.0

    def test_yaml_override(self, tmp_path):
        """YAML config overrides curriculum settings."""
        yaml_content = """
curriculum:
  enabled: false
  initial_difficulty: 0.1
  ramp_schedule: cosine
  hard_example_mining: false
"""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(yaml_content)

        config = ForgeConfig.from_yaml(yaml_file)
        assert config.curriculum.enabled is False
        assert config.curriculum.initial_difficulty == 0.1
        assert config.curriculum.ramp_schedule == "cosine"
        assert config.curriculum.hard_example_mining is False
        # Defaults preserved
        assert config.curriculum.plateau_threshold == 0.01

    def test_forgeconfig_includes_curriculum(self):
        """ForgeConfig has curriculum field with correct defaults."""
        config = ForgeConfig.default()
        assert hasattr(config, "curriculum")
        assert isinstance(config.curriculum, CurriculumConfig)
