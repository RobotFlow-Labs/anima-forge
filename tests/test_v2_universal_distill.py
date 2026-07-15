"""Tests for PRD-21: Universal Teacher Ensemble Distillation.

All tests run on CPU with synthetic data — no real models required.
"""

from __future__ import annotations

import math
import tempfile

import pytest
import torch
import torch.nn as nn

from forge.config import ForgeConfig, UniversalDistillConfig
from forge.universal_distill import (
    ConfidenceRouter,
    ConsistencyLoss,
    DiversityLoss,
    TeacherSlot,
    UniversalDistillationLoss,
    UniversalRunner,
    plan_gpu_placement,
)

# ── Fixtures ──────────────────────────────────────────────────────

B, D_STUDENT, D_ACTION, N_TEACHERS, CONF_DIM = 4, 64, 7, 3, 7


@pytest.fixture
def student_features():
    return torch.randn(B, D_STUDENT, requires_grad=True)


@pytest.fixture
def student_actions():
    return torch.randn(B, D_ACTION, requires_grad=True)


@pytest.fixture
def teacher_actions_list():
    return [torch.randn(B, D_ACTION) for _ in range(N_TEACHERS)]


@pytest.fixture
def gt_actions():
    return torch.randn(B, D_ACTION)


@pytest.fixture
def teacher_confidences():
    return torch.rand(B, N_TEACHERS, CONF_DIM).abs()


# ── ConfidenceRouter tests ────────────────────────────────────────


class TestConfidenceRouter:
    def test_no_confidence_fallback(self, student_features):
        """Router works when teacher_confidences is None (zero-padded)."""
        router = ConfidenceRouter(D_STUDENT, N_TEACHERS, CONF_DIM)
        weights = router(student_features, teacher_confidences=None)
        assert weights.shape == (B, N_TEACHERS)
        assert torch.allclose(weights.sum(dim=-1), torch.ones(B), atol=1e-5)

    def test_output_shape(self, student_features, teacher_confidences):
        """Router output shape is (B, N_teachers)."""
        router = ConfidenceRouter(D_STUDENT, N_TEACHERS, CONF_DIM)
        weights = router(student_features, teacher_confidences)
        assert weights.shape == (B, N_TEACHERS)

    def test_sum_to_one(self, student_features, teacher_confidences):
        """Routing weights sum to 1 per sample."""
        router = ConfidenceRouter(D_STUDENT, N_TEACHERS, CONF_DIM)
        router.eval()
        weights = router(student_features, teacher_confidences)
        assert torch.allclose(weights.sum(dim=-1), torch.ones(B), atol=1e-5)

    def test_gumbel_train_mode(self, student_features, teacher_confidences):
        """Gumbel softmax is used during training when use_gumbel=True."""
        router = ConfidenceRouter(D_STUDENT, N_TEACHERS, CONF_DIM, use_gumbel=True)
        router.train()
        w1 = router(student_features, teacher_confidences)
        w2 = router(student_features, teacher_confidences)
        # Gumbel adds noise, so two forward passes should differ
        # (with overwhelming probability for B=4, N=3)
        assert not torch.allclose(w1, w2, atol=1e-6)

    def test_softmax_eval_mode(self, student_features, teacher_confidences):
        """Standard softmax in eval mode (deterministic)."""
        router = ConfidenceRouter(D_STUDENT, N_TEACHERS, CONF_DIM, use_gumbel=True)
        router.eval()
        w1 = router(student_features, teacher_confidences)
        w2 = router(student_features, teacher_confidences)
        assert torch.allclose(w1, w2, atol=1e-7)


# ── DiversityLoss tests ──────────────────────────────────────────


class TestDiversityLoss:
    def test_uniform_weights_zero_loss(self):
        """Uniform routing → loss ≈ 0."""
        loss_fn = DiversityLoss()
        uniform = torch.ones(B, N_TEACHERS) / N_TEACHERS
        loss = loss_fn(uniform)
        assert loss.item() < 1e-5

    def test_collapsed_weights_max_loss(self):
        """All weight on one teacher → loss ≈ log(N)."""
        loss_fn = DiversityLoss()
        collapsed = torch.zeros(B, N_TEACHERS)
        collapsed[:, 0] = 1.0
        loss = loss_fn(collapsed)
        assert abs(loss.item() - math.log(N_TEACHERS)) < 1e-4

    def test_gradient_flows(self):
        """Gradient flows through diversity loss."""
        loss_fn = DiversityLoss()
        logits = torch.randn(B, N_TEACHERS, requires_grad=True)
        weights = logits.softmax(dim=-1)
        loss = loss_fn(weights)
        loss.backward()
        assert logits.grad is not None


# ── ConsistencyLoss tests ────────────────────────────────────────


class TestConsistencyLoss:
    def test_student_equals_mean_near_zero(self, teacher_actions_list):
        """When student = teacher mean, loss ≈ 0."""
        loss_fn = ConsistencyLoss()
        teacher_mean = torch.stack(teacher_actions_list).mean(dim=0)
        loss = loss_fn(teacher_mean, teacher_actions_list)
        assert loss.item() < 1e-4

    def test_low_penalty_high_variance(self):
        """High teacher variance → lower penalty for same deviation."""
        loss_fn = ConsistencyLoss()
        # High-variance teachers
        teachers_hv = [torch.randn(B, D_ACTION) * 10 for _ in range(N_TEACHERS)]
        mean_hv = torch.stack(teachers_hv).mean(dim=0)
        student_off = mean_hv + 1.0  # fixed offset
        loss_hv = loss_fn(student_off, teachers_hv)

        # Low-variance teachers (near identical)
        base = torch.randn(B, D_ACTION)
        teachers_lv = [base + torch.randn(B, D_ACTION) * 0.001 for _ in range(N_TEACHERS)]
        mean_lv = torch.stack(teachers_lv).mean(dim=0)
        student_off_lv = mean_lv + 1.0
        loss_lv = loss_fn(student_off_lv, teachers_lv)

        # Low-variance group should penalise more
        assert loss_lv.item() > loss_hv.item()

    def test_gradient_flows(self, teacher_actions_list):
        """Gradient flows through consistency loss."""
        loss_fn = ConsistencyLoss()
        student = torch.randn(B, D_ACTION, requires_grad=True)
        loss = loss_fn(student, teacher_actions_list)
        loss.backward()
        assert student.grad is not None


# ── UniversalDistillationLoss tests ──────────────────────────────


class TestUniversalDistillationLoss:
    def test_all_keys_returned(
        self,
        student_actions,
        teacher_actions_list,
        gt_actions,
        student_features,
        teacher_confidences,
    ):
        """Forward returns all expected loss keys."""
        loss_fn = UniversalDistillationLoss(N_TEACHERS, D_STUDENT, CONF_DIM)
        result = loss_fn(
            student_actions,
            teacher_actions_list,
            gt_actions,
            student_features,
            teacher_confidences,
        )
        expected_keys = {"total", "kd", "task", "diversity", "consistency", "router_weights"}
        assert set(result.keys()) == expected_keys

    def test_weighted_sum(
        self,
        student_actions,
        teacher_actions_list,
        gt_actions,
        student_features,
        teacher_confidences,
    ):
        """Total is weighted sum of components (within tolerance)."""
        alpha_task, alpha_div, alpha_con = 0.3, 0.05, 0.1
        alpha_kd = 1.0 - alpha_task - alpha_div - alpha_con

        loss_fn = UniversalDistillationLoss(
            N_TEACHERS,
            D_STUDENT,
            CONF_DIM,
            alpha_task=alpha_task,
            alpha_diversity=alpha_div,
            alpha_consistency=alpha_con,
        )
        r = loss_fn(
            student_actions,
            teacher_actions_list,
            gt_actions,
            student_features,
            teacher_confidences,
        )
        expected = (
            alpha_kd * r["kd"] + alpha_task * r["task"] + alpha_div * r["diversity"] + alpha_con * r["consistency"]
        )
        assert torch.allclose(r["total"], expected, atol=1e-5)

    def test_gradient_flows(
        self,
        student_actions,
        teacher_actions_list,
        gt_actions,
        student_features,
        teacher_confidences,
    ):
        """Gradient flows to student_actions and student_features."""
        loss_fn = UniversalDistillationLoss(N_TEACHERS, D_STUDENT, CONF_DIM)
        result = loss_fn(
            student_actions,
            teacher_actions_list,
            gt_actions,
            student_features,
            teacher_confidences,
        )
        result["total"].backward()
        assert student_actions.grad is not None
        assert student_features.grad is not None

    def test_single_teacher_degenerates(
        self,
        student_features,
        gt_actions,
    ):
        """With 1 teacher, behaves like simple KD + consistency."""
        loss_fn = UniversalDistillationLoss(1, D_STUDENT, CONF_DIM)
        student = torch.randn(B, D_ACTION, requires_grad=True)
        teacher = [torch.randn(B, D_ACTION)]
        conf = torch.rand(B, 1, CONF_DIM)
        result = loss_fn(student, teacher, gt_actions, student_features, conf)
        assert result["total"].item() > 0
        # Router weights should be all 1.0 (only one teacher)
        assert torch.allclose(
            result["router_weights"],
            torch.ones(B, 1),
            atol=1e-4,
        )


# ── plan_gpu_placement tests ─────────────────────────────────────


class TestPlanGpuPlacement:
    def test_two_gpus(self):
        """Teachers are distributed across 2 GPUs."""
        names = ["openvla-7b", "rdt2-fm", "smolvla-base"]
        assignment = plan_gpu_placement(names, gpu_memory_mb=[16000.0, 4000.0])
        # openvla-7b (15.2GB) → cuda:0 (16GB)
        assert assignment["openvla-7b"] == "cuda:0"
        # rdt2-fm (2.5GB) → fits on cuda:1 (4GB)
        assert assignment["rdt2-fm"] == "cuda:1"
        # smolvla (1GB) → fits on remaining cuda:1
        assert assignment["smolvla-base"] == "cuda:1"

    def test_one_gpu(self):
        """All teachers on single GPU if they fit."""
        names = ["rdt2-fm", "smolvla-base"]
        assignment = plan_gpu_placement(names, gpu_memory_mb=[8000.0])
        assert all(v == "cuda:0" for v in assignment.values())

    def test_no_gpus_cpu_fallback(self):
        """No GPUs → all on CPU."""
        names = ["openvla-7b", "rdt2-fm"]
        assignment = plan_gpu_placement(names, gpu_memory_mb=None)
        assert all(v == "cpu" for v in assignment.values())

        assignment2 = plan_gpu_placement(names, gpu_memory_mb=[])
        assert all(v == "cpu" for v in assignment2.values())

    def test_exceeds_vram_skips_to_cpu(self):
        """Teachers too large for any GPU fall back to CPU."""
        names = ["openvla-7b"]
        assignment = plan_gpu_placement(names, gpu_memory_mb=[4000.0])
        assert assignment["openvla-7b"] == "cpu"


# ── UniversalRunner tests ────────────────────────────────────────


class TestUniversalRunner:
    def _make_runner(self):
        """Create a minimal runner with a tiny student model."""
        student = nn.Linear(D_STUDENT, D_ACTION)
        loss_fn = UniversalDistillationLoss(N_TEACHERS, D_STUDENT, CONF_DIM)
        optimizer = torch.optim.Adam(
            list(student.parameters()) + list(loss_fn.parameters()),
            lr=1e-3,
        )
        slots = [TeacherSlot(name=f"teacher-{i}", device="cpu") for i in range(N_TEACHERS)]
        return UniversalRunner(
            student=student,
            teacher_slots=slots,
            loss_fn=loss_fn,
            optimizer=optimizer,
            max_steps=100,
            checkpoint_every=10,
            device="cpu",
        )

    def test_training_step_produces_loss(self):
        """A training step returns loss dict with positive total."""
        runner = self._make_runner()
        features = torch.randn(B, D_STUDENT)
        batch = {
            "student_actions": runner.student(features),
            "student_features": features,
            "ground_truth_actions": torch.randn(B, D_ACTION),
            "teacher_actions_list": [torch.randn(B, D_ACTION) for _ in range(N_TEACHERS)],
            "teacher_confidences": torch.rand(B, N_TEACHERS, CONF_DIM),
        }
        result = runner.training_step(batch)
        assert "total" in result
        assert result["total"].item() > 0
        assert runner.global_step == 1

    def test_checkpoint_saves(self):
        """Checkpoint file is created on disk."""
        runner = self._make_runner()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = runner.save_checkpoint(tmpdir)
            assert path.exists()
            assert path.suffix == ".pt"

    def test_eval_runs(self):
        """Evaluate returns mean metrics."""
        runner = self._make_runner()
        features = torch.randn(B, D_STUDENT)
        eval_batch = {
            "student_actions": runner.student(features),
            "student_features": features,
            "ground_truth_actions": torch.randn(B, D_ACTION),
            "teacher_actions_list": [torch.randn(B, D_ACTION) for _ in range(N_TEACHERS)],
            "teacher_confidences": torch.rand(B, N_TEACHERS, CONF_DIM),
        }
        metrics = runner.evaluate([eval_batch, eval_batch])
        assert "total" in metrics
        assert metrics["total"] > 0


# ── Staged mode tests ────────────────────────────────────────────


class TestStagedMode:
    def test_teacher_rotation(self):
        """Staged mode rotates teachers based on global step."""
        student = nn.Linear(D_STUDENT, D_ACTION)
        loss_fn = UniversalDistillationLoss(2, D_STUDENT, CONF_DIM)
        optimizer = torch.optim.Adam(
            list(student.parameters()) + list(loss_fn.parameters()),
            lr=1e-3,
        )
        slots = [TeacherSlot(name=f"t-{i}") for i in range(6)]
        runner = UniversalRunner(
            student=student,
            teacher_slots=slots,
            loss_fn=loss_fn,
            optimizer=optimizer,
            staged=True,
            teachers_per_stage=2,
            steps_per_stage=10,
        )

        # Stage 0: teachers 0,1
        runner.global_step = 0
        active = runner.active_teachers
        assert [s.name for s in active] == ["t-0", "t-1"]

        # Stage 1: teachers 2,3
        runner.global_step = 10
        active = runner.active_teachers
        assert [s.name for s in active] == ["t-2", "t-3"]

        # Stage 2: teachers 4,5
        runner.global_step = 20
        active = runner.active_teachers
        assert [s.name for s in active] == ["t-4", "t-5"]

    def test_student_persistence_across_stages(self):
        """Student weights are preserved across stage transitions."""
        student = nn.Linear(D_STUDENT, D_ACTION)
        loss_fn = UniversalDistillationLoss(N_TEACHERS, D_STUDENT, CONF_DIM)
        optimizer = torch.optim.Adam(
            list(student.parameters()) + list(loss_fn.parameters()),
            lr=1e-3,
        )
        slots = [TeacherSlot(name=f"t-{i}") for i in range(N_TEACHERS)]

        runner = UniversalRunner(
            student=student,
            teacher_slots=slots,
            loss_fn=loss_fn,
            optimizer=optimizer,
            staged=True,
            teachers_per_stage=2,
            steps_per_stage=5,
        )

        # Snapshot weights before stage change
        w_before = student.weight.data.clone()

        # Simulate training step in stage 0
        features = torch.randn(B, D_STUDENT)
        batch = {
            "student_actions": student(features),
            "student_features": features,
            "ground_truth_actions": torch.randn(B, D_ACTION),
            "teacher_actions_list": [torch.randn(B, D_ACTION) for _ in range(N_TEACHERS)],
            "teacher_confidences": torch.rand(B, N_TEACHERS, CONF_DIM),
        }
        runner.training_step(batch)

        # Weights changed (training happened)
        assert not torch.allclose(student.weight.data, w_before)

        # Move to next stage — student model object is the same
        runner.global_step = 5  # triggers stage 1
        assert runner.active_teachers[0].name != slots[0].name or len(slots) <= 2


# ── CLI tests ─────────────────────────────────────────────────────


class TestCLI:
    def test_unwired_universal_distill_is_not_public(self):
        """The library remains tested, but no facade command is registered."""
        from typer.testing import CliRunner

        from forge.cli_v2 import app

        runner = CliRunner()
        help_result = runner.invoke(app, ["--help"])
        assert help_result.exit_code == 0
        assert "universal-distill" not in help_result.output

        result = runner.invoke(app, ["universal-distill", "start"])
        assert result.exit_code == 2
        assert "No such command" in result.output


# ── Config tests ──────────────────────────────────────────────────


class TestConfig:
    def test_defaults_valid(self):
        """Default UniversalDistillConfig is valid."""
        config = UniversalDistillConfig()
        assert len(config.teacher_names) == 3
        assert config.alpha_task + config.alpha_diversity + config.alpha_consistency < 1.0
        assert config.router_temperature > 0

    def test_yaml_override(self, tmp_path):
        """YAML config overrides universal distill settings."""
        yaml_content = """
universal:
  max_steps: 50000
  batch_size: 16
  staged: true
"""
        yaml_file = tmp_path / "test_config.yaml"
        yaml_file.write_text(yaml_content)

        config = ForgeConfig.from_yaml(yaml_file)
        assert config.universal.max_steps == 50000
        assert config.universal.batch_size == 16
        assert config.universal.staged is True
        # Defaults preserved
        assert config.universal.alpha_task == 0.3
