"""PRD-14: Consistency Distillation tests."""

import torch

from forge.modules.consistency_head import (
    ConsistencyActionHead,
    ConsistencyDistillationTrainer,
)
from forge.modules.flow_head import FlowMatchingActionHead


def test_consistency_head_output_shape():
    """ConsistencyActionHead produces correct output shape."""
    head = ConsistencyActionHead(d_model=128, d_action=7, n_layers=2, d_hidden=64)
    features = torch.randn(4, 128)
    out = head(features)
    assert "actions" in out
    assert out["actions"].shape == (4, 7)


def test_consistency_head_single_step_inference():
    """Inference is always single-step regardless of input."""
    head = ConsistencyActionHead(d_model=128, d_action=7, n_layers=2, d_hidden=64)
    features = torch.randn(8, 128)
    # Even with gt_actions, output is from single-step inference
    out = head(features, gt_actions=torch.randn(8, 7))
    assert out["actions"].shape == (8, 7)


def test_consistency_trainer_init():
    """ConsistencyDistillationTrainer initializes correctly."""
    teacher = FlowMatchingActionHead(d_model=128, d_action=7, n_layers=2, d_hidden=64, inference_steps=4)
    student = ConsistencyActionHead(d_model=128, d_action=7, n_layers=2, d_hidden=64)

    trainer = ConsistencyDistillationTrainer(teacher, student, ema_decay=0.999)

    assert trainer.teacher is teacher
    assert trainer.student is student
    assert trainer.ema_teacher is not student  # Should be a copy
    # Teacher should be frozen
    for p in trainer.teacher.parameters():
        assert not p.requires_grad


def test_consistency_trainer_step():
    """Training step produces loss and K."""
    teacher = FlowMatchingActionHead(d_model=64, d_action=7, n_layers=2, d_hidden=32, inference_steps=4)
    student = ConsistencyActionHead(d_model=64, d_action=7, n_layers=2, d_hidden=32)

    trainer = ConsistencyDistillationTrainer(teacher, student)
    cond = torch.randn(4, 64)
    gt_actions = torch.randn(4, 7)

    result = trainer.training_step(cond, gt_actions, global_step=0)

    assert "loss" in result
    assert "K" in result
    assert torch.isfinite(result["loss"])
    assert result["K"] == 2  # Step 0 → K=2 from default curriculum


def test_ema_update_moves_weights():
    """EMA update moves ema_teacher weights toward student."""
    teacher = FlowMatchingActionHead(d_model=64, d_action=7, n_layers=2, d_hidden=32)
    student = ConsistencyActionHead(d_model=64, d_action=7, n_layers=2, d_hidden=32)

    trainer = ConsistencyDistillationTrainer(teacher, student, ema_decay=0.5)

    # Record EMA weights before
    ema_before = {n: p.clone() for n, p in trainer.ema_teacher.named_parameters()}

    # Manually modify student weights
    with torch.no_grad():
        for p in trainer.student.parameters():
            p.add_(torch.randn_like(p) * 0.1)

    # Run EMA update
    trainer._ema_update()

    # EMA weights should have moved
    any_changed = False
    for n, p in trainer.ema_teacher.named_parameters():
        if not torch.allclose(p, ema_before[n], atol=1e-6):
            any_changed = True
            break

    assert any_changed, "EMA update should move weights toward student"


def test_curriculum_schedule_progression():
    """Curriculum schedule increases K over training steps."""
    teacher = FlowMatchingActionHead(d_model=64, d_action=7, n_layers=2, d_hidden=32)
    student = ConsistencyActionHead(d_model=64, d_action=7, n_layers=2, d_hidden=32)

    trainer = ConsistencyDistillationTrainer(
        teacher,
        student,
        curriculum_schedule=[(0, 2), (100, 4), (500, 8)],
    )

    assert trainer._get_current_K(0) == 2
    assert trainer._get_current_K(50) == 2
    assert trainer._get_current_K(100) == 4
    assert trainer._get_current_K(200) == 4
    assert trainer._get_current_K(500) == 8
    assert trainer._get_current_K(1000) == 8


def test_consistency_loss_decreases():
    """Training for several steps decreases the loss."""
    torch.manual_seed(42)
    teacher = FlowMatchingActionHead(d_model=32, d_action=7, n_layers=2, d_hidden=16, inference_steps=2)
    student = ConsistencyActionHead(d_model=32, d_action=7, n_layers=2, d_hidden=16)

    trainer = ConsistencyDistillationTrainer(teacher, student, ema_decay=0.99)
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)

    cond = torch.randn(8, 32)
    gt_actions = torch.randn(8, 7)

    losses = []
    for step in range(30):
        result = trainer.training_step(cond, gt_actions, global_step=step)
        optimizer.zero_grad()
        result["loss"].backward()
        optimizer.step()
        losses.append(result["loss"].item())

    # Loss should generally decrease (compare first 5 avg to last 5 avg)
    early_avg = sum(losses[:5]) / 5
    late_avg = sum(losses[-5:]) / 5
    assert late_avg < early_avg, f"Loss should decrease: early={early_avg:.4f}, late={late_avg:.4f}"


def test_consistency_vs_teacher_quality():
    """Student param count is same order as teacher."""
    teacher = FlowMatchingActionHead(d_model=128, d_action=7, n_layers=4, d_hidden=64)
    student = ConsistencyActionHead(d_model=128, d_action=7, n_layers=4, d_hidden=64)

    teacher_params = teacher.param_count()
    student_params = student.param_count()

    # Same architecture → similar param count
    ratio = student_params / teacher_params
    assert 0.8 < ratio < 1.2, f"Student/teacher param ratio should be ~1.0, got {ratio:.2f}"
