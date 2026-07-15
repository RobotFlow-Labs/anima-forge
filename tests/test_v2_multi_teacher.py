"""PRD-12: Multi-Teacher & Multi-Path Distillation tests."""

import torch

from forge.multi_teacher import (
    MultiTeacherDistillationLoss,
    MultiTeacherDistiller,
    TeacherRouter,
)


def test_teacher_router_output_shape():
    """Router produces (B, N_teachers) output."""
    router = TeacherRouter(d_input=256, n_teachers=3)
    features = torch.randn(8, 256)
    weights = router(features)
    assert weights.shape == (8, 3)


def test_teacher_router_weights_sum_to_one():
    """Router weights sum to 1 along teacher dimension."""
    router = TeacherRouter(d_input=128, n_teachers=4, temperature=0.5)
    features = torch.randn(16, 128)
    weights = router(features)
    sums = weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_teacher_router_accepts_bfloat16_student_features():
    """Router normalizes mixed-precision feature inputs to its parameter dtype."""
    router = TeacherRouter(d_input=128, n_teachers=4)
    features = torch.randn(16, 128, dtype=torch.bfloat16)

    weights = router(features)

    assert weights.dtype == torch.float32
    assert weights.shape == (16, 4)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(16), atol=1e-5)


def test_multi_teacher_distiller_init():
    """MultiTeacherDistiller initializes with registry adapters."""
    # Ensure adapters are registered
    import forge.teachers.openvla_adapter  # noqa: F401
    import forge.teachers.rdt2_adapter  # noqa: F401

    distiller = MultiTeacherDistiller(
        teacher_names=["openvla-7b", "rdt2-fm"],
        model_dir="/tmp/nonexistent",
    )
    assert distiller.n_teachers == 2
    assert "openvla-7b" in distiller.teachers
    assert "rdt2-fm" in distiller.teachers


def test_multi_teacher_distiller_generate_labels():
    """generate_labels returns empty dict when no teachers loaded."""
    import forge.teachers.openvla_adapter  # noqa: F401

    distiller = MultiTeacherDistiller(
        teacher_names=["openvla-7b"],
        model_dir="/tmp/nonexistent",
    )
    # No teachers loaded (model dir doesn't exist)
    labels = distiller.generate_labels(torch.randn(3, 384, 384), "pick up the cup")
    assert isinstance(labels, dict)
    assert len(labels) == 0  # No loaded teachers


def test_multi_teacher_loss_forward():
    """MultiTeacherDistillationLoss produces correct output keys."""
    loss_fn = MultiTeacherDistillationLoss(n_teachers=3, d_student=256, temperature=4.0, alpha_task=0.3)
    student_actions = torch.randn(8, 7)
    teacher_actions_list = [torch.randn(8, 7) for _ in range(3)]
    gt_actions = torch.randn(8, 7)
    features = torch.randn(8, 256)

    out = loss_fn(student_actions, teacher_actions_list, gt_actions, features)

    assert "total" in out
    assert "kd" in out
    assert "task" in out
    assert "router_weights" in out
    assert out["total"].shape == ()
    assert out["router_weights"].shape == (8, 3)
    assert torch.isfinite(out["total"])


def test_multi_teacher_loss_gradient_flows():
    """Gradients flow through both router and student actions."""
    loss_fn = MultiTeacherDistillationLoss(n_teachers=2, d_student=64, alpha_task=0.3)

    student_actions = torch.randn(4, 7, requires_grad=True)
    teacher_actions_list = [torch.randn(4, 7), torch.randn(4, 7)]
    gt_actions = torch.randn(4, 7)
    features = torch.randn(4, 64, requires_grad=True)

    out = loss_fn(student_actions, teacher_actions_list, gt_actions, features)
    out["total"].backward()

    # Student actions get gradients from both KD and task loss
    assert student_actions.grad is not None
    assert torch.any(student_actions.grad != 0)

    # Features get gradients through the router
    assert features.grad is not None
    assert torch.any(features.grad != 0)

    # Router parameters get gradients
    for p in loss_fn.router.parameters():
        assert p.grad is not None


def test_multi_teacher_loss_single_teacher_degenerates():
    """With 1 teacher, router weights are always 1.0."""
    loss_fn = MultiTeacherDistillationLoss(n_teachers=1, d_student=64, alpha_task=0.3)

    student_actions = torch.randn(4, 7)
    teacher_actions_list = [torch.randn(4, 7)]
    gt_actions = torch.randn(4, 7)
    features = torch.randn(4, 64)

    out = loss_fn(student_actions, teacher_actions_list, gt_actions, features)

    # Single teacher → softmax always outputs 1.0
    assert torch.allclose(out["router_weights"], torch.ones(4, 1), atol=1e-5)


def test_multi_teacher_router_learns_routing():
    """Router can learn to prefer the better teacher through gradient descent."""
    torch.manual_seed(42)
    d_student = 32
    d_action = 7
    B = 16

    loss_fn = MultiTeacherDistillationLoss(
        n_teachers=2,
        d_student=d_student,
        alpha_task=0.0,  # Pure KD
    )
    optimizer = torch.optim.Adam(loss_fn.parameters(), lr=1e-2)

    # Teacher 1 is always close to GT, Teacher 2 is always far
    gt_actions = torch.randn(B, d_action)
    good_teacher = gt_actions + torch.randn(B, d_action) * 0.01  # Close
    bad_teacher = gt_actions + torch.randn(B, d_action) * 5.0  # Far

    features = torch.randn(B, d_student)

    # Train for a few steps
    for _ in range(50):
        student_actions = gt_actions + torch.randn(B, d_action) * 0.1
        out = loss_fn(
            student_actions,
            [good_teacher, bad_teacher],
            gt_actions,
            features,
        )
        optimizer.zero_grad()
        out["total"].backward()
        optimizer.step()

    # After training, router should prefer teacher 0 (good) over teacher 1 (bad)
    with torch.no_grad():
        final_weights = loss_fn.router(features)
        avg_weight_good = final_weights[:, 0].mean().item()
        avg_weight_bad = final_weights[:, 1].mean().item()

    assert avg_weight_good > avg_weight_bad, (
        f"Router should prefer good teacher: w_good={avg_weight_good:.3f}, w_bad={avg_weight_bad:.3f}"
    )
