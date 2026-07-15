"""Tests for PRD-03: Knowledge Distillation Training Loop."""

import tempfile

import numpy as np
import pytest
import torch


def test_kd_loss():
    """Verify KD loss computation."""
    from forge.losses import kd_loss

    student = torch.randn(4, 7)
    teacher = torch.randn(4, 7)

    loss = kd_loss(student, teacher, temperature=4.0)
    assert loss.shape == ()
    assert loss.item() > 0

    # Same input → zero loss
    loss_same = kd_loss(student, student, temperature=4.0)
    assert loss_same.item() < 1e-6


def test_task_loss():
    """Verify task loss (MSE on ground truth)."""
    from forge.losses import task_loss

    pred = torch.randn(4, 7)
    gt = torch.randn(4, 7)

    loss = task_loss(pred, gt)
    assert loss.shape == ()
    assert loss.item() > 0

    loss_same = task_loss(pred, pred)
    assert loss_same.item() < 1e-6


def test_feature_alignment_loss():
    """Verify vision feature alignment loss."""
    from forge.losses import feature_alignment_loss

    student_feat = torch.randn(4, 64, 128)
    teacher_feat = torch.randn(4, 64, 128)

    loss = feature_alignment_loss(student_feat, teacher_feat)
    assert loss.shape == ()

    # Same features → loss should be ~0
    loss_same = feature_alignment_loss(student_feat, student_feat)
    assert loss_same.item() < 0.1  # cosine similarity ≈ 1


def test_action_distribution_loss():
    """Verify confidence-weighted action distribution loss."""
    from forge.losses import action_distribution_loss

    student = torch.randn(4, 7)
    teacher_mean = torch.randn(4, 7)
    teacher_std = torch.ones(4, 7) * 0.1
    confidence = torch.ones(4, 7) * 0.9

    loss = action_distribution_loss(student, teacher_mean, teacher_std, confidence)
    assert loss.shape == ()
    assert loss.item() > 0


def test_composite_loss():
    """Verify full composite distillation loss."""
    from forge.losses import ForgeDistillationLoss

    criterion = ForgeDistillationLoss(
        temperature=4.0,
        alpha_kd=0.4,
        alpha_task=0.3,
        alpha_feat=0.2,
        alpha_action=0.1,
    )

    losses = criterion(
        student_actions=torch.randn(4, 7),
        teacher_action_logits=torch.randn(4, 7),
        ground_truth_actions=torch.randn(4, 7),
        teacher_action_mean=torch.randn(4, 7),
        teacher_action_std=torch.ones(4, 7) * 0.1,
        teacher_confidence=torch.ones(4, 7) * 0.9,
    )

    assert "total" in losses
    assert "kd" in losses
    assert "task" in losses
    assert "action" in losses
    assert losses["total"].item() > 0


def test_teacher_dataset():
    """Verify TeacherLabelDataset loads episodes correctly."""
    import numpy as np

    from forge.data.label_writer import LabelWriter
    from forge.data.teacher_dataset import TeacherLabelDataset
    from forge.types import EpisodeData

    with tempfile.TemporaryDirectory() as tmpdir:
        writer = LabelWriter(tmpdir, episodes_per_file=10, save_vision_features=False)

        timesteps = 5
        for i in range(3):
            ep = EpisodeData(
                episode_id=f"ep_{i}",
                task_id=f"task_{i}",
                language_instruction=f"do thing {i}",
                timesteps=timesteps,
                images=np.random.randint(0, 255, (timesteps, 64, 64, 3), dtype=np.uint8),
                proprioception=np.random.randn(timesteps, 7).astype(np.float32),
                teacher_action_logits=np.random.randn(timesteps, 7).astype(np.float32),
                teacher_action_mean=np.random.randn(timesteps, 7).astype(np.float32),
                teacher_action_std=np.abs(np.random.randn(timesteps, 7)).astype(np.float32) + 0.01,
                teacher_vision_features=None,
                confidence=np.random.rand(timesteps, 7).astype(np.float32),
                ground_truth_actions=np.random.randn(timesteps, 7).astype(np.float32),
                success=True,
            )
            writer.write_episode(ep)
        writer.finalize()

        dataset = TeacherLabelDataset(tmpdir, image_size=64)
        assert len(dataset) == 3

        item = dataset[0]
        assert item["image"].shape == (3, 64, 64)
        assert item["teacher_action_logits"].shape == (7,)
        assert item["ground_truth_actions"].shape == (7,)
        assert isinstance(item["language_instruction"], str)
        dataset.close()


def test_teacher_dataset_preserves_action_chunks_and_future_ground_truth(tmp_path):
    """Chunk labels remain full horizon and receive aligned future ground truth."""
    from forge.data.label_writer import LabelWriter
    from forge.data.teacher_dataset import TeacherLabelDataset
    from forge.types import EpisodeData

    timesteps, horizon, action_dim = 3, 2, 4
    teacher = np.arange(timesteps * horizon * action_dim, dtype=np.float32).reshape(timesteps, horizon, action_dim)
    ground_truth = np.arange(timesteps * action_dim, dtype=np.float32).reshape(timesteps, action_dim)
    writer = LabelWriter(tmp_path)
    writer.write_episode(
        EpisodeData(
            episode_id="chunk",
            task_id="task",
            language_instruction="move",
            timesteps=timesteps,
            images=np.zeros((timesteps, 16, 16, 3), dtype=np.uint8),
            proprioception=np.zeros((timesteps, action_dim), dtype=np.float32),
            teacher_action_logits=teacher,
            teacher_action_mean=teacher,
            teacher_action_std=np.ones_like(teacher),
            teacher_vision_features=None,
            confidence=np.ones_like(teacher),
            ground_truth_actions=ground_truth,
            success=None,
        )
    )
    writer.finalize()

    dataset = TeacherLabelDataset(tmp_path, image_size=16, sample_timestep="first")
    item = dataset[0]
    dataset.close()

    assert item["teacher_action_logits"].shape == (horizon, action_dim)
    assert torch.equal(item["ground_truth_actions"], torch.from_numpy(ground_truth[:horizon]))


def test_action_batch_contract_rejects_chunk_labels_for_single_step_head():
    from forge.config import ForgeConfig
    from forge.distill import _validate_action_batch

    config = ForgeConfig.default()
    chunk = torch.zeros(2, 24, config.student.action_dim)
    with pytest.raises(ValueError, match="incompatible with 'diffusion' action head"):
        _validate_action_batch(
            config,
            teacher_logits=chunk,
            teacher_mean=chunk,
            teacher_std=chunk,
            confidence=chunk,
            ground_truth=chunk,
        )


def test_cosine_schedule():
    """Verify cosine schedule with warmup."""
    from forge.distill import get_cosine_schedule_with_warmup

    optimizer = torch.optim.SGD([torch.randn(1, requires_grad=True)], lr=1.0)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps=10, total_steps=100)

    lrs = []
    for _ in range(100):
        lrs.append(scheduler.get_last_lr()[0])
        optimizer.step()
        scheduler.step()

    # Warmup: LR increases
    assert lrs[0] < lrs[5]
    # After warmup: LR decreases
    assert lrs[20] > lrs[80]
    # End: LR near 0
    assert lrs[-1] < 0.1


def test_training_phases():
    """Verify phase transitions."""
    from forge.distill import _get_phase

    assert _get_phase(0, 1000) == 1  # Start: bridge warmup
    assert _get_phase(50, 1000) == 1  # Still warmup
    assert _get_phase(150, 1000) == 2  # Full distillation
    assert _get_phase(500, 1000) == 2  # Still full
    assert _get_phase(850, 1000) == 3  # Action fine-tune
    assert _get_phase(999, 1000) == 3  # End


def test_mini_training_loop(tmp_path):
    """End-to-end mini training loop (10 steps, mock data, CPU)."""
    from forge.config import ForgeConfig

    config = ForgeConfig.default()
    config.paths.data_dir = str(tmp_path / "data")
    config.paths.output_dir = str(tmp_path / "outputs")
    config.student.bridge_d_vision = 128
    config.student.bridge_d_model = 64
    config.student.bridge_n_queries = 8
    config.student.bridge_n_heads = 4
    config.student.bridge_n_layers = 2
    config.student.action_head_layers = 2
    config.student.action_diffusion_steps = 3
    config.student.lora_rank = 4
    config.student.lora_alpha = 8
    config.distill.batch_size = 2
    config.distill.gradient_accumulation_steps = 1
    config.distill.warmup_steps = 2
    config.distill.save_every = 5
    config.distill.eval_every = 5

    # Override model_dir to None so we use mocks (no real model loading)
    config.paths.model_dir = "/nonexistent"

    from forge.distill import train_forge

    progress = []
    summary = train_forge(config, device="cpu", max_steps=10, progress_callback=progress.append)

    assert summary["total_steps"] == 10
    assert summary["final_loss"] > 0
    assert summary["device"] == "cpu"
    assert progress[0]["step"] == 1
    assert progress[-1]["step"] == 10
    assert all(event["total_steps"] == 10 for event in progress)
    assert all(event["eta_seconds"] >= 0 for event in progress)
    assert (tmp_path / "outputs" / "checkpoints" / "final.pt").exists()
