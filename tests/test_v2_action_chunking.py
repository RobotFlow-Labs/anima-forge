"""PRD-09: Action Chunking & Temporal Prediction tests."""

import torch


def test_action_chunk_head_shapes():
    """ActionChunkHead outputs (B, H, D_action) for various horizons."""
    from forge.modules.action_chunking import ActionChunkHead

    for H in [1, 4, 8, 16]:
        head = ActionChunkHead(d_model=64, d_action=7, horizon=H, n_layers=1, n_heads=4, d_hidden=32)
        features = torch.randn(2, 64)
        out = head(features)
        assert out["actions"].shape == (2, H, 7), f"Expected (2, {H}, 7), got {out['actions'].shape}"


def test_action_chunk_head_v1_compat():
    """H=1 produces (B, 1, D_action) — backward compatible."""
    from forge.modules.action_chunking import ActionChunkHead

    head = ActionChunkHead(d_model=64, d_action=7, horizon=1, n_layers=1, n_heads=4, d_hidden=32)
    features = torch.randn(4, 64)
    out = head(features)
    assert out["actions"].shape == (4, 1, 7)


def test_temporal_position_embeddings():
    """Each horizon step gets unique temporal embedding."""
    from forge.modules.action_chunking import ActionChunkHead

    head = ActionChunkHead(d_model=64, d_action=7, horizon=8, n_layers=1, n_heads=4, d_hidden=32)
    # Check temporal embeddings are unique per position
    embeddings = head.temporal_pos_embed.weight.data
    assert embeddings.shape == (8, 32)
    # No two embeddings should be identical (random init)
    for i in range(8):
        for j in range(i + 1, 8):
            assert not torch.allclose(embeddings[i], embeddings[j])


def test_chunk_weighted_loss_decay():
    """Loss weights decay exponentially over horizon."""
    from forge.modules.action_chunking import chunk_weighted_loss

    B, H, D = 4, 8, 7
    predicted = torch.randn(B, H, D)
    target = torch.randn(B, H, D)

    loss = chunk_weighted_loss(predicted, target, H, decay_factor=0.95)
    assert loss.dim() == 0  # Scalar
    assert loss.item() > 0


def test_chunk_weighted_loss_v1_compat():
    """H=1 chunk loss equals standard MSE (approximately)."""
    from forge.modules.action_chunking import chunk_weighted_loss

    B, D = 4, 7
    predicted = torch.randn(B, 1, D)
    target = torch.randn(B, 1, D)

    chunk_loss = chunk_weighted_loss(predicted, target, horizon=1)
    mse_loss = torch.nn.functional.mse_loss(predicted, target)

    # With H=1, chunk loss should equal MSE (weight is [1.0])
    assert torch.allclose(chunk_loss, mse_loss, atol=1e-6)


def test_blend_action_chunks_single():
    """Single chunk returns unchanged."""
    from forge.modules.action_chunking import blend_action_chunks

    chunk = torch.randn(8, 7)
    result = blend_action_chunks([chunk], overlap=2)
    assert torch.allclose(result, chunk)


def test_blend_action_chunks_overlap():
    """Overlapping chunks blend smoothly."""
    from forge.modules.action_chunking import blend_action_chunks

    H, D = 8, 7
    overlap = 2
    chunks = [torch.randn(H, D) for _ in range(3)]

    result = blend_action_chunks(chunks, overlap=overlap)
    step = H - overlap
    expected_T = H + step * 2  # H + step * (n_chunks - 1)
    assert result.shape == (expected_T, D)


def test_blend_action_chunks_no_overlap():
    """Zero overlap concatenates chunks."""
    from forge.modules.action_chunking import blend_action_chunks

    H, D = 4, 7
    c1 = torch.ones(H, D) * 1.0
    c2 = torch.ones(H, D) * 2.0

    result = blend_action_chunks([c1, c2], overlap=0)
    assert result.shape == (8, D)
    assert torch.allclose(result[:4], c1)
    assert torch.allclose(result[4:], c2)


def test_student_chunked_forward():
    """Student with action_horizon=8 and chunk head returns (B, 8, 7)."""
    from forge.config import StudentConfig
    from forge.modules.action_chunking import ActionChunkHead

    config = StudentConfig(action_horizon=8)
    head = ActionChunkHead(
        d_model=config.bridge_d_model,
        d_action=config.action_dim,
        horizon=config.action_horizon,
        n_layers=2,
        d_hidden=64,
    )
    features = torch.randn(2, config.bridge_d_model)
    out = head(features)
    assert out["actions"].shape == (2, 8, 7)


def test_chunk_aware_kd_loss():
    """Chunk-aware KD loss works with action chunks."""
    from forge.losses import chunk_aware_kd_loss

    B, H, D = 4, 8, 7
    student = torch.randn(B, H, D)
    teacher = torch.randn(B, H, D)

    loss = chunk_aware_kd_loss(student, teacher, temperature=4.0)
    assert loss.dim() == 0
    assert loss.item() > 0

    # Also works with 2D (single step, v1 compat)
    loss_2d = chunk_aware_kd_loss(student[:, 0], teacher[:, 0], temperature=4.0)
    assert loss_2d.dim() == 0
