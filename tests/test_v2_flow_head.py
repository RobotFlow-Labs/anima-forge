"""PRD-11: Flow Matching Action Head & Action Head Factory tests."""

import pytest
import torch

from forge.config import StudentConfig
from forge.modules.action_head_factory import create_action_head
from forge.modules.flow_head import (
    FlowBlock,
    FlowMatchingActionHead,
    SinusoidalTimeEmbedding,
)

# --- FlowMatchingActionHead tests ---


def test_flow_head_training_forward():
    """Training forward produces loss and predicted velocity."""
    head = FlowMatchingActionHead(d_model=128, d_action=7, n_layers=2, d_hidden=64)
    features = torch.randn(4, 128)
    gt_actions = torch.randn(4, 7)
    out = head(features, gt_actions=gt_actions)
    assert "loss" in out
    assert "predicted_velocity" in out
    assert out["loss"].shape == ()
    assert out["predicted_velocity"].shape == (4, 7)


def test_flow_head_inference_forward():
    """Inference forward produces actions with correct shape."""
    head = FlowMatchingActionHead(d_model=128, d_action=7, n_layers=2, d_hidden=64, inference_steps=4)
    features = torch.randn(4, 128)
    out = head(features, gt_actions=None)
    assert "actions" in out
    assert out["actions"].shape == (4, 7)


def test_flow_head_single_step_inference():
    """K=1 single-step inference works."""
    head = FlowMatchingActionHead(d_model=128, d_action=7, n_layers=2, d_hidden=64, inference_steps=1)
    features = torch.randn(2, 128)
    out = head(features)
    assert out["actions"].shape == (2, 7)


def test_flow_head_two_step_inference():
    """K=2 inference works."""
    head = FlowMatchingActionHead(d_model=128, d_action=7, n_layers=2, d_hidden=64, inference_steps=2)
    features = torch.randn(2, 128)
    out = head(features)
    assert out["actions"].shape == (2, 7)


def test_flow_head_param_count():
    """param_count returns positive integer."""
    head = FlowMatchingActionHead(d_model=128, d_action=7, n_layers=2, d_hidden=64)
    count = head.param_count()
    assert count > 0
    assert isinstance(count, int)


def test_flow_head_loss_is_finite():
    """Training loss is finite and positive."""
    head = FlowMatchingActionHead(d_model=128, d_action=7, n_layers=2, d_hidden=64)
    features = torch.randn(8, 128)
    gt_actions = torch.randn(8, 7)
    out = head(features, gt_actions=gt_actions)
    assert torch.isfinite(out["loss"])
    assert out["loss"].item() > 0


# --- SinusoidalTimeEmbedding tests ---


def test_sinusoidal_time_embedding_shape():
    """Time embedding produces correct output shape."""
    emb = SinusoidalTimeEmbedding(d_model=64)
    t = torch.rand(8)
    out = emb(t)
    assert out.shape == (8, 64)


def test_sinusoidal_time_embedding_odd_dim():
    """Odd d_model pads to correct size."""
    emb = SinusoidalTimeEmbedding(d_model=65)
    t = torch.rand(4)
    out = emb(t)
    assert out.shape == (4, 65)


# --- FlowBlock tests ---


def test_flow_block_residual():
    """FlowBlock produces output with correct shape."""
    block = FlowBlock(d_action=7, d_cond=64, d_hidden=64)
    actions = torch.randn(4, 7)
    cond = torch.randn(4, 64)
    out = block(actions, cond)
    assert out.shape == (4, 64)


# --- Action Head Factory tests ---


def test_factory_creates_diffusion_head():
    """Factory creates DiffusionActionHead for type='diffusion'."""
    config = StudentConfig(action_head_type="diffusion")
    head = create_action_head(config)
    from forge.modules.diffusion_head import DiffusionActionHead

    assert isinstance(head, DiffusionActionHead)


def test_factory_creates_flow_head():
    """Factory creates FlowMatchingActionHead for type='flow'."""
    config = StudentConfig(action_head_type="flow")
    head = create_action_head(config)
    assert isinstance(head, FlowMatchingActionHead)


def test_factory_creates_chunk_head():
    """Factory creates ActionChunkHead for type='chunk'."""
    config = StudentConfig(action_head_type="chunk", action_horizon=8)
    head = create_action_head(config)
    from forge.modules.action_chunking import ActionChunkHead

    assert isinstance(head, ActionChunkHead)


def test_factory_raises_on_unknown():
    """Factory raises ValueError for unknown head type."""
    config = StudentConfig(action_head_type="unknown")
    with pytest.raises(ValueError, match="Unknown action head type"):
        create_action_head(config)


def test_factory_default_is_diffusion():
    """Default config produces diffusion head (v1 compat)."""
    config = StudentConfig()
    assert config.action_head_type == "diffusion"
    head = create_action_head(config)
    from forge.modules.diffusion_head import DiffusionActionHead

    assert isinstance(head, DiffusionActionHead)
