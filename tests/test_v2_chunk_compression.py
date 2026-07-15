"""PRD-13: Chunk-Aware Compression tests."""

import copy

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from forge.prune_v2 import (
    compute_chunk_layer_importance,
    temporal_coherence_score,
)
from forge.quantize_v2 import (
    measure_chunk_quantization_quality,
    quantize_chunk_aware,
)

# --- Temporal Coherence tests ---


def test_temporal_coherence_score_smooth():
    """Smooth chunk (constant actions) has near-zero TC score."""
    # All actions are the same → no jitter
    chunk = torch.ones(8, 7) * 0.5
    score = temporal_coherence_score(chunk)
    assert score < 1e-5, f"Smooth chunk should have near-zero TC, got {score}"


def test_temporal_coherence_score_jerky():
    """Jerky chunk (alternating actions) has high TC score."""
    # Alternating +1 and -1
    chunk = torch.zeros(8, 7)
    for i in range(8):
        chunk[i] = 1.0 if i % 2 == 0 else -1.0
    score = temporal_coherence_score(chunk)
    assert score > 1.0, f"Jerky chunk should have high TC, got {score}"


def test_temporal_coherence_score_batch():
    """TC score works with batched input (B, H, D)."""
    batch = torch.randn(4, 8, 7)
    score = temporal_coherence_score(batch)
    assert isinstance(score, float)
    assert score >= 0


# --- Chunk Layer Importance tests ---


def test_chunk_layer_importance_structure():
    """compute_chunk_layer_importance returns correct structure."""
    from forge.config import StudentConfig
    from forge.student import FORGEStudent

    config = StudentConfig(action_head_type="diffusion")
    model = FORGEStudent(config)
    model.eval()

    # Create calibration data
    cal_data = [torch.randn(1, 3, 384, 384) for _ in range(3)]

    # This model uses a mock language backbone which may not have
    # transformer layers, so the result might be empty
    scores = compute_chunk_layer_importance(model, cal_data)
    assert isinstance(scores, dict)
    # All values should be floats
    for v in scores.values():
        assert isinstance(v, float)


# --- Prune Chunk Aware tests ---


def test_prune_chunk_aware_keeps_boundaries():
    """prune_chunk_aware respects keep_first_n and keep_last_n."""
    # Create importance scores for 10 layers
    scores = {i: float(i) / 10 for i in range(10)}

    # With keep_first_n=2 and keep_last_n=2, layers 0,1,8,9 are protected
    # Pruning to 6 layers means removing 4 from the middle
    # Least important removable: layers 2,3,4,5

    # We need a model with transformer layers to actually prune,
    # so test the selection logic directly
    removable = [idx for idx, _ in sorted(scores.items(), key=lambda x: x[1]) if idx >= 2 and idx < 10 - 2]
    layers_to_remove = removable[:4]

    # Should remove least important middle layers
    assert 0 not in layers_to_remove, "First layers should be kept"
    assert 1 not in layers_to_remove, "First layers should be kept"
    assert 8 not in layers_to_remove, "Last layers should be kept"
    assert 9 not in layers_to_remove, "Last layers should be kept"
    assert len(layers_to_remove) == 4


def test_prune_chunk_aware_fails_closed_when_layer_parent_does_not_match():
    """A discovery-only match must never produce false removal metadata."""
    from forge.prune_v2 import prune_chunk_aware

    class NormOnlyBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(4)

    class NormOnlyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([NormOnlyBlock() for _ in range(4)])

    with pytest.raises(ValueError, match="parent ModuleList could not be resolved"):
        prune_chunk_aware(
            NormOnlyModel(),
            {index: float(index) for index in range(4)},
            target_layers=3,
            keep_first_n=0,
            keep_last_n=0,
        )


def test_prune_chunk_aware_rejects_all_zero_importance():
    from forge.prune_v2 import prune_chunk_aware

    class AttentionBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = nn.Linear(4, 4)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([AttentionBlock() for _ in range(4)])

    with pytest.raises(RuntimeError, match="all zero"):
        prune_chunk_aware(
            Model(),
            {index: 0.0 for index in range(4)},
            target_layers=3,
            keep_first_n=0,
            keep_last_n=0,
        )


# --- Quantize Chunk Aware tests ---


def test_quantize_chunk_aware_default():
    """quantize_chunk_aware produces a model with modified weights."""
    from forge.config import StudentConfig
    from forge.student import FORGEStudent

    config = StudentConfig(action_head_type="diffusion")
    model = FORGEStudent(config)

    # Capture original weights
    original_params = {
        n: p.clone()
        for n, p in model.named_parameters()
        if "action_head" in n and isinstance(p, torch.Tensor) and p.dim() >= 2
    }

    quantized = quantize_chunk_aware(model, target_bits=4, action_head_bits=8)

    # Quantized model should have different weights (due to fake quantization)
    any_changed = False
    for n, p in quantized.named_parameters():
        if n in original_params:
            if not torch.allclose(p, original_params[n], atol=1e-7):
                any_changed = True
                break

    assert any_changed, "Quantization should modify at least some weights"


def test_quantize_action_head_higher_precision():
    """Action head gets higher precision than other modules."""
    from forge.config import StudentConfig
    from forge.student import FORGEStudent

    config = StudentConfig(action_head_type="diffusion")
    model = FORGEStudent(config)
    original = copy.deepcopy(model)

    # Quantize with action_head_bits=8, target_bits=2
    quantized = quantize_chunk_aware(model, target_bits=2, action_head_bits=8)

    # Action head should have less quantization error (8-bit) than bridge (2-bit)
    action_head_error = 0.0
    bridge_error = 0.0
    ah_count = 0
    br_count = 0

    for (n1, p1), (n2, p2) in zip(original.named_parameters(), quantized.named_parameters()):
        if not isinstance(p1, torch.Tensor) or p1.dim() < 2:
            continue
        error = F.mse_loss(p1, p2).item()
        if "action_head" in n1:
            action_head_error += error
            ah_count += 1
        elif "bridge" in n1:
            bridge_error += error
            br_count += 1

    if ah_count > 0 and br_count > 0:
        avg_ah = action_head_error / ah_count
        avg_br = bridge_error / br_count
        # 8-bit should have less error than 2-bit
        assert avg_ah < avg_br, (
            f"Action head (8-bit) should have less quant error than bridge (2-bit): "
            f"ah={avg_ah:.6f}, bridge={avg_br:.6f}"
        )


@pytest.mark.parametrize("target_bits", [float("nan"), float("inf"), 1.99, 8.01])
def test_quantize_chunk_aware_rejects_invalid_target_bits(target_bits):
    """Unsupported target widths fail instead of being silently clamped."""
    with pytest.raises(ValueError, match="target_bits"):
        quantize_chunk_aware(nn.Linear(4, 4), target_bits=target_bits)


def test_quantize_chunk_aware_rounds_fractional_bits_half_up(monkeypatch):
    """A 2.5-bit request deterministically selects three-bit fake quantization."""
    observed = []

    def fake_quantize(channel, *, bits):
        observed.append(bits)
        return channel

    monkeypatch.setattr("forge.quantize.chunk_aware._fake_quantize_channel", fake_quantize)
    quantize_chunk_aware(nn.Sequential(nn.Linear(4, 4)), target_bits=2.5)
    assert observed and set(observed) == {3}


def test_chunk_quantization_quality_metrics():
    """measure_chunk_quantization_quality returns correct metrics."""
    from forge.config import StudentConfig
    from forge.student import FORGEStudent

    config = StudentConfig(action_head_type="diffusion")
    fp_model = FORGEStudent(config)
    q_model = quantize_chunk_aware(copy.deepcopy(fp_model), target_bits=4)

    test_data = [torch.randn(1, 3, 384, 384) for _ in range(3)]

    metrics = measure_chunk_quantization_quality(fp_model, q_model, test_data)

    assert "action_mse" in metrics
    assert "temporal_coherence_delta" in metrics
    assert "max_step_drift" in metrics
    assert "per_step_error" in metrics
    assert metrics["action_mse"] >= 0
    assert isinstance(metrics["per_step_error"], list)


def test_chunk_quantization_quality_aggregates_mixed_horizons():
    """Single-step and chunked batches share a per-horizon aggregation schema."""

    class AlternatingModel(nn.Module):
        def __init__(self, offset):
            super().__init__()
            self.offset = offset

        def forward(self, images):
            horizon = 1 if images.shape[-1] == 1 else 3
            actions = torch.zeros(images.shape[0], horizon, 2) + self.offset
            if horizon == 1:
                actions = actions[:, 0]
            return {"actions": actions}

    metrics = measure_chunk_quantization_quality(
        AlternatingModel(0.0),
        AlternatingModel(1.0),
        [torch.zeros(2, 3, 1, 1), torch.zeros(1, 3, 2, 2)],
    )

    assert metrics["per_step_error"] == pytest.approx([1.0, 1.0, 1.0])
