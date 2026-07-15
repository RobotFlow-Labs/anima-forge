"""Tests for PRD-04: Layer Pruning (Shallow-Pi)."""

import pytest
import torch
import torch.nn as nn


class MockTransformerLayer(nn.Module):
    """Mock transformer layer with attention + FFN."""

    def __init__(self, d_model: int = 64):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, 4, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model))
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ffn(self.norm2(x))
        return x


class MockTransformerModel(nn.Module):
    """Mock model with transformer layers for pruning tests."""

    def __init__(self, n_layers: int = 12, d_model: int = 64):
        super().__init__()
        self.embed = nn.Linear(3 * 16 * 16, d_model)
        self.layers = nn.ModuleList([MockTransformerLayer(d_model) for _ in range(n_layers)])
        self.head = nn.Linear(d_model, 7)

    def forward(self, images, **kwargs):
        B = images.shape[0]
        x = images.view(B, -1)[:, : 3 * 16 * 16]
        x = self.embed(x).unsqueeze(1)  # (B, 1, d_model)
        for layer in self.layers:
            x = layer(x)
        actions = self.head(x.squeeze(1))
        return {"actions": actions}


def test_find_transformer_layers():
    """Verify we can find transformer layers in a model."""
    from forge.prune import _find_transformer_layers

    model = MockTransformerModel(n_layers=12)
    layers = _find_transformer_layers(model)
    assert len(layers) == 12


def test_normalize_scores():
    """Verify score normalization."""
    from forge.prune import _normalize

    scores = {0: 1.0, 1: 5.0, 2: 3.0, 3: 10.0}
    normalized = _normalize(scores)

    assert normalized[0] == pytest.approx(0.0)  # min
    assert normalized[3] == pytest.approx(1.0)  # max
    assert 0 <= normalized[1] <= 1
    assert 0 <= normalized[2] <= 1


def test_decoder_identity_preserves_tuple_contract():
    """HF decoder layers return tuples; identity replacement must do the same."""
    from forge.prune import _identity_forward_for_observed_output

    hidden = torch.randn(2, 3, 4)
    auxiliary = torch.randn(2, 3, 4)
    identity = _identity_forward_for_observed_output((hidden + 1, auxiliary))

    replacement = identity(hidden)

    assert isinstance(replacement, tuple)
    assert torch.equal(replacement[0], hidden)
    assert torch.equal(replacement[1], auxiliary)


def test_real_qwen_decoder_importance_is_nonzero_and_finite():
    """Production decoder tuple contracts survive identity-based scoring."""
    import math

    from transformers import Qwen3Config, Qwen3Model

    from forge.prune import _compute_action_sensitivity, _find_transformer_layers, compute_layer_importance
    from forge.prune_v2 import compute_chunk_layer_importance

    class TinyQwenStudentBackbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            config = Qwen3Config(
                vocab_size=32,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=4,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=8,
                max_position_embeddings=32,
            )
            self.input_projection = nn.Linear(3, 32)
            self.language = Qwen3Model(config)
            self.action_head = nn.Linear(32, 14)

        def forward(self, images):
            tokens = images.flatten(2).transpose(1, 2)[:, :8]
            hidden = self.input_projection(tokens)
            decoded = self.language(inputs_embeds=hidden, use_cache=False)
            actions = self.action_head(decoded.last_hidden_state.mean(dim=1))
            return {"actions": actions.view(images.shape[0], 2, 7)}

    torch.manual_seed(7)
    model = TinyQwenStudentBackbone().eval()
    calibration_batches = [torch.randn(1, 3, 4, 4), torch.randn(1, 3, 4, 4)]

    standard = compute_layer_importance(
        model,
        [{"image": batch} for batch in calibration_batches],
        n_samples=2,
    )
    action_sensitivity = _compute_action_sensitivity(
        model,
        _find_transformer_layers(model),
        [{"image": batch} for batch in calibration_batches],
        n_samples=2,
    )
    chunk = compute_chunk_layer_importance(model, calibration_batches)

    for scores in (standard, chunk):
        assert set(scores) == {0, 1, 2, 3}
        assert all(math.isfinite(score) for score in scores.values())
        assert any(score > 0.0 for score in scores.values())
    assert any(score > 0.0 for score in action_sensitivity.values())
    assert len({round(score, 6) for score in chunk.values()}) > 1


def test_prune_rejects_all_zero_importance():
    from forge.config import PruningConfig
    from forge.prune import prune_layers

    model = MockTransformerModel(n_layers=6)

    with pytest.raises(RuntimeError, match="all zero"):
        prune_layers(
            model,
            {index: 0.0 for index in range(6)},
            PruningConfig(target_layers=5, keep_first_n=0, keep_last_n=0),
        )


def test_prune_layers():
    """Verify layer removal."""
    from forge.config import PruningConfig
    from forge.prune import prune_layers

    model = MockTransformerModel(n_layers=12)

    # Fake scores — layers 3, 4, 5 are least important
    scores = {i: float(i) for i in range(12)}
    scores[3] = 0.1
    scores[4] = 0.2
    scores[5] = 0.3

    config = PruningConfig(target_layers=8, keep_first_n=2, keep_last_n=2)
    pruned, removed = prune_layers(model, scores, config)

    # Should have removed 4 layers (12 → 8)
    pruned_layers = _find_transformer_layers_from_model(pruned)
    assert len(pruned_layers) == 8
    assert len(removed) == 4


def test_prune_preserves_forward():
    """Verify pruned model still runs forward pass."""
    from forge.config import PruningConfig
    from forge.prune import prune_layers

    model = MockTransformerModel(n_layers=12)
    scores = {i: float(i) for i in range(12)}

    config = PruningConfig(target_layers=6, keep_first_n=1, keep_last_n=1)
    pruned, removed = prune_layers(model, scores, config)

    # Forward pass should work
    images = torch.randn(2, 3, 32, 32)
    out = pruned(images)
    assert out["actions"].shape == (2, 7)


def test_prune_no_pruning_needed():
    """Verify no-op when target >= current layers."""
    from forge.config import PruningConfig
    from forge.prune import prune_layers

    model = MockTransformerModel(n_layers=8)
    scores = {i: float(i) for i in range(8)}

    config = PruningConfig(target_layers=8)
    pruned, removed = prune_layers(model, scores, config)
    assert len(removed) == 0


def test_prune_keeps_first_last():
    """Verify first and last layers are always kept."""
    from forge.config import PruningConfig
    from forge.prune import prune_layers

    model = MockTransformerModel(n_layers=10)

    # Make first and last layers the LEAST important
    scores = {i: 5.0 for i in range(10)}
    scores[0] = 0.0  # Least important
    scores[1] = 0.1
    scores[8] = 0.1
    scores[9] = 0.0  # Least important

    config = PruningConfig(target_layers=6, keep_first_n=2, keep_last_n=2)
    pruned, removed = prune_layers(model, scores, config)

    # Layers 0, 1, 8, 9 should NOT be removed
    assert 0 not in removed
    assert 1 not in removed
    assert 8 not in removed
    assert 9 not in removed


def test_recovery_finetune_requires_real_labels(tmp_path):
    """Recovery must never create synthetic labels without explicit opt-in."""
    from forge.config import ForgeConfig
    from forge.errors import ForgeDataNotFoundError
    from forge.prune import recovery_finetune

    config = ForgeConfig()
    config.paths.data_dir = str(tmp_path / "data")
    config.student.allow_mock = False

    with pytest.raises(ForgeDataNotFoundError, match="Recovery fine-tuning requires"):
        recovery_finetune(MockTransformerModel(), config, max_steps=1)


def test_recovery_finetune_rejects_mock_label_metadata(tmp_path):
    """An existing mock-labelled directory is not trusted implicitly."""
    from forge.config import ForgeConfig
    from forge.distill import _create_mock_dataset
    from forge.errors import ForgeDataNotFoundError
    from forge.prune import recovery_finetune

    config = ForgeConfig()
    config.paths.data_dir = str(tmp_path / "data")
    config.student.allow_mock = False
    dataset = _create_mock_dataset(tmp_path / "data" / "teacher_labels", n_episodes=1)
    dataset.close()

    with pytest.raises(ForgeDataNotFoundError, match="mock-derived or untrusted"):
        recovery_finetune(MockTransformerModel(), config, max_steps=1)


def _find_transformer_layers_from_model(model):
    """Helper to count layers after pruning."""
    from forge.prune import _find_transformer_layers

    return _find_transformer_layers(model)
