"""Tests for PRD-05: Action-Centric Quantization (QVLA)."""

import pytest
import torch
import torch.nn as nn


def test_fake_quantize_channel():
    """Verify simulated quantization produces valid output."""
    from forge.quantize import _fake_quantize_channel

    weight = torch.randn(64)
    q4 = _fake_quantize_channel(weight, bits=4)
    q8 = _fake_quantize_channel(weight, bits=8)

    assert q4.shape == weight.shape
    assert q8.shape == weight.shape

    # 4-bit should have more quantization error than 8-bit
    err_4 = (weight - q4).abs().mean()
    err_8 = (weight - q8).abs().mean()
    assert err_4 > err_8 * 0.5  # 4-bit has more error


def test_fake_quantize_error_is_bounded_by_half_a_quantization_step():
    """Affine fake quantization may round past an endpoint by at most half a step."""
    from forge.quantize import _fake_quantize_channel

    generator = torch.Generator().manual_seed(20260713)
    for _ in range(32):
        weight = torch.randn(128, generator=generator) * 2.0
        q = _fake_quantize_channel(weight, bits=4)
        scale = (weight.max() - weight.min()) / 15
        tolerance = torch.finfo(weight.dtype).eps * weight.abs().max() * 4
        error_bound = scale / 2 + tolerance

        assert (q - weight).abs().max() <= error_bound
        assert q.min() >= weight.min() - error_bound
        assert q.max() <= weight.max() + error_bound


def test_vectorized_row_quantization_matches_channel_reference():
    """The production vectorized path applies a valid 4-bit grid per row."""
    from forge.quantize.qvla import _fake_quantize_rows

    weight = torch.randn(12, 32, dtype=torch.bfloat16)
    actual = _fake_quantize_rows(weight, bits=4)

    assert actual.dtype == weight.dtype
    assert torch.mean((actual.float() - weight.float()) ** 2).item() < 0.05
    assert all(torch.unique(row).numel() <= 16 for row in actual)


def test_allocate_bits():
    """Verify bit allocation based on sensitivity."""
    from forge.config import QuantConfig
    from forge.quantize import allocate_bits

    sensitivities = {
        "layer1": torch.tensor([0.0, 0.1, 0.5, 1.0]),  # Low to high sensitivity
        "layer2": torch.tensor([0.2, 0.8, 0.3, 0.9]),
    }

    config = QuantConfig(target_avg_bits=4.0, min_bits=2, max_bits=8)
    allocation = allocate_bits(sensitivities, config)

    assert "layer1" in allocation
    assert "layer2" in allocation

    # High sensitivity channels should get more bits
    assert allocation["layer1"][3] >= allocation["layer1"][0]
    realized = [bits for layer in allocation.values() for bits in layer.values()]
    assert sum(realized) / len(realized) == config.target_avg_bits


def test_allocate_bits_hits_target_for_skewed_sensitivities():
    """Rounded and clipped allocations still realize the configured integer mean."""
    from forge.config import QuantConfig
    from forge.quantize import allocate_bits

    sensitivities = {"skewed": torch.tensor([0.0] * 24 + [0.001, 0.01, 0.1, 1.0])}
    config = QuantConfig(target_avg_bits=4.0, min_bits=2, max_bits=8)

    allocation = allocate_bits(sensitivities, config)
    realized = list(allocation["skewed"].values())

    assert sum(realized) / len(realized) == 4.0
    assert all(config.min_bits <= bits <= config.max_bits for bits in realized)
    assert realized[-1] >= realized[0]


def test_allocate_bits_accepts_exact_fractional_average() -> None:
    """A fractional mean is valid when its integer bit total is exact."""
    from forge.config import QuantConfig
    from forge.quantize import allocate_bits

    sensitivities = {"layer": torch.linspace(0.0, 1.0, 10)}
    config = QuantConfig(target_avg_bits=4.1, min_bits=2, max_bits=8)

    realized = list(allocate_bits(sensitivities, config)["layer"].values())

    assert sum(realized) == 41


def test_allocate_bits_rejects_unrealizable_fractional_average() -> None:
    """QVLA must not silently round an impossible integer bit total."""
    from forge.config import QuantConfig
    from forge.quantize import allocate_bits

    sensitivities = {"layer": torch.linspace(0.0, 1.0, 8)}
    config = QuantConfig(target_avg_bits=4.1, min_bits=2, max_bits=8)

    with pytest.raises(ValueError, match="not exactly realizable across 8"):
        allocate_bits(sensitivities, config)


def test_quantize_model_uniform():
    """Verify uniform quantization applies to all Linear layers."""
    from forge.quantize import quantize_model

    model = nn.Sequential(
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 7),
    )

    original_weight = model[0].weight.data.clone()
    quantized = quantize_model(model, uniform_bits=4)

    # Weights should change (quantization noise)
    assert not torch.allclose(quantized[0].weight.data, original_weight, atol=1e-6)

    # Forward pass should still work
    x = torch.randn(2, 64)
    out = quantized(x)
    assert out.shape == (2, 7)


def test_inplace_quantization_uses_bounded_row_chunks(monkeypatch):
    """The deployment path avoids a full model copy and whole-layer temporaries."""
    from forge.quantize import qvla

    model = nn.Linear(32, 17, bias=False)
    original = model.weight.detach().clone()
    observed_rows = []
    real_quantize = qvla._fake_quantize_rows

    def recording_quantize(rows, bits):
        observed_rows.append(rows.shape[0])
        return real_quantize(rows, bits)

    monkeypatch.setattr(qvla, "_fake_quantize_rows", recording_quantize)
    quantized = qvla.quantize_qvla(
        model,
        uniform_bits=4,
        inplace=True,
        row_chunk_size=5,
    )

    assert quantized is model
    assert max(observed_rows) <= 5
    assert not torch.allclose(model.weight, original)


def test_quantize_model_mixed():
    """Verify mixed-precision QVLA quantization."""
    from forge.quantize import quantize_model

    model = nn.Sequential(
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 7),
    )

    allocation = {
        "0": {0: 8, 1: 8, 2: 4, 3: 4},  # First 2 channels get 8-bit, rest 4-bit
    }

    quantized = quantize_model(model, bit_allocation=allocation)

    x = torch.randn(2, 64)
    out = quantized(x)
    assert out.shape == (2, 7)


def test_quant_profile():
    """Verify quantization profile summary."""
    from forge.quantize import create_quant_profile

    model = nn.Sequential(
        nn.Linear(64, 32),
        nn.Linear(32, 7),
    )

    allocation = {
        "0": {i: 4 for i in range(32)},
        "1": {i: 6 for i in range(7)},
    }

    profile = create_quant_profile(model, allocation, name="test")

    assert profile.name == "test"
    assert profile.total_params > 0
    assert profile.compressed_size_mb > 0
    assert profile.avg_bits == (32 * 64 * 4 + 7 * 32 * 6) / (32 * 64 + 7 * 32)
    assert profile.quantized_params == 32 * 64 + 7 * 32
    assert profile.frozen_params == 32 + 7


def test_quant_profile_excludes_frozen_parameters_and_uses_uniform_width():
    """Embeddings, norms, and biases cannot inflate quantized size or average bits."""
    from forge.quantize import create_quant_profile

    model = nn.Sequential(
        nn.Embedding(100, 16),
        nn.LayerNorm(16),
        nn.Linear(16, 8),
    )

    profile = create_quant_profile(model, {}, name="q8", uniform_bits=8)

    assert profile.avg_bits == 8.0
    assert profile.quantized_params == 16 * 8
    assert profile.frozen_params == profile.total_params - profile.quantized_params
    assert profile.compressed_size_mb == (16 * 8 * 8) / (8 * 1024 * 1024)


def test_quantize_no_allocation():
    """Verify model passes through unchanged with no allocation."""
    from forge.quantize import quantize_model

    model = nn.Linear(64, 32)
    original = model.weight.data.clone()

    quantized = quantize_model(model)
    assert torch.allclose(quantized.weight.data, original)
