"""Tests for TurboQuant and PolarQuant integration."""

import torch


def test_turboquant_mse_preserves_shape():
    from forge.turboquant import TurboQuantizer

    quantizer = TurboQuantizer(bits=3, mode="mse")
    x = torch.randn(12, 32)
    y = quantizer.quantize_dequantize(x)
    assert y.shape == x.shape


def test_turboquant_prod_preserves_shape():
    from forge.turboquant import TurboQuantizer

    quantizer = TurboQuantizer(bits=3, mode="prod")
    x = torch.randn(8, 16)
    y = quantizer.quantize_dequantize(x)
    assert y.shape == x.shape


def test_polarquant_preserves_shape():
    from forge.turboquant import PolarQuantizer

    quantizer = PolarQuantizer(bits=3)
    x = torch.randn(6, 18)
    y = quantizer.quantize_dequantize(x)
    assert y.shape == x.shape


def test_quantize_model_turboquant_method():
    import torch.nn as nn

    from forge.quantize import quantize_model

    model = nn.Sequential(nn.Linear(24, 12), nn.ReLU(), nn.Linear(12, 4))
    quantized = quantize_model(model, method="turboquant-mse", bits=3)
    assert quantized[0].weight.shape == model[0].weight.shape


def test_turboquant_inplace_uses_bounded_row_chunks(monkeypatch):
    import torch.nn as nn

    from forge.quantize import api
    from forge.turboquant import TurboQuantizer

    model = nn.Linear(32, 17, bias=False)
    observed_rows = []
    real_quantize = TurboQuantizer.quantize_weight

    def recording_quantize(self, rows):
        observed_rows.append(rows.shape[0])
        return real_quantize(self, rows)

    monkeypatch.setattr(TurboQuantizer, "quantize_weight", recording_quantize)
    quantized = api.quantize_model(
        model,
        method="turboquant-mse",
        bits=4,
        inplace=True,
        row_chunk_size=5,
    )

    assert quantized is model
    assert max(observed_rows) <= 5


def test_turboquant_large_projection_uses_structured_rotation():
    """LLM-width projections avoid cubic dense-QR allocation."""
    from forge.turboquant import TurboQuantizer

    quantizer = TurboQuantizer(bits=4, mode="mse", seed=7)
    weight = torch.randn(4, 1024)
    output = quantizer.quantize_weight(weight)
    state = quantizer._state_cache[(1024, "cpu")]

    assert output.shape == weight.shape
    assert state.rotation is None
    assert state.permutation is not None
    assert state.signs is not None
