"""Platform-independent behavioral tests for the mandatory MLX quantizer."""

from __future__ import annotations

import numpy as np
import pytest

from forge.turboquant.mlx_backend import MLXTurboQuantizer


class _NumpyMLX:
    """Small MLX-compatible array surface used to test the algorithm on Linux CI."""

    float32 = np.float32
    array = staticmethod(np.asarray)
    abs = staticmethod(np.abs)
    argmin = staticmethod(np.argmin)
    concatenate = staticmethod(np.concatenate)
    expand_dims = staticmethod(np.expand_dims)
    maximum = staticmethod(np.maximum)
    sqrt = staticmethod(np.sqrt)
    sum = staticmethod(np.sum)
    take = staticmethod(np.take)
    where = staticmethod(np.where)


@pytest.mark.parametrize("method", ["turboquant-mse", "turboquant-prod"])
def test_mlx_turboquant_quantizes_finite_weights_deterministically(method: str) -> None:
    rng = np.random.default_rng(7)
    weight = rng.standard_normal((7, 16), dtype=np.float32)
    first = MLXTurboQuantizer(bits=3, method=method, group_size=3, seed=11, _array_module=_NumpyMLX)
    second = MLXTurboQuantizer(bits=3, method=method, group_size=3, seed=11, _array_module=_NumpyMLX)

    quantized = first.quantize_weight(weight)
    repeated = second.quantize_weight(weight)

    assert quantized.shape == weight.shape
    assert quantized.dtype == weight.dtype
    assert np.isfinite(quantized).all()
    assert np.array_equal(quantized, repeated)
    assert not np.array_equal(quantized, weight)
    assert first.info()["implementation"] == "native-turboquant"


def test_mlx_turboquant_signed_permutation_handles_modern_projection_width() -> None:
    weight = np.linspace(-1.0, 1.0, 2 * 513, dtype=np.float32).reshape(2, 513)
    quantizer = MLXTurboQuantizer(bits=4, group_size=1, _array_module=_NumpyMLX)

    quantized = quantizer.quantize_weight(weight)

    assert quantized.shape == weight.shape
    assert np.isfinite(quantized).all()
    assert quantizer._state(513)["rotation"] is None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"bits": 0}, "bits must be"),
        ({"method": "unknown"}, "method must be"),
        ({"method": "turboquant-prod", "bits": 1}, "requires at least 2 bits"),
        ({"group_size": 0}, "group_size must be"),
    ],
)
def test_mlx_turboquant_rejects_invalid_configuration(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        MLXTurboQuantizer(_array_module=_NumpyMLX, **kwargs)
