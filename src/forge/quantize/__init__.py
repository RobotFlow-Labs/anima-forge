"""Modular quantization package for FORGE."""

from forge.quantize.api import (
    QUANT_METHODS,
    benchmark_quantization,
    quantize_model,
    quantize_model_with_config,
)
from forge.quantize.profile import QuantProfile, create_quant_profile
from forge.quantize.qvla import (
    _fake_quantize_channel,
    allocate_bits,
    compute_channel_sensitivity,
)

__all__ = [
    "QUANT_METHODS",
    "QuantProfile",
    "_fake_quantize_channel",
    "allocate_bits",
    "benchmark_quantization",
    "compute_channel_sensitivity",
    "create_quant_profile",
    "quantize_model",
    "quantize_model_with_config",
]
