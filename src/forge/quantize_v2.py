"""Compatibility shim for chunk-aware quantization."""

from forge.quantize.chunk_aware import (
    calibrate_chunk_ranges,
    measure_chunk_quantization_quality,
    quantize_chunk_aware,
)

__all__ = [
    "calibrate_chunk_ranges",
    "measure_chunk_quantization_quality",
    "quantize_chunk_aware",
]
