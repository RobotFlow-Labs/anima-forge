"""Quantization profile models and size estimation."""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn


@dataclass
class QuantProfile:
    """Quantization profile for a model."""

    name: str
    avg_bits: float
    bit_allocation: dict[str, dict[int, int]]
    total_params: int
    compressed_size_mb: float
    quantized_params: int = 0
    frozen_params: int = 0


def create_quant_profile(
    model: nn.Module,
    bit_allocation: dict[str, dict[int, int]],
    name: str = "q4_action",
    *,
    uniform_bits: int | None = None,
) -> QuantProfile:
    """Create a profile of the weights represented by the quantized payload.

    ``avg_bits`` and ``compressed_size_mb`` describe quantized ``Linear``
    weights only. Biases, embeddings, norms, and unallocated rows are reported
    separately through ``frozen_params`` instead of being mixed into the
    quantized average.

    An explicit ``uniform_bits`` applies to every otherwise-unallocated linear
    row. For backwards compatibility, an empty allocation with no explicit
    width describes the historical uniform 4-bit profile.
    """
    total_params = sum(p.numel() for p in model.parameters())
    quantized_params = 0
    quantized_bits = 0
    effective_uniform_bits = 4 if not bit_allocation and uniform_bits is None else uniform_bits

    if effective_uniform_bits is not None and effective_uniform_bits < 1:
        raise ValueError("uniform_bits must be >= 1")

    for mod_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        rows = module.weight.shape[0]
        elements_per_row = module.weight[0].numel() if rows else 0
        allocated_rows: set[int] = set()
        for ch_idx, bits in bit_allocation.get(mod_name, {}).items():
            if not 0 <= ch_idx < rows:
                continue
            if bits < 1:
                raise ValueError(f"bit allocation for {mod_name}[{ch_idx}] must be >= 1")
            allocated_rows.add(ch_idx)
            quantized_params += elements_per_row
            quantized_bits += elements_per_row * bits

        if effective_uniform_bits is not None:
            remaining_rows = rows - len(allocated_rows)
            quantized_params += remaining_rows * elements_per_row
            quantized_bits += remaining_rows * elements_per_row * effective_uniform_bits

    compressed_mb = quantized_bits / (8 * 1024 * 1024)
    return QuantProfile(
        name=name,
        avg_bits=quantized_bits / max(quantized_params, 1),
        bit_allocation=bit_allocation,
        total_params=total_params,
        compressed_size_mb=compressed_mb,
        quantized_params=quantized_params,
        frozen_params=total_params - quantized_params,
    )
