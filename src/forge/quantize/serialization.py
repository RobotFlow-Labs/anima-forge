"""Compact, loadable serialization for quantized FORGE state dictionaries."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch

PACKED_STATE_KEY = "packed_state_dict"
PACKED_SCHEMA = "forge.packed-state.v1"

_DTYPES = {
    str(torch.float16): torch.float16,
    str(torch.bfloat16): torch.bfloat16,
    str(torch.float32): torch.float32,
    str(torch.float64): torch.float64,
}


def _pack_nibbles(values: torch.Tensor) -> torch.Tensor:
    flat = values.to(dtype=torch.uint8, device="cpu").flatten()
    if flat.numel() % 2:
        flat = torch.cat([flat, torch.zeros(1, dtype=torch.uint8)])
    return flat[0::2] | (flat[1::2] << 4)


def _unpack_nibbles(values: torch.Tensor, numel: int) -> torch.Tensor:
    packed = values.to(dtype=torch.uint8, device="cpu").flatten()
    unpacked = torch.empty(packed.numel() * 2, dtype=torch.uint8)
    unpacked[0::2] = packed & 0x0F
    unpacked[1::2] = packed >> 4
    return unpacked[:numel]


def pack_state_dict(
    state_dict: Mapping[str, Any],
    *,
    bits: int,
) -> tuple[dict[str, Any], dict[str, float | int | str]]:
    """Quantize and pack floating tensors while preserving loadable metadata."""
    if bits not in {4, 8}:
        raise ValueError(f"Packed FORGE checkpoints support 4 or 8 bits, got {bits}")

    packed_state: dict[str, Any] = {}
    original_bytes = 0
    packed_bytes = 0

    for name, value in state_dict.items():
        if not torch.is_tensor(value):
            packed_state[name] = {"kind": "value", "value": value}
            continue

        tensor = value.detach().cpu().contiguous()
        original_bytes += tensor.numel() * tensor.element_size()
        if not tensor.is_floating_point() or tensor.numel() == 0:
            packed_state[name] = {"kind": "tensor", "value": tensor}
            packed_bytes += tensor.numel() * tensor.element_size()
            continue

        qmin = -(2 ** (bits - 1))
        qmax = (2 ** (bits - 1)) - 1
        max_abs = tensor.float().abs().max()
        scale = max_abs / qmax if max_abs.item() > 0 else torch.tensor(1.0)
        codes = torch.clamp(torch.round(tensor.float() / scale), qmin, qmax).to(torch.int16)

        if bits == 4:
            payload = _pack_nibbles(codes + 8)
        else:
            payload = codes.to(torch.int8)

        packed_state[name] = {
            "kind": "quantized",
            "bits": bits,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "scale": scale.to(dtype=torch.float32, device="cpu"),
            "payload": payload,
        }
        packed_bytes += payload.numel() * payload.element_size() + 4

    metadata: dict[str, float | int | str] = {
        "schema": PACKED_SCHEMA,
        "bits": bits,
        "original_bytes": original_bytes,
        "packed_bytes": packed_bytes,
        "compression_ratio": original_bytes / max(packed_bytes, 1),
    }
    return packed_state, metadata


def unpack_state_dict(packed_state: Mapping[str, Any]) -> dict[str, Any]:
    """Restore a normal tensor state dictionary from a packed checkpoint."""
    state_dict: dict[str, Any] = {}
    for name, entry in packed_state.items():
        if not isinstance(entry, Mapping):
            raise ValueError(f"Packed state entry {name!r} is not a mapping")
        kind = entry.get("kind")
        if kind in {"tensor", "value"}:
            state_dict[name] = entry["value"]
            continue
        if kind != "quantized":
            raise ValueError(f"Packed state entry {name!r} has unknown kind {kind!r}")

        bits = int(entry["bits"])
        shape = tuple(int(dim) for dim in entry["shape"])
        numel = math.prod(shape)
        payload = entry["payload"]
        if bits == 4:
            codes = _unpack_nibbles(payload, numel).to(torch.int16) - 8
        elif bits == 8:
            codes = payload.to(dtype=torch.int8, device="cpu").to(torch.int16)
        else:
            raise ValueError(f"Packed state entry {name!r} uses unsupported {bits}-bit codes")

        dtype_name = str(entry["dtype"])
        if dtype_name not in _DTYPES:
            raise ValueError(f"Packed state entry {name!r} uses unsupported dtype {dtype_name}")
        scale = entry["scale"].to(dtype=torch.float32, device="cpu")
        state_dict[name] = (codes.float() * scale).reshape(shape).to(dtype=_DTYPES[dtype_name])
    return state_dict


__all__ = [
    "PACKED_SCHEMA",
    "PACKED_STATE_KEY",
    "pack_state_dict",
    "unpack_state_dict",
]
