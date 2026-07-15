"""TurboQuant KV-cache wrapper for PyTorch transformers."""

from __future__ import annotations

import torch

from forge.turboquant.quantizer import TurboQuantizer

try:
    from transformers import DynamicCache
except Exception:
    raise RuntimeError(
        "TurboQuant KV-cache requires `transformers` but it is not installed or broken. "
        "Run `uv sync` to ensure dependencies are installed."
    )


class TurboQuantKVCache(DynamicCache):
    """Drop-in cache that stores compressed keys and values."""

    def __init__(self, bits: int = 3, mode: str = "mse", seed: int = 42):
        super().__init__()
        self.quantizer = TurboQuantizer(bits=bits, mode=mode, seed=seed)
        self._compressed_keys: list[torch.Tensor | None] = []
        self._compressed_values: list[torch.Tensor | None] = []

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        while len(self._compressed_keys) <= layer_idx:
            self._compressed_keys.append(None)
            self._compressed_values.append(None)

        full_keys = key_states
        full_values = value_states
        cached_keys = self._compressed_keys[layer_idx]
        cached_values = self._compressed_values[layer_idx]
        if cached_keys is not None and cached_values is not None:
            full_keys = torch.cat([cached_keys, key_states], dim=2)
            full_values = torch.cat([cached_values, value_states], dim=2)

        compressed_keys = self.quantizer.quantize_dequantize(full_keys)
        compressed_values = self.quantizer.quantize_dequantize(full_values)
        self._compressed_keys[layer_idx] = compressed_keys
        self._compressed_values[layer_idx] = compressed_values
        return compressed_keys, compressed_values
