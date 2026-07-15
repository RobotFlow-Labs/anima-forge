"""Native MLX implementation of the TurboQuant vector quantizer."""

from __future__ import annotations

import math
from importlib import import_module
from typing import Any

import numpy as np

from forge.turboquant.codebook import _gaussian_lloyd_max


class MLXTurboQuantizer:
    """Quantize MLX tensors with the same rotation/codebook design as TurboQuant.

    State construction happens on the CPU once per feature dimension. Tensor
    normalization, rotation, centroid assignment, and reconstruction stay in MLX.
    Weight matrices are processed in bounded row groups so the centroid-distance
    tensor cannot consume all unified memory for modern language-model layers.
    """

    def __init__(
        self,
        bits: int = 3,
        method: str = "turboquant-mse",
        group_size: int = 128,
        seed: int = 42,
        *,
        _array_module: Any | None = None,
    ) -> None:
        if bits < 1:
            raise ValueError("bits must be >= 1")
        if method not in {"turboquant-mse", "turboquant-prod"}:
            raise ValueError("method must be 'turboquant-mse' or 'turboquant-prod'")
        if method == "turboquant-prod" and bits < 2:
            raise ValueError("TurboQuant product mode requires at least 2 bits")
        if group_size < 1:
            raise ValueError("group_size must be >= 1")

        if _array_module is None:
            try:
                mx = import_module("mlx.core")
            except ImportError as exc:  # pragma: no cover - exercised on non-Apple callers
                raise RuntimeError(
                    "MLX TurboQuant requires the mandatory Apple Silicon `mlx` runtime. "
                    "Install FORGE with install.sh or install.ps1 on a supported platform."
                ) from exc
            _array_module = mx

        self.bits = bits
        self.method = method
        self.group_size = group_size
        self.seed = seed
        self.mx = _array_module
        self._state_cache: dict[int, dict[str, Any]] = {}

    def info(self) -> dict[str, int | str]:
        """Return the concrete backend configuration."""
        return {
            "backend": "mlx",
            "bits": self.bits,
            "method": self.method,
            "group_size": self.group_size,
            "seed": self.seed,
            "implementation": "native-turboquant",
        }

    def _state(self, dim: int) -> dict[str, Any]:
        state = self._state_cache.get(dim)
        if state is not None:
            return state

        rng = np.random.default_rng(self.seed + dim)
        rotation = None
        permutation = None
        inverse_permutation = None
        signs = None
        if dim <= 512:
            gaussian = rng.standard_normal((dim, dim), dtype=np.float32)
            rotation_np, triangular = np.linalg.qr(gaussian)
            diagonal_signs = np.sign(np.diag(triangular))
            diagonal_signs[diagonal_signs == 0] = 1
            rotation = self.mx.array(rotation_np * diagonal_signs[None, :], dtype=self.mx.float32)
        else:
            permutation_np = rng.permutation(dim).astype(np.int32)
            inverse_np = np.argsort(permutation_np).astype(np.int32)
            signs_np = (rng.integers(0, 2, size=dim, dtype=np.int8) * 2 - 1).astype(np.float32)
            permutation = self.mx.array(permutation_np)
            inverse_permutation = self.mx.array(inverse_np)
            signs = self.mx.array(signs_np, dtype=self.mx.float32)

        codebook_bits = self.bits if self.method == "turboquant-mse" else self.bits - 1
        _boundaries, centroids_np = _gaussian_lloyd_max(1 << codebook_bits, 1.0 / math.sqrt(max(dim, 1)))
        centroids = self.mx.array(centroids_np, dtype=self.mx.float32)

        projection = None
        if self.method == "turboquant-prod" and dim <= 512:
            projection_rng = np.random.default_rng(self.seed + 1)
            projection = self.mx.array(
                projection_rng.standard_normal((dim, dim), dtype=np.float32),
                dtype=self.mx.float32,
            )

        state = {
            "rotation": rotation,
            "permutation": permutation,
            "inverse_permutation": inverse_permutation,
            "signs": signs,
            "centroids": centroids,
            "projection": projection,
        }
        self._state_cache[dim] = state
        return state

    def quantize_dequantize(self, tensor: Any) -> Any:
        """Quantize and reconstruct an MLX tensor along its final dimension."""
        if not getattr(tensor, "shape", None) or tensor.shape[-1] < 1:
            raise ValueError("tensor must have a non-empty final dimension")

        original_dtype = tensor.dtype
        values = tensor.astype(self.mx.float32)
        dim = int(values.shape[-1])
        state = self._state(dim)

        norms = self.mx.maximum(
            self.mx.sqrt(self.mx.sum(values * values, axis=-1, keepdims=True)),
            self.mx.array(1e-8, dtype=self.mx.float32),
        )
        normalized = values / norms
        rotation = state["rotation"]
        if rotation is not None:
            rotated = normalized @ rotation.T
        else:
            rotated = self.mx.take(normalized, state["permutation"], axis=-1) * state["signs"]

        centroids = state["centroids"]
        distances = self.mx.abs(self.mx.expand_dims(rotated, axis=-1) - centroids)
        indices = self.mx.argmin(distances, axis=-1)
        quantized_rotated = self.mx.take(centroids, indices, axis=0)

        if rotation is not None:
            reconstructed = quantized_rotated @ rotation
        else:
            signed = quantized_rotated * state["signs"]
            reconstructed = self.mx.take(signed, state["inverse_permutation"], axis=-1)
        reconstructed = reconstructed * norms

        if self.method == "turboquant-prod":
            residual = values - reconstructed
            residual_norms = self.mx.sqrt(self.mx.sum(residual * residual, axis=-1, keepdims=True))
            projection = state["projection"]
            if projection is not None:
                projected = residual @ projection.T
                residual_signs = self.mx.where(projected >= 0, 1.0, -1.0)
                scale = math.sqrt(math.pi / 2.0) / max(dim, 1)
                reconstructed = reconstructed + scale * residual_norms * (residual_signs @ projection)
            else:
                residual_signs = self.mx.where(residual >= 0, 1.0, -1.0)
                reconstructed = reconstructed + residual_signs * residual_norms / math.sqrt(dim)

        return reconstructed.astype(original_dtype)

    def quantize_weight(self, weight: Any) -> Any:
        """Quantize a weight matrix in bounded row groups."""
        if len(weight.shape) != 2:
            return self.quantize_dequantize(weight)
        chunks = [
            self.quantize_dequantize(weight[start : start + self.group_size])
            for start in range(0, int(weight.shape[0]), self.group_size)
        ]
        if not chunks:
            raise ValueError("weight must contain at least one row")
        return chunks[0] if len(chunks) == 1 else self.mx.concatenate(chunks, axis=0)
