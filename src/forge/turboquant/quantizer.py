"""TurboQuant weight/vector quantizers."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from forge.turboquant.codebook import build_codebook, quantize_scalar
from forge.turboquant.qjl import make_projection, quantize_residual, reconstruct_residual


@dataclass
class _TurboState:
    rotation: torch.Tensor | None
    boundaries: torch.Tensor
    centroids: torch.Tensor
    projection: torch.Tensor | None = None
    permutation: torch.Tensor | None = None
    signs: torch.Tensor | None = None


class TurboQuantizer:
    """TurboQuant implementation for weight matrices and general vectors."""

    def __init__(self, bits: int = 3, mode: str = "mse", seed: int = 42):
        if bits < 1:
            raise ValueError("bits must be >= 1")
        if mode not in {"mse", "prod"}:
            raise ValueError("mode must be 'mse' or 'prod'")
        if mode == "prod" and bits < 2:
            raise ValueError("TurboQuant product mode requires at least 2 bits")
        self.bits = bits
        self.mode = mode
        self.seed = seed
        self._state_cache: dict[tuple[int, str], _TurboState] = {}

    def _state(self, dim: int, device: torch.device) -> _TurboState:
        key = (dim, str(device))
        state = self._state_cache.get(key)
        if state is not None:
            return state

        generator = torch.Generator(device="cpu").manual_seed(self.seed + dim)
        rotation = None
        permutation = None
        signs = None
        # Dense Gaussian QR is cubic and becomes impractical for modern LLM
        # projections. A signed permutation is also orthogonal and O(n).
        if dim <= 512:
            gaussian = torch.randn(dim, dim, generator=generator, dtype=torch.float32)
            rotation, r = torch.linalg.qr(gaussian)
            rotation = rotation * torch.sign(torch.diag(r)).unsqueeze(0)
        else:
            permutation = torch.randperm(dim, generator=generator).to(device=device)
            signs = torch.randint(0, 2, (dim,), generator=generator, dtype=torch.float32)
            signs = (signs * 2 - 1).to(device=device)
        cb_bits = self.bits if self.mode == "mse" else self.bits - 1
        boundaries, centroids = build_codebook(cb_bits, dim)
        projection = None
        if self.mode == "prod" and dim <= 512:
            projection = make_projection(dim, self.seed + 1, device).projection
        state = _TurboState(
            rotation=rotation.to(device=device) if rotation is not None else None,
            boundaries=boundaries.to(device=device),
            centroids=centroids.to(device=device),
            projection=projection,
            permutation=permutation,
            signs=signs,
        )
        self._state_cache[key] = state
        return state

    def quantize_dequantize(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize and dequantize a tensor with shape (..., dim)."""
        original_dtype = x.dtype
        x_float = x.float()
        dim = x_float.shape[-1]
        state = self._state(dim, x_float.device)

        norms = x_float.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        normalized = x_float / norms
        if state.rotation is not None:
            rotated = normalized @ state.rotation.T
        else:
            assert state.permutation is not None and state.signs is not None
            permutation = state.permutation
            signs = state.signs
            rotated = normalized[..., permutation] * signs
        indices, _ = quantize_scalar(rotated, state.boundaries, state.centroids)
        quantized_rotated = state.centroids[indices]
        if state.rotation is not None:
            reconstructed = quantized_rotated @ state.rotation
        else:
            reconstructed = torch.empty_like(quantized_rotated)
            reconstructed[..., permutation] = quantized_rotated * signs
        reconstructed = reconstructed * norms

        if self.mode == "prod":
            residual = x_float - reconstructed
            if state.projection is not None:
                residual_signs, residual_norms = quantize_residual(residual, state.projection)
                reconstructed = reconstructed + reconstruct_residual(
                    residual_signs,
                    residual_norms,
                    state.projection,
                )
            else:
                residual_norms = residual.norm(dim=-1, keepdim=True)
                reconstructed = reconstructed + residual.sign() * residual_norms / (dim**0.5)

        return reconstructed.to(dtype=original_dtype)

    def quantize_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """Quantize a linear layer weight matrix row-wise."""
        return self.quantize_dequantize(weight)
