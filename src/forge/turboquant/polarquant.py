"""PolarQuant implementation for tensors and weight matrices."""

from __future__ import annotations

import math

import torch


class PolarQuantizer:
    """Polar-coordinate quantizer with an optional random rotation."""

    def __init__(self, bits: int = 3, use_rotation: bool = True, seed: int = 42):
        if bits < 1:
            raise ValueError("bits must be >= 1")
        self.bits = bits
        self.use_rotation = use_rotation
        self.seed = seed
        self._rotations: dict[tuple[int, str], torch.Tensor | None] = {}

    def _rotation(self, dim: int, device: torch.device) -> torch.Tensor | None:
        key = (dim, str(device))
        if key in self._rotations:
            return self._rotations[key]
        if not self.use_rotation:
            self._rotations[key] = None
            return None
        generator = torch.Generator(device="cpu").manual_seed(self.seed + dim)
        gaussian = torch.randn(dim, dim, generator=generator, dtype=torch.float32)
        rotation, r = torch.linalg.qr(gaussian)
        rotation = rotation * torch.sign(torch.diag(r)).unsqueeze(0)
        self._rotations[key] = rotation.to(device=device)
        return self._rotations[key]

    def quantize_dequantize(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize and dequantize a tensor with shape (..., dim)."""
        original_shape = x.shape
        original_dtype = x.dtype
        x_float = x.float().reshape(-1, original_shape[-1])
        original_dim = x_float.shape[-1]
        padded = False

        if original_dim % 2 != 0:
            x_float = torch.cat([x_float, torch.zeros(x_float.shape[0], 1, device=x_float.device)], dim=-1)
            padded = True

        rotation = self._rotation(x_float.shape[-1], x_float.device)
        if rotation is not None:
            x_float = x_float @ rotation.T

        pairs = x_float.reshape(x_float.shape[0], -1, 2)
        radius = torch.sqrt((pairs**2).sum(dim=-1) + 1e-10)
        theta = torch.atan2(pairs[..., 1], pairs[..., 0])
        theta = torch.where(theta < 0, theta + 2 * math.pi, theta)

        levels = max((1 << self.bits) - 1, 1)
        r_min = radius.min(dim=-1, keepdim=True).values
        r_max = radius.max(dim=-1, keepdim=True).values
        r_scale = (r_max - r_min).clamp(min=1e-8) / levels

        t_min = theta.min(dim=-1, keepdim=True).values
        t_max = theta.max(dim=-1, keepdim=True).values
        t_scale = (t_max - t_min).clamp(min=1e-8) / levels

        r_q = torch.clamp(((radius - r_min) / r_scale).round(), 0, levels)
        t_q = torch.clamp(((theta - t_min) / t_scale).round(), 0, levels)

        radius_hat = r_min + r_q * r_scale
        theta_hat = t_min + t_q * t_scale
        reconstructed = torch.stack(
            [radius_hat * torch.cos(theta_hat), radius_hat * torch.sin(theta_hat)],
            dim=-1,
        ).reshape(x_float.shape[0], -1)

        if rotation is not None:
            reconstructed = reconstructed @ rotation
        if padded:
            reconstructed = reconstructed[:, :original_dim]

        return reconstructed.reshape(original_shape).to(dtype=original_dtype)

    def quantize_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """Quantize a weight matrix row-wise."""
        return self.quantize_dequantize(weight)
