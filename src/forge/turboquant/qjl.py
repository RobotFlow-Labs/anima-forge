"""Torch QJL utilities used by TurboQuant product mode."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class QJLState:
    projection: torch.Tensor


def make_projection(dim: int, seed: int, device: torch.device) -> QJLState:
    """Create a deterministic Gaussian projection matrix."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    projection = torch.randn(dim, dim, generator=generator, dtype=torch.float32)
    return QJLState(projection=projection.to(device=device))


def quantize_residual(
    residual: torch.Tensor,
    projection: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project a residual and keep only its sign bits plus the residual norm."""
    projected = residual @ projection.T
    signs = torch.sign(projected)
    signs[signs == 0] = 1.0
    norms = residual.norm(dim=-1)
    return signs, norms


def reconstruct_residual(
    signs: torch.Tensor,
    norms: torch.Tensor,
    projection: torch.Tensor,
) -> torch.Tensor:
    """Approximate the residual from sign bits using the TurboQuant/QJL scaling."""
    dim = projection.shape[0]
    scale = (torch.pi / 2) ** 0.5 / max(dim, 1)
    return scale * norms.unsqueeze(-1) * (signs @ projection)
