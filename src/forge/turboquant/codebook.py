"""TurboQuant codebook utilities."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
from scipy.stats import norm  # type: ignore[import-untyped]


def _gaussian_lloyd_max(
    n_levels: int,
    sigma: float,
    max_iter: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the Lloyd-Max quantizer for a zero-mean Gaussian."""
    quantiles = np.linspace(0.0, 1.0, n_levels + 1)
    quantiles[0] = 1e-6
    quantiles[-1] = 1 - 1e-6
    boundaries = norm.ppf(quantiles, scale=sigma)
    boundaries[0] = -np.inf
    boundaries[-1] = np.inf
    centroids = np.zeros(n_levels, dtype=np.float64)

    for _ in range(max_iter):
        for idx in range(n_levels):
            lo = boundaries[idx]
            hi = boundaries[idx + 1]
            alpha = lo / sigma if np.isfinite(lo) else -30.0
            beta = hi / sigma if np.isfinite(hi) else 30.0
            phi_a = norm.pdf(alpha)
            phi_b = norm.pdf(beta)
            cdf_a = norm.cdf(alpha)
            cdf_b = norm.cdf(beta)
            denom = cdf_b - cdf_a
            if denom < 1e-12:
                centroids[idx] = 0.0 if not (np.isfinite(lo) and np.isfinite(hi)) else (lo + hi) / 2.0
            else:
                centroids[idx] = sigma * (phi_a - phi_b) / denom

        for idx in range(1, n_levels):
            boundaries[idx] = (centroids[idx - 1] + centroids[idx]) / 2.0

    return boundaries.astype(np.float32), centroids.astype(np.float32)


@lru_cache(maxsize=64)
def build_codebook(bits: int, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a TurboQuant codebook for the given bit-width and dimension."""
    levels = 1 << bits
    sigma = 1.0 / np.sqrt(max(dim, 1))
    boundaries, centroids = _gaussian_lloyd_max(levels, sigma)
    return torch.from_numpy(boundaries), torch.from_numpy(centroids)


def quantize_scalar(
    x: torch.Tensor,
    boundaries: torch.Tensor,
    centroids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize coordinates using precomputed boundaries and centroids."""
    interior = boundaries[1:-1]
    indices = torch.bucketize(x, interior)
    return indices, centroids[indices]
