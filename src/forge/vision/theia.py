"""Theia vision encoder — multi-resolution features."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from forge.vision.registry import VisionEncoderRegistry, VisionEncoderSpec

logger = logging.getLogger(__name__)


class MockTheiaEncoder(nn.Module):
    """Mock Theia encoder."""

    def __init__(self, d_output: int = 384, n_tokens: int = 576):
        super().__init__()
        self.d_output = d_output
        self.n_tokens = n_tokens
        self.proj = nn.Linear(3 * 16 * 16, d_output)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        B = images.shape[0]
        patches = images.unfold(2, 16, 16).unfold(3, 16, 16)
        n_patches = patches.shape[2] * patches.shape[3]
        patches = patches.contiguous().view(B, 3, n_patches, 16, 16)
        patches = patches.permute(0, 2, 1, 3, 4).contiguous().view(B, n_patches, -1)
        if n_patches < self.n_tokens:
            padding = torch.zeros(B, self.n_tokens - n_patches, patches.shape[-1], device=images.device)
            patches = torch.cat([patches, padding], dim=1)
        else:
            patches = patches[:, : self.n_tokens]
        return self.proj(patches)


def create_theia(model_dir: str | None = None, *, allow_mock: bool = False) -> nn.Module:
    """Create a test-only mock until a real Theia runtime is configured."""
    if allow_mock:
        return MockTheiaEncoder()
    raise RuntimeError("Theia has no configured production weights; pass allow_mock=True only in tests")


VisionEncoderRegistry.register(
    VisionEncoderSpec(
        name="theia-tiny",
        d_output=384,
        n_tokens=576,
        param_count_m=5,
        input_size=384,
        factory=create_theia,
    )
)
