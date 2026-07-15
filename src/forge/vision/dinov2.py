"""DINOv2 vision encoder."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from forge.vision.registry import VisionEncoderRegistry, VisionEncoderSpec

logger = logging.getLogger(__name__)


class MockDINOv2Encoder(nn.Module):
    """Mock DINOv2 encoder."""

    def __init__(self, d_output: int = 384, n_tokens: int = 729):
        super().__init__()
        self.d_output = d_output
        self.n_tokens = n_tokens
        self.proj = nn.Linear(3 * 14 * 14, d_output)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        B = images.shape[0]
        patches = images.unfold(2, 14, 14).unfold(3, 14, 14)
        n_patches = patches.shape[2] * patches.shape[3]
        patches = patches.contiguous().view(B, 3, n_patches, 14, 14)
        patches = patches.permute(0, 2, 1, 3, 4).contiguous().view(B, n_patches, -1)
        if n_patches < self.n_tokens:
            padding = torch.zeros(B, self.n_tokens - n_patches, patches.shape[-1], device=images.device)
            patches = torch.cat([patches, padding], dim=1)
        else:
            patches = patches[:, : self.n_tokens]
        return self.proj(patches)


def create_dinov2(model_dir: str | None = None, *, allow_mock: bool = False) -> nn.Module:
    """Create a test-only mock until a real DINOv2 runtime is configured."""
    if allow_mock:
        return MockDINOv2Encoder()
    raise RuntimeError("DINOv2 has no configured production weights; pass allow_mock=True only in tests")


VisionEncoderRegistry.register(
    VisionEncoderSpec(
        name="dinov2-small",
        d_output=384,
        n_tokens=729,
        param_count_m=22,
        input_size=518,
        factory=create_dinov2,
    )
)
