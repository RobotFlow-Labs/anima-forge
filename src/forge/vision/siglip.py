"""Canonical SigLIP2-SO400M vision encoder."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

from forge.vision.registry import VisionEncoderRegistry, VisionEncoderSpec

logger = logging.getLogger(__name__)


class MockSigLIPEncoder(nn.Module):
    """Mock SigLIP encoder for testing without weights."""

    def __init__(self, d_output: int = 1152, n_tokens: int = 729):
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


def create_siglip(model_dir: str | None = None, *, allow_mock: bool = False) -> nn.Module:
    """Create the canonical local SigLIP2 encoder without a silent fallback."""
    if model_dir:
        local_path = Path(model_dir) / "google--siglip2-so400m-patch14-384"
        if local_path.exists():
            try:
                from transformers import SiglipVisionModel

                encoder = SiglipVisionModel.from_pretrained(str(local_path), local_files_only=True)
                for p in encoder.parameters():
                    p.requires_grad = False
                encoder.eval()
                logger.info("SigLIP2 loaded from %s", local_path)
                return encoder
            except (AttributeError, OSError):
                # Full SiglipConfig doesn't have hidden_size — load full model and extract vision tower
                try:
                    from transformers import AutoModel

                    full_model = AutoModel.from_pretrained(str(local_path), local_files_only=True)
                    encoder = full_model.vision_model
                    del full_model
                    for p in encoder.parameters():
                        p.requires_grad = False
                    encoder.eval()
                    logger.info("SigLIP2 vision tower extracted from %s", local_path)
                    return encoder
                except Exception as e2:
                    if not allow_mock:
                        raise RuntimeError(f"Failed to load canonical SigLIP2 from {local_path}: {e2}") from e2
            except Exception as e:
                if not allow_mock:
                    raise RuntimeError(f"Failed to load canonical SigLIP2 from {local_path}: {e}") from e

    if allow_mock:
        logger.warning("Using explicitly requested mock SigLIP2 encoder")
        return MockSigLIPEncoder()
    expected = Path(model_dir or "models") / "google--siglip2-so400m-patch14-384"
    raise FileNotFoundError(f"Canonical SigLIP2 weights not found at {expected}")


# Register
VisionEncoderRegistry.register(
    VisionEncoderSpec(
        name="siglip2-so400m",
        d_output=1152,
        n_tokens=729,
        param_count_m=400,
        input_size=384,
        factory=create_siglip,
    )
)

# Backward-compatible config spelling; it resolves to the same v3 SigLIP2 weights.
VisionEncoderRegistry.register(
    VisionEncoderSpec(
        name="siglip-so400m",
        d_output=1152,
        n_tokens=729,
        param_count_m=400,
        input_size=384,
        factory=create_siglip,
    )
)
