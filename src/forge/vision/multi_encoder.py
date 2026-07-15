"""Multi-Encoder Fusion — combine features from multiple vision encoders."""

from __future__ import annotations

import torch
import torch.nn as nn

from forge.vision.registry import VisionEncoderRegistry


class MultiEncoderFusion(nn.Module):
    """Fuse features from multiple vision encoders.

    Strategy:
    1. Run each encoder on the input image
    2. Project each encoder's output to a common dimension
    3. Concatenate along token dimension
    4. Attention-pool to fixed number of output tokens
    """

    def __init__(
        self,
        encoder_names: list[str],
        d_output: int = 1152,
        n_output_tokens: int = 729,
        model_dir: str | None = None,
        allow_mock: bool = False,
    ):
        super().__init__()
        self.d_output = d_output
        self.n_output_tokens = n_output_tokens

        # Load encoders and create projectors
        self.encoders = nn.ModuleList()
        self.projectors = nn.ModuleList()

        for name in encoder_names:
            # Ensure encoder module is imported to register
            _ensure_registered(name)
            spec = VisionEncoderRegistry.get_spec(name)
            encoder = VisionEncoderRegistry.create(name, model_dir=model_dir, allow_mock=allow_mock)
            self.encoders.append(encoder)
            self.projectors.append(nn.Linear(spec.d_output, d_output, bias=False))

        # Attention pooling to fixed token count
        self.pool_queries = nn.Parameter(torch.randn(1, n_output_tokens, d_output) * 0.02)
        self.pool_attn = nn.MultiheadAttention(d_output, num_heads=8, batch_first=True)
        self.pool_norm = nn.LayerNorm(d_output)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (B, n_output_tokens, d_output)"""
        B = images.shape[0]
        all_features = []

        for encoder, projector in zip(self.encoders, self.projectors):
            with torch.no_grad():
                features = encoder(images)
            if hasattr(features, "last_hidden_state"):
                features = features.last_hidden_state
            projected = projector(features)
            all_features.append(projected)

        # Concatenate along token dim
        concat = torch.cat(all_features, dim=1)  # (B, sum(N_i), d_output)

        # Attention pool to fixed size
        queries = self.pool_queries.expand(B, -1, -1)
        pooled, _ = self.pool_attn(queries, concat, concat)
        return self.pool_norm(pooled + queries)


def _ensure_registered(name: str) -> None:
    """Import encoder modules to trigger registration."""
    if name not in VisionEncoderRegistry._encoders:
        import forge.vision.dinov2  # noqa: F401
        import forge.vision.siglip  # noqa: F401
        import forge.vision.theia  # noqa: F401
