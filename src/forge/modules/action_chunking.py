"""Action Chunking Head — predict H actions per forward pass.

Based on ACT (Action Chunking with Transformers) and OFT (One-step Flow Training).
Key insight: predicting action chunks amortizes perception cost over H steps,
giving up to 26x throughput improvement.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionChunkHead(nn.Module):
    """Predicts a chunk of H future actions from a single observation.

    Args:
        d_model: Input feature dimension (from language backbone)
        d_action: Per-step action dimension (e.g., 7 for 6DoF + gripper)
        horizon: Number of future steps to predict (H)
        n_layers: Number of transformer decoder layers
        n_heads: Number of attention heads
        d_hidden: Hidden dimension for the transformer
        chunk_overlap: Number of overlapping steps for blending
    """

    def __init__(
        self,
        d_model: int = 896,
        d_action: int = 7,
        horizon: int = 8,
        n_layers: int = 4,
        n_heads: int = 8,
        d_hidden: int = 256,
        chunk_overlap: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_action = d_action
        self.horizon = horizon
        self.d_hidden = d_hidden
        self.chunk_overlap = chunk_overlap

        # Project conditioning features
        self.cond_proj = nn.Linear(d_model, d_hidden)

        # Temporal position embeddings for each step in the horizon
        self.temporal_pos_embed = nn.Embedding(horizon, d_hidden)

        # Learnable action queries — one per horizon step
        self.action_queries = nn.Parameter(torch.randn(1, horizon, d_hidden) * 0.02)

        # Transformer decoder layers
        self.decoder_layers = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    d_model=d_hidden,
                    nhead=n_heads,
                    dim_feedforward=d_hidden * 4,
                    dropout=0.1,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(n_layers)
            ]
        )

        # Final projection to action space
        self.action_proj = nn.Linear(d_hidden, d_action)

        # Layer norm
        self.norm = nn.LayerNorm(d_hidden)

    def forward(
        self,
        action_features: torch.Tensor,
        gt_actions: torch.Tensor | None = None,
    ) -> dict:
        """Forward pass.

        Args:
            action_features: (B, d_model) from language backbone
            gt_actions: (B, H, d_action) ground truth action chunks (training)
                        OR (B, d_action) single step (v1 compat, expanded to H=1)

        Returns:
            dict with 'actions': (B, H, d_action) and optionally 'loss'
        """
        B = action_features.shape[0]
        device = action_features.device

        # Project conditioning
        cond = self.cond_proj(action_features)  # (B, d_hidden)

        # Prepare action queries with temporal embeddings
        positions = torch.arange(self.horizon, device=device)
        temporal_emb = self.temporal_pos_embed(positions)  # (H, d_hidden)
        queries = self.action_queries.expand(B, -1, -1) + temporal_emb.unsqueeze(0)  # (B, H, d_hidden)

        # Conditioning as memory for cross-attention
        memory = cond.unsqueeze(1)  # (B, 1, d_hidden)

        # Decode
        for layer in self.decoder_layers:
            queries = layer(queries, memory)

        queries = self.norm(queries)
        actions = self.action_proj(queries)  # (B, H, d_action)

        result = {"actions": actions}

        # Training loss
        if gt_actions is not None:
            # Handle v1 compat: (B, D_action) → (B, 1, D_action)
            if gt_actions.dim() == 2:
                gt_actions = gt_actions.unsqueeze(1).expand(-1, self.horizon, -1)

            loss = chunk_weighted_loss(actions, gt_actions, self.horizon)
            result["loss"] = loss

        return result

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def chunk_weighted_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    horizon: int,
    decay_factor: float = 0.95,
) -> torch.Tensor:
    """Chunk-aware loss with exponential decay weighting.

    Near-future actions get higher weight than far-future.

    Args:
        predicted: (B, H, D_action) predicted action chunk
        target: (B, H, D_action) target action chunk
        horizon: Number of steps in chunk
        decay_factor: Exponential decay per step

    Returns:
        Scalar loss
    """
    weights = torch.tensor(
        [decay_factor**i for i in range(horizon)],
        device=predicted.device,
        dtype=predicted.dtype,
    )
    weights = weights / weights.sum()  # Normalize

    # Per-step MSE
    per_step_loss = F.mse_loss(predicted, target, reduction="none")  # (B, H, D)
    per_step_loss = per_step_loss.mean(dim=-1)  # (B, H)

    # Weighted sum over horizon
    weighted_loss = (per_step_loss * weights.unsqueeze(0)).sum(dim=1)  # (B,)

    return weighted_loss.mean()


def blend_action_chunks(
    chunks: list[torch.Tensor],
    overlap: int = 2,
) -> torch.Tensor:
    """Blend overlapping action chunks for smooth trajectories.

    Args:
        chunks: List of (H, D_action) tensors
        overlap: Number of overlapping steps

    Returns:
        (T, D_action) blended trajectory
    """
    if len(chunks) == 0:
        raise ValueError("No chunks to blend")
    if len(chunks) == 1:
        return chunks[0]

    H = chunks[0].shape[0]
    D = chunks[0].shape[1]
    step = H - overlap

    # Total length
    T = H + step * (len(chunks) - 1)
    result = torch.zeros(T, D, device=chunks[0].device, dtype=chunks[0].dtype)
    weights = torch.zeros(T, 1, device=chunks[0].device, dtype=chunks[0].dtype)

    for i, chunk in enumerate(chunks):
        start = i * step
        end = start + H

        # Linear blending weights
        w = torch.ones(H, 1, device=chunk.device, dtype=chunk.dtype)
        if i > 0 and overlap > 0:
            w[:overlap] = torch.linspace(0, 1, overlap, device=chunk.device).unsqueeze(1)
        if i < len(chunks) - 1 and overlap > 0:
            w[-overlap:] = torch.linspace(1, 0, overlap, device=chunk.device).unsqueeze(1)

        result[start:end] += chunk * w
        weights[start:end] += w

    result = result / weights.clamp(min=1e-8)

    return result
