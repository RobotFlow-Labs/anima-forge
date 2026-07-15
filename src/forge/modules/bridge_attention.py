"""Bridge Attention module — compresses vision tokens for the language backbone.

Reduces 729 SigLIP vision tokens to 64 learned query tokens via cross-attention.
Based on VLA-Adapter (ICRA 2025) pattern.

Input: (B, 729, 1152) from SigLIP-SO400M
Output: (B, 64, d_model) ready for language backbone
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BridgeAttention(nn.Module):
    """Lightweight cross-attention that compresses vision features.

    Uses learned query vectors to attend over vision tokens,
    producing a fixed number of compressed visual representations.
    """

    def __init__(
        self,
        d_vision: int = 1152,
        d_model: int = 896,
        n_queries: int = 64,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_vision = d_vision
        self.d_model = d_model
        self.n_queries = n_queries
        self.n_heads = n_heads
        self.n_layers = n_layers

        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.head_dim = d_model // n_heads

        # Learned query vectors
        self.queries = nn.Parameter(torch.randn(1, n_queries, d_model) * 0.02)

        # Project vision features to model dimension
        self.vision_proj = nn.Linear(d_vision, d_model, bias=False)

        # Cross-attention layers
        self.layers = nn.ModuleList([BridgeCrossAttentionLayer(d_model, n_heads, dropout) for _ in range(n_layers)])

        # Final layer norm
        self.norm = nn.LayerNorm(d_model)

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """Compress vision features.

        Args:
            vision_features: (B, N_vis, d_vision) from SigLIP

        Returns:
            (B, n_queries, d_model) compressed visual tokens
        """
        B = vision_features.shape[0]

        # Project vision to model dimension
        kv = self.vision_proj(vision_features)  # (B, N_vis, d_model)

        # Expand queries for batch
        queries = self.queries.expand(B, -1, -1)  # (B, n_queries, d_model)

        # Cross-attention layers
        for layer in self.layers:
            queries = layer(queries, kv)

        return self.norm(queries)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class BridgeCrossAttentionLayer(nn.Module):
    """Single cross-attention layer with FFN."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = math.sqrt(self.head_dim)

        # Cross-attention projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """Cross-attend from queries to vision key-values.

        Args:
            queries: (B, n_queries, d_model)
            kv: (B, N_vis, d_model)

        Returns:
            (B, n_queries, d_model)
        """
        B, N_q, D = queries.shape
        _, N_kv, _ = kv.shape

        # Pre-norm cross-attention
        q_norm = self.norm1(queries)

        Q = self.q_proj(q_norm).view(B, N_q, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(kv).view(B, N_kv, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(kv).view(B, N_kv, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, N_q, D)
        out = self.o_proj(out)

        # Residual + FFN
        queries = queries + out
        queries = queries + self.ffn(self.norm2(queries))

        return queries
