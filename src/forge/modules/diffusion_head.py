"""Diffusion Transformer Action Head.

Generates robot actions using a denoising diffusion process.
Following pi0 and RDT2 patterns — diffusion heads outperform
direct regression for multi-modal action distributions.

Input: (B, d_model) action features from language backbone
Output: (B, d_action) predicted actions (e.g., 7-dim: 6DoF + gripper)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionActionHead(nn.Module):
    """Diffusion-based action prediction head.

    Uses a small transformer to denoise action predictions.
    During training: learns to predict noise added to ground truth actions.
    During inference: iteratively denoises random noise into actions.
    """

    def __init__(
        self,
        d_model: int = 896,
        d_action: int = 7,
        n_layers: int = 4,
        n_diffusion_steps: int = 10,
        d_hidden: int = 256,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_action = d_action
        self.n_diffusion_steps = n_diffusion_steps
        self.d_hidden = d_hidden

        # Project conditioning features
        self.cond_proj = nn.Linear(d_model, d_hidden)

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(d_hidden),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
        )

        # Noise prediction network (small MLP with conditioning)
        self.noise_pred = nn.ModuleList()
        for i in range(n_layers):
            self.noise_pred.append(
                DiffusionBlock(
                    d_action=d_action,
                    d_cond=d_hidden,
                    d_hidden=d_hidden,
                )
            )

        # Final projection
        self.final_proj = nn.Linear(d_hidden, d_action)

        # Noise schedule (linear beta schedule)
        betas = torch.linspace(1e-4, 0.02, n_diffusion_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas: torch.Tensor
        self.alphas: torch.Tensor
        self.alphas_cumprod: torch.Tensor
        self.sqrt_alphas_cumprod: torch.Tensor
        self.sqrt_one_minus_alphas_cumprod: torch.Tensor
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def forward(
        self,
        action_features: torch.Tensor,
        gt_actions: torch.Tensor | None = None,
    ) -> dict:
        """Forward pass.

        During training (gt_actions provided): return noise prediction loss.
        During inference (gt_actions=None): denoise and return predicted actions.

        Args:
            action_features: (B, d_model) from language backbone
            gt_actions: (B, d_action) ground truth actions (training only)

        Returns:
            dict with 'actions' and optionally 'loss'
        """
        cond = self.cond_proj(action_features)  # (B, d_hidden)

        if gt_actions is not None:
            # Training: predict noise
            return self._training_step(cond, gt_actions)
        else:
            # Inference: denoise
            return {"actions": self._inference(cond)}

    def _training_step(self, cond: torch.Tensor, gt_actions: torch.Tensor) -> dict:
        """Diffusion training: add noise, predict it."""
        B = cond.shape[0]
        device = cond.device

        # Sample random timesteps
        t = torch.randint(0, self.n_diffusion_steps, (B,), device=device)

        # Sample noise
        noise = torch.randn_like(gt_actions)

        # Add noise to ground truth actions
        sqrt_alpha = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        noisy_actions = sqrt_alpha * gt_actions + sqrt_one_minus_alpha * noise

        # Predict noise
        predicted_noise = self._predict_noise(noisy_actions, cond, t)

        # MSE loss on noise prediction
        loss = F.mse_loss(predicted_noise, noise)

        return {
            "actions": gt_actions,  # Pass through for metrics
            "loss": loss,
            "predicted_noise": predicted_noise,
        }

    def _inference(self, cond: torch.Tensor) -> torch.Tensor:
        """DDPM inference: iteratively denoise from random noise."""
        B = cond.shape[0]
        device = cond.device

        # Start from pure noise
        x = torch.randn(B, self.d_action, device=device)

        for t_idx in reversed(range(self.n_diffusion_steps)):
            t = torch.full((B,), t_idx, device=device, dtype=torch.long)
            predicted_noise = self._predict_noise(x, cond, t)

            # DDPM step
            alpha = self.alphas[t_idx]
            alpha_cumprod = self.alphas_cumprod[t_idx]
            beta = self.betas[t_idx]

            x = (1 / torch.sqrt(alpha)) * (x - (beta / torch.sqrt(1 - alpha_cumprod)) * predicted_noise)

            # Add noise (except at last step)
            if t_idx > 0:
                noise = torch.randn_like(x)
                x = x + torch.sqrt(beta) * noise

        return x

    def _predict_noise(
        self,
        noisy_actions: torch.Tensor,
        cond: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Predict noise using the denoising network.

        Args:
            noisy_actions: (B, d_action) noisy action input
            cond: (B, d_hidden) conditioning features
            t: (B,) timestep indices

        Returns:
            (B, d_action) predicted noise
        """
        # Time embedding
        t_emb = self.time_embed(t)  # (B, d_hidden)

        # Combined conditioning
        h = cond + t_emb  # (B, d_hidden)

        # Pass through denoising blocks
        for block in self.noise_pred:
            h = block(noisy_actions, h)

        return self.final_proj(h)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class DiffusionBlock(nn.Module):
    """Single block of the noise prediction network."""

    def __init__(self, d_action: int, d_cond: int, d_hidden: int):
        super().__init__()
        self.action_proj = nn.Linear(d_action, d_hidden)
        self.cond_proj = nn.Linear(d_cond, d_hidden)
        self.net = nn.Sequential(
            nn.LayerNorm(d_hidden),
            nn.Linear(d_hidden, d_hidden * 2),
            nn.GELU(),
            nn.Linear(d_hidden * 2, d_hidden),
        )

    def forward(self, actions: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.action_proj(actions) + self.cond_proj(cond)
        return cond + self.net(h)


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal position embedding for diffusion timesteps."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.d_model // 2
        frequency_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -frequency_scale)
        emb = t.float().unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.d_model % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb
