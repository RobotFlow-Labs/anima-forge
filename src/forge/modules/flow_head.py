"""Flow Matching Action Head — 1-step inference via Conditional Flow Matching.

Replaces the 10-step DDPM diffusion head with a flow-based approach.
Key advantage: inference in K=1,2,4 ODE steps instead of 10+ diffusion steps.

Based on:
- Flow Matching (Lipman et al., 2023)
- Rectified Flow (Liu et al., 2023)
- pi0 architecture (Physical Intelligence, 2024)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowMatchingActionHead(nn.Module):
    """Flow-based action prediction head.

    Training: learns velocity field v_theta(x_t, t, cond) that maps noise -> actions
    Inference: Euler ODE integration in K steps (K=1 for fastest)

    Args:
        d_model: Conditioning feature dimension
        d_action: Action output dimension
        n_layers: Number of residual blocks in velocity network
        d_hidden: Hidden dimension
        inference_steps: Number of ODE steps at inference (1, 2, or 4)
        sigma_min: Minimum noise scale for numerical stability
    """

    def __init__(
        self,
        d_model: int = 896,
        d_action: int = 7,
        n_layers: int = 4,
        d_hidden: int = 256,
        inference_steps: int = 4,
        sigma_min: float = 1e-4,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_action = d_action
        self.inference_steps = inference_steps
        self.sigma_min = sigma_min
        self.d_hidden = d_hidden

        # Conditioning projection
        self.cond_proj = nn.Linear(d_model, d_hidden)

        # Time embedding (sinusoidal)
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(d_hidden),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
        )

        # Velocity prediction network
        self.velocity_net = nn.ModuleList(
            [FlowBlock(d_action=d_action, d_cond=d_hidden, d_hidden=d_hidden) for _ in range(n_layers)]
        )

        # Final projection
        self.final_proj = nn.Linear(d_hidden, d_action)

    def forward(
        self,
        action_features: torch.Tensor,
        gt_actions: torch.Tensor | None = None,
    ) -> dict:
        """Forward pass.

        Training: compute flow matching loss
        Inference: ODE integration from noise to actions
        """
        cond = self.cond_proj(action_features)  # (B, d_hidden)

        if gt_actions is not None:
            return self._training_step(cond, gt_actions)
        else:
            return {"actions": self._inference(cond)}

    def _training_step(self, cond: torch.Tensor, gt_actions: torch.Tensor) -> dict:
        """Flow Matching training step.

        1. Sample t ~ U(0,1)
        2. Create x_t = (1-t)*noise + t*data  (linear interpolation)
        3. Target velocity = data - noise
        4. Predict velocity, compute MSE loss
        """
        B = cond.shape[0]
        device = cond.device

        # Sample time uniformly
        t = torch.rand(B, device=device)

        # Sample noise (source distribution)
        noise = torch.randn_like(gt_actions)

        # Linear interpolation: x_t = (1-t)*noise + t*data
        t_expanded = t.unsqueeze(-1)  # (B, 1)
        x_t = (1 - t_expanded) * noise + t_expanded * gt_actions

        # Target velocity: v = data - noise (optimal transport path)
        target_velocity = gt_actions - noise

        # Predict velocity
        predicted_velocity = self._predict_velocity(x_t, cond, t)

        # MSE loss on velocity
        loss = F.mse_loss(predicted_velocity, target_velocity)

        return {
            "actions": gt_actions,
            "loss": loss,
            "predicted_velocity": predicted_velocity,
        }

    def _inference(self, cond: torch.Tensor) -> torch.Tensor:
        """Euler ODE integration from noise to actions.

        x_{k+1} = x_k + (1/K) * v_theta(x_k, t_k, cond)
        where t_k = k/K
        """
        B = cond.shape[0]
        device = cond.device
        K = self.inference_steps

        # Start from pure noise
        x = torch.randn(B, self.d_action, device=device)

        dt = 1.0 / K
        for k in range(K):
            t = torch.full((B,), k * dt, device=device)
            velocity = self._predict_velocity(x, cond, t)
            x = x + dt * velocity

        return x

    def _predict_velocity(
        self,
        x_t: torch.Tensor,
        cond: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Predict velocity field v_theta(x_t, t, cond)."""
        t_emb = self.time_embed(t)  # (B, d_hidden)
        h = cond + t_emb  # (B, d_hidden)

        for block in self.velocity_net:
            h = block(x_t, h)

        return self.final_proj(h)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class FlowBlock(nn.Module):
    """Residual block for velocity prediction."""

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
    """Sinusoidal position embedding for continuous time t in [0, 1]."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.d_model // 2
        frequency_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -frequency_scale)
        emb = (t.float() * 1000).unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.d_model % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb
