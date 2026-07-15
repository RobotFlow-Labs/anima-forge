"""Consistency Distillation Head — single-step inference from multi-step teacher.

Post-training compression for flow/diffusion action heads.
Trains a student head to match the teacher's multi-step output in 1 step.

Architecture mirrors FlowMatchingActionHead but trained differently:
- Teacher: uses K steps of ODE integration
- Student: trained to match teacher's K-step output in 1 step
- EMA: exponential moving average teacher for training stability

Based on:
- Consistency Models (Song et al., 2023)
- Consistency Distillation (Song & Dhariwal, 2023)
- FORGE v2 single-step inference design
"""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from forge.modules.flow_head import FlowBlock, SinusoidalTimeEmbedding


class ConsistencyActionHead(nn.Module):
    """Action head trained via consistency distillation.

    Same architecture as FlowMatchingActionHead, but inference always uses
    a single step. During training, it learns to match the multi-step
    teacher output in one forward pass.

    Args:
        d_model: Conditioning feature dimension
        d_action: Action output dimension
        n_layers: Number of residual blocks in velocity network
        d_hidden: Hidden dimension
    """

    def __init__(
        self,
        d_model: int = 896,
        d_action: int = 7,
        n_layers: int = 4,
        d_hidden: int = 256,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_action = d_action
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
        """Forward pass — always single-step inference.

        Args:
            action_features: (B, d_model) conditioning features
            gt_actions: ignored (consistency head doesn't train via this path)

        Returns:
            dict with 'actions' key
        """
        cond = self.cond_proj(action_features)
        return {"actions": self._single_step_inference(cond)}

    def _single_step_inference(self, cond: torch.Tensor) -> torch.Tensor:
        """Single-step denoising: noise → actions in one forward pass."""
        B = cond.shape[0]
        device = cond.device

        # Start from noise
        x = torch.randn(B, self.d_action, device=device)

        # Single step at t=0 (start of trajectory)
        t = torch.zeros(B, device=device)
        velocity = self._predict_velocity(x, cond, t)
        return x + velocity  # Full step: dt=1.0

    def _predict_velocity(
        self,
        x_t: torch.Tensor,
        cond: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Predict velocity field v_theta(x_t, t, cond)."""
        t_emb = self.time_embed(t)
        h = cond + t_emb

        for block in self.velocity_net:
            h = block(x_t, h)

        return self.final_proj(h)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class ConsistencyDistillationTrainer:
    """Trains a ConsistencyActionHead from a multi-step teacher.

    Algorithm:
    1. Initialize student from teacher weights
    2. EMA teacher = copy of student (updated via Polyak averaging)
    3. For each batch:
       a. Sample noise, compute x_0 (noise)
       b. Teacher: run K ODE steps from x_0 → target actions
       c. Student: run 1 step from x_0 → prediction
       d. Loss = ||prediction - target||^2
    4. EMA update: ema = mu * ema + (1-mu) * student
    5. Curriculum: increase K from 2 → 4 → 8 during training

    Args:
        teacher_head: Trained FlowMatchingActionHead (multi-step)
        student_head: ConsistencyActionHead to train (single-step)
        ema_decay: Polyak averaging decay rate
        curriculum_schedule: List of (step_threshold, K) pairs
    """

    def __init__(
        self,
        teacher_head: nn.Module,
        student_head: ConsistencyActionHead,
        ema_decay: float = 0.999,
        curriculum_schedule: list[tuple[int, int]] | None = None,
    ):
        self.teacher = teacher_head
        self.student = student_head
        self.ema_teacher = copy.deepcopy(student_head)
        self.ema_decay = ema_decay
        self.curriculum = curriculum_schedule or [
            (0, 2),  # Start with K=2
            (1000, 4),  # At step 1000, K=4
            (3000, 8),  # At step 3000, K=8
        ]

        # Freeze teacher
        for p in self.teacher.parameters():
            p.requires_grad = False
        for p in self.ema_teacher.parameters():
            p.requires_grad = False

    def training_step(
        self,
        cond: torch.Tensor,
        gt_actions: torch.Tensor,
        global_step: int,
    ) -> dict:
        """One training step of consistency distillation.

        Args:
            cond: (B, d_model) conditioning features (from student bridge)
            gt_actions: (B, d_action) ground truth actions (for teacher reference)
            global_step: Current training step (for curriculum)

        Returns:
            dict with 'loss' and 'K'
        """
        K = self._get_current_K(global_step)

        # Get EMA teacher's multi-step prediction as target
        with torch.no_grad():
            teacher_actions = self._teacher_multi_step(cond, K, use_ema=True)

        # Student's single-step prediction
        student_out = self.student(cond)
        student_actions = student_out["actions"]

        # Consistency loss: match teacher's K-step output
        loss = F.mse_loss(student_actions, teacher_actions)

        # EMA update after gradient step
        self._ema_update()

        return {"loss": loss, "K": K}

    def _teacher_multi_step(self, cond: torch.Tensor, K: int, use_ema: bool = False) -> torch.Tensor:
        """Run teacher ODE for K steps to get target actions.

        Uses the teacher's (or EMA teacher's) velocity network for Euler integration.
        """
        model: Any = self.ema_teacher if use_ema else self.teacher
        B = cond.shape[0]
        device = cond.device

        # Project conditioning through teacher
        if hasattr(model, "cond_proj"):
            teacher_cond = model.cond_proj(cond)
        else:
            teacher_cond = cond

        # Start from noise
        d_action_value: Any = model.d_action if hasattr(model, "d_action") else self.teacher.d_action
        if not isinstance(d_action_value, int):
            raise TypeError("Consistency teacher d_action must be an integer")
        d_action = d_action_value
        x = torch.randn(B, d_action, device=device)

        # K-step Euler integration
        dt = 1.0 / K
        for k in range(K):
            t = torch.full((B,), k * dt, device=device)
            if hasattr(model, "_predict_velocity"):
                velocity = model._predict_velocity(x, teacher_cond, t)
            else:
                # Fallback: treat teacher as producing actions directly
                velocity = torch.zeros_like(x)
            x = x + dt * velocity

        return x.detach()

    def _ema_update(self) -> None:
        """Polyak averaging: ema = mu * ema + (1-mu) * student."""
        with torch.no_grad():
            for ema_p, student_p in zip(
                self.ema_teacher.parameters(),
                self.student.parameters(),
            ):
                ema_p.data.mul_(self.ema_decay).add_(student_p.data, alpha=1 - self.ema_decay)

    def _get_current_K(self, step: int) -> int:
        """Get current teacher steps from curriculum schedule."""
        K = 2
        for step_thresh, new_K in self.curriculum:
            if step >= step_thresh:
                K = new_K
        return K
