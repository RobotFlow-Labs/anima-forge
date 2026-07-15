"""PRD-03: Knowledge Distillation Loss Functions.

Composite KD loss for VLA model distillation:
L_total = α_kd · L_KD + α_task · L_task + α_feat · L_feature + α_action · L_action

Key insight: Standard LLM distillation fails for VLA because small action errors
compound into catastrophic trajectory failures. We use action-distribution alignment
weighted by teacher confidence.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ForgeDistillationLoss(nn.Module):
    """Composite knowledge distillation loss for VLA models.

    Components:
    1. KD loss: KL divergence on soft labels (temperature-scaled)
    2. Task loss: MSE on ground truth actions
    3. Feature loss: Cosine alignment of vision features
    4. Action distribution loss: Confidence-weighted action matching
    """

    def __init__(
        self,
        temperature: float = 4.0,
        alpha_kd: float = 0.4,
        alpha_task: float = 0.3,
        alpha_feat: float = 0.2,
        alpha_action: float = 0.1,
        feature_proj_dim: tuple[int, int] | None = None,
    ):
        super().__init__()
        self.temperature = temperature
        self.alpha_kd = alpha_kd
        self.alpha_task = alpha_task
        self.alpha_feat = alpha_feat
        self.alpha_action = alpha_action

        # Feature projector (teacher dim → student dim) if dimensions differ
        self.feature_projector = None
        if feature_proj_dim is not None:
            self.feature_projector = nn.Linear(feature_proj_dim[0], feature_proj_dim[1], bias=False)

    def forward(
        self,
        student_actions: torch.Tensor,
        teacher_action_logits: torch.Tensor,
        ground_truth_actions: torch.Tensor,
        student_vision_features: torch.Tensor | None = None,
        teacher_vision_features: torch.Tensor | None = None,
        teacher_action_mean: torch.Tensor | None = None,
        teacher_action_std: torch.Tensor | None = None,
        teacher_confidence: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute composite distillation loss.

        Args:
            student_actions: (B, D_action) student predicted actions
            teacher_action_logits: (B, D_action) teacher soft labels
            ground_truth_actions: (B, D_action) ground truth actions
            student_vision_features: (B, N, D_student) student vision features
            teacher_vision_features: (B, N, D_teacher) teacher vision features
            teacher_action_mean: (B, D_action) teacher action distribution mean
            teacher_action_std: (B, D_action) teacher action distribution std
            teacher_confidence: (B, D_action) teacher confidence per action dim

        Returns:
            dict with 'total_loss' and individual components
        """
        losses = {}

        # 1. KD Loss — soft label matching via MSE (continuous actions, not discrete tokens)
        loss_kd = kd_loss(student_actions, teacher_action_logits, self.temperature)
        losses["kd"] = loss_kd

        # 2. Task Loss — ground truth supervision
        loss_task = task_loss(student_actions, ground_truth_actions)
        losses["task"] = loss_task

        # 3. Feature Loss — vision feature alignment
        loss_feat = torch.tensor(0.0, device=student_actions.device)
        if student_vision_features is not None and teacher_vision_features is not None:
            loss_feat = feature_alignment_loss(
                student_vision_features,
                teacher_vision_features,
                self.feature_projector,
            )
        losses["feature"] = loss_feat

        # 4. Action Distribution Loss — confidence-weighted
        loss_action = torch.tensor(0.0, device=student_actions.device)
        if teacher_action_mean is not None and teacher_action_std is not None:
            loss_action = action_distribution_loss(
                student_actions,
                teacher_action_mean,
                teacher_action_std,
                teacher_confidence,
            )
        losses["action"] = loss_action

        # Composite loss
        total = (
            self.alpha_kd * loss_kd
            + self.alpha_task * loss_task
            + self.alpha_feat * loss_feat
            + self.alpha_action * loss_action
        )
        losses["total"] = total

        return losses


def kd_loss(
    student_actions: torch.Tensor,
    teacher_actions: torch.Tensor,
    temperature: float = 4.0,
) -> torch.Tensor:
    """Knowledge distillation loss for continuous action space.

    For VLA models, actions are continuous (not discrete tokens),
    so we use temperature-scaled MSE instead of KL divergence.
    Higher temperature smooths the loss landscape.
    """
    # Scale by temperature for softer matching
    scaled_student = student_actions / temperature
    scaled_teacher = teacher_actions / temperature
    return F.mse_loss(scaled_student, scaled_teacher) * (temperature**2)


def task_loss(
    predicted_actions: torch.Tensor,
    ground_truth_actions: torch.Tensor,
) -> torch.Tensor:
    """Direct supervision from demonstration actions."""
    return F.mse_loss(predicted_actions, ground_truth_actions)


def feature_alignment_loss(
    student_features: torch.Tensor,
    teacher_features: torch.Tensor,
    projector: nn.Module | None = None,
) -> torch.Tensor:
    """Align student vision features with teacher's.

    Uses cosine similarity to match feature directions
    (scale-invariant, works across different magnitudes).
    """
    if projector is not None:
        teacher_features = projector(teacher_features)

    # Flatten spatial dimensions: (B, N, D) → (B, N*D)
    s_flat = student_features.flatten(1)
    t_flat = teacher_features.flatten(1)

    # Cosine embedding loss (target=1 means same direction)
    targets = torch.ones(s_flat.shape[0], device=s_flat.device)
    return F.cosine_embedding_loss(s_flat, t_flat, targets)


def action_distribution_loss(
    student_actions: torch.Tensor,
    teacher_mean: torch.Tensor,
    teacher_std: torch.Tensor,
    teacher_confidence: torch.Tensor | None = None,
) -> torch.Tensor:
    """Match action distributions, weighted by teacher confidence.

    Teacher-confident dimensions get higher weight in the loss.
    """
    # Mean matching
    mean_diff = student_actions - teacher_mean

    if teacher_confidence is not None:
        # Weight by confidence: high confidence → high weight
        mean_loss = (mean_diff**2 * teacher_confidence).mean()
    else:
        mean_loss = (mean_diff**2).mean()

    return mean_loss


def chunk_aware_kd_loss(
    student_actions: torch.Tensor,
    teacher_actions: torch.Tensor,
    temperature: float = 4.0,
    decay_factor: float = 0.95,
) -> torch.Tensor:
    """KD loss that handles action chunks with temporal weighting.

    Args:
        student_actions: (B, H, D_action) or (B, D_action)
        teacher_actions: (B, H, D_action) or (B, D_action)
        temperature: Temperature scaling
        decay_factor: Exponential decay per horizon step

    Returns:
        Scalar loss
    """
    # Handle 2D inputs (single step)
    if student_actions.dim() == 2:
        student_actions = student_actions.unsqueeze(1)
    if teacher_actions.dim() == 2:
        teacher_actions = teacher_actions.unsqueeze(1)

    H = student_actions.shape[1]

    # Temperature-scaled MSE
    scaled_student = student_actions / temperature
    scaled_teacher = teacher_actions / temperature
    per_step_loss = F.mse_loss(scaled_student, scaled_teacher, reduction="none").mean(dim=-1)  # (B, H)

    # Exponential decay weights
    weights = torch.tensor(
        [decay_factor**i for i in range(H)],
        device=student_actions.device,
        dtype=student_actions.dtype,
    )
    weights = weights / weights.sum()

    weighted = (per_step_loss * weights.unsqueeze(0)).sum(dim=1)  # (B,)
    return weighted.mean() * (temperature**2)
