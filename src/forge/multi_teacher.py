"""Multi-Teacher Distillation — learn from N teachers with routing.

Key components:
1. TeacherRouter: nn.Module that learns per-sample teacher weights
2. MultiTeacherDistiller: orchestrates N teacher adapters for label generation
3. MultiTeacherDistillationLoss: weighted combination of per-teacher KD losses

Based on:
- Multi-teacher KD (You et al., 2017)
- Routing Networks (Rosenbaum et al., 2018)
- FORGE v2 multi-path distillation design
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as functional

from forge.teachers.base import ActionChunk, TeacherAdapter


class TeacherRouter(nn.Module):
    """Learns per-sample routing weights across N teachers.

    Input: student features (B, D)
    Output: teacher weights (B, N_teachers) via temperature-scaled softmax

    Architecture: 2-layer MLP with GELU activation.
    """

    def __init__(self, d_input: int, n_teachers: int, temperature: float = 1.0):
        super().__init__()
        self.n_teachers = n_teachers
        self.temperature = temperature
        self.router = nn.Sequential(
            nn.Linear(d_input, d_input // 2),
            nn.GELU(),
            nn.Linear(d_input // 2, n_teachers),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """(B, D) → (B, N_teachers) routing weights summing to 1."""
        router_param = next(self.router.parameters())
        if features.device != router_param.device or features.dtype != router_param.dtype:
            features = features.to(device=router_param.device, dtype=router_param.dtype)
        logits = self.router(features) / self.temperature
        return functional.softmax(logits, dim=-1)


class MultiTeacherDistiller:
    """Orchestrates multi-teacher label generation.

    Creates teacher adapters from the registry and manages their lifecycle.
    Each teacher generates ActionChunk predictions that are then weighted
    by the TeacherRouter during training.

    Usage:
        distiller = MultiTeacherDistiller(["openvla-7b", "rdt2-fm"], model_dir)
        distiller.load_all(device="cuda")
        labels = distiller.generate_labels(image, instruction)
    """

    # Map adapter names to model directory names
    DIR_MAP = {
        "openvla-7b": "openvla--openvla-7b",
        "rdt2-fm": "robotics-diffusion-transformer--RDT2-FM",
        "smolvla-base": "lerobot--smolvla_base",
        "molmoact2-libero": "allenai--MolmoAct2-LIBERO-LeRobot",
        "vla-jepa-3b": "lerobot--VLA-JEPA-Pretrain",
    }

    def __init__(self, teacher_names: list[str], model_dir: str):
        from forge.teachers.registry import get_registry

        self.registry = get_registry()
        self.teacher_names = teacher_names
        self.teachers: dict[str, TeacherAdapter] = {name: self.registry.create(name) for name in teacher_names}
        self.model_dir = model_dir

    @property
    def n_teachers(self) -> int:
        return len(self.teachers)

    @property
    def loaded_teachers(self) -> list[str]:
        return [name for name, adapter in self.teachers.items() if adapter.is_loaded]

    def load_all(self, device: str = "cpu") -> None:
        """Load all teacher models from local paths."""
        for name, adapter in self.teachers.items():
            model_path = Path(self.model_dir) / self.DIR_MAP.get(name, name)
            if model_path.exists():
                adapter.load(model_path, device=device)

    def generate_labels(
        self,
        image: torch.Tensor,
        instruction: str,
    ) -> dict[str, ActionChunk]:
        """Generate labels from all loaded teachers.

        Args:
            image: (C, H, W) or (B, C, H, W) input image
            instruction: Language instruction string

        Returns:
            Dict mapping teacher name → ActionChunk prediction
        """
        image_array = image.detach().cpu()
        if image_array.ndim == 4:
            if image_array.shape[0] != 1:
                raise ValueError("Multi-teacher label generation accepts exactly one image at a time")
            image_array = image_array[0]
        if image_array.ndim != 3:
            raise ValueError(f"Expected CHW or HWC image tensor, got {tuple(image_array.shape)}")
        if image_array.shape[0] in {1, 3}:
            image_array = image_array.permute(1, 2, 0)
        if image_array.dtype.is_floating_point:
            image_array = image_array.clamp(0, 1).mul(255)
        image_numpy = image_array.to(torch.uint8).numpy()

        results: dict[str, ActionChunk] = {}
        for name, adapter in self.teachers.items():
            if adapter.is_loaded:
                results[name] = adapter.predict(image_numpy, instruction)
        return results

    def unload_all(self) -> None:
        """Free all teacher models from memory."""
        for adapter in self.teachers.values():
            adapter.unload()


class MultiTeacherDistillationLoss(nn.Module):
    """Loss that combines KD signals from multiple teachers with learned routing.

    For each sample:
    1. Router assigns weights w_i to each teacher based on student features
    2. Per-teacher KD loss is computed as MSE between student and teacher actions
    3. Total = (1 - α_task) * Σ w_i * L_KD_i + α_task * L_task

    The router is trained end-to-end, learning which teacher to trust per sample.
    """

    def __init__(
        self,
        n_teachers: int,
        d_student: int,
        temperature: float = 4.0,
        alpha_task: float = 0.3,
    ):
        super().__init__()
        self.router = TeacherRouter(d_student, n_teachers)
        self.temperature = temperature
        self.alpha_task = alpha_task

    def forward(
        self,
        student_actions: torch.Tensor,
        teacher_actions_list: list[torch.Tensor],
        ground_truth_actions: torch.Tensor,
        student_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute multi-teacher weighted loss.

        Args:
            student_actions: (B, D_action) student predictions
            teacher_actions_list: list of N (B, D_action) teacher predictions
            ground_truth_actions: (B, D_action) ground truth
            student_features: (B, D_student) features for routing

        Returns:
            dict with 'total', 'kd', 'task', 'router_weights'
        """
        weights = self.router(student_features)  # (B, N)

        # Per-teacher KD losses: MSE reduced over action dim, kept per-batch
        teacher_losses: list[torch.Tensor] = []
        for teacher_actions in teacher_actions_list:
            loss_i = functional.mse_loss(student_actions, teacher_actions, reduction="none").mean(dim=-1)  # (B,)
            teacher_losses.append(loss_i)

        stacked_teacher_losses = torch.stack(teacher_losses, dim=-1)  # (B, N)
        weighted_kd = (weights * stacked_teacher_losses).sum(dim=-1).mean()  # scalar

        # Task loss on ground truth
        task_loss = functional.mse_loss(student_actions, ground_truth_actions)

        total = (1 - self.alpha_task) * weighted_kd + self.alpha_task * task_loss

        return {
            "total": total,
            "kd": weighted_kd,
            "task": task_loss,
            "router_weights": weights.detach(),
        }
