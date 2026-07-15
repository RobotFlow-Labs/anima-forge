"""LoRA (Low-Rank Adaptation) wrapper for FORGE student models.

Adds trainable low-rank adapters to frozen language model weights.
Only LoRA parameters are trained during distillation.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation.

    Wraps an existing Linear layer with low-rank A and B matrices:
    output = original(x) + (x @ A @ B) * (alpha / rank)
    """

    def __init__(self, original: nn.Linear, rank: int = 32, alpha: int = 64):
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original.in_features
        out_features = original.out_features

        # Freeze original weights
        for param in original.parameters():
            param.requires_grad = False

        # LoRA matrices
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)

        # Initialize: A from normal, B from zeros (starts as identity)
        nn.init.kaiming_uniform_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_out = self.original(x)
        # Keep trainable adapters in fp32 even when the frozen v3 backbone is
        # bf16, then return in the backbone activation dtype.
        lora_input = x.to(dtype=self.lora_A.weight.dtype)
        lora_out = self.lora_B(self.lora_A(lora_input)) * self.scaling
        return original_out + lora_out.to(dtype=original_out.dtype)

    @property
    def trainable_params(self) -> int:
        return sum(p.numel() for p in [self.lora_A.weight, self.lora_B.weight])


def apply_lora(
    model: nn.Module,
    rank: int = 32,
    alpha: int = 64,
    target_modules: list[str] | None = None,
) -> nn.Module:
    """Apply LoRA to target modules in a model.

    Args:
        model: The model to modify
        rank: LoRA rank
        alpha: LoRA alpha (scaling factor)
        target_modules: List of module name patterns to apply LoRA to.
                       Defaults to attention projections.

    Returns:
        Modified model with LoRA layers
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            # Check if this module name matches any target pattern
            if any(target in name for target in target_modules):
                # Replace with LoRA-wrapped version
                parent_name, attr_name = _split_module_name(name)
                parent = _get_module(model, parent_name) if parent_name else model
                lora_layer = LoRALinear(module, rank=rank, alpha=alpha)
                setattr(parent, attr_name, lora_layer)

    return model


def get_lora_params(model: nn.Module) -> list[nn.Parameter]:
    """Get only LoRA parameters (for optimizer)."""
    params: list[nn.Parameter] = []
    for module in model.modules():
        if isinstance(module, LoRALinear):
            params.extend(module.lora_A.parameters())
            params.extend(module.lora_B.parameters())
    return params


def _split_module_name(name: str) -> tuple[str, str]:
    """Split 'a.b.c' into ('a.b', 'c')."""
    parts = name.rsplit(".", 1)
    if len(parts) == 1:
        return "", parts[0]
    return parts[0], parts[1]


def _get_module(model: nn.Module, name: str) -> nn.Module:
    """Get nested module by dot-separated name."""
    parts = name.split(".")
    module = model
    for part in parts:
        module = getattr(module, part)
    return module
