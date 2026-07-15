"""Fail-closed helpers shared by real training and benchmark loops."""

from __future__ import annotations

from collections.abc import Iterable

import torch


def backward_with_finite_gradients(
    loss: torch.Tensor,
    parameters: Iterable[torch.nn.Parameter],
    *,
    max_norm: float = 1.0,
) -> float:
    """Backpropagate one scalar loss and reject non-finite loss or gradients."""
    if loss.numel() != 1:
        raise ValueError(f"Training loss must be scalar, received shape {tuple(loss.shape)}")
    if not bool(torch.isfinite(loss.detach()).item()):
        raise FloatingPointError(f"Training loss is non-finite: {loss.detach().item()!r}")

    trainable = [parameter for parameter in parameters if parameter.requires_grad]
    if not trainable:
        raise ValueError("Training step has no trainable parameters")

    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        trainable,
        max_norm=max_norm,
        error_if_nonfinite=True,
    )
    return float(gradient_norm.detach().cpu().item())
