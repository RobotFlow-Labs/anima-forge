"""Fail-closed training-step regression tests."""

from __future__ import annotations

import pytest
import torch

from forge.training_safety import backward_with_finite_gradients


def test_backward_with_finite_gradients_updates_finite_parameters() -> None:
    parameter = torch.nn.Parameter(torch.tensor([2.0]))
    loss = parameter.square().sum()

    norm = backward_with_finite_gradients(loss, [parameter])

    assert norm > 0
    assert parameter.grad is not None
    assert torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_backward_with_finite_gradients_rejects_non_finite_loss(value: float) -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    loss = parameter.sum() * torch.tensor(value)

    with pytest.raises(FloatingPointError, match="non-finite"):
        backward_with_finite_gradients(loss, [parameter])

    assert parameter.grad is None


def test_backward_with_finite_gradients_rejects_non_finite_gradient() -> None:
    parameter = torch.nn.Parameter(torch.tensor([0.0]))
    loss = torch.sqrt(parameter).sum()

    with pytest.raises(RuntimeError, match="non-finite"):
        backward_with_finite_gradients(loss, [parameter])


def test_backward_with_finite_gradients_requires_scalar_loss() -> None:
    parameter = torch.nn.Parameter(torch.ones(2))
    with pytest.raises(ValueError, match="scalar"):
        backward_with_finite_gradients(parameter.square(), [parameter])
