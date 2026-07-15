"""Chunk-aware quantization utilities."""

from __future__ import annotations

import copy
import logging
import math
from typing import TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from forge.prune_v2 import temporal_coherence_score
from forge.quantize.qvla import _fake_quantize_channel

logger = logging.getLogger(__name__)


class ChunkQuantizationQuality(TypedDict):
    """Measured differences between full-precision and quantized actions."""

    action_mse: float
    temporal_coherence_delta: float
    max_step_drift: float
    per_step_error: list[float]


def calibrate_chunk_ranges(
    model: nn.Module,
    calibration_data: list[torch.Tensor],
    action_horizon: int = 8,
) -> dict[str, tuple[float, float]]:
    """Calibrate quantization ranges using chunk predictions."""
    model.eval()
    activation_ranges: dict[str, list[tuple[float, float]]] = {}
    hooks = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):

            def make_hook(mod_name: str):
                def hook_fn(mod, inp, out):
                    if mod_name not in activation_ranges:
                        activation_ranges[mod_name] = []
                    activation_ranges[mod_name].append((out.detach().min().item(), out.detach().max().item()))

                return hook_fn

            hooks.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        for images in calibration_data:
            if images.dim() == 3:
                images = images.unsqueeze(0)
            try:
                model(images)
            except Exception:
                continue

    for hook in hooks:
        hook.remove()

    calibrated: dict[str, tuple[float, float]] = {}
    for name, ranges in activation_ranges.items():
        if ranges:
            calibrated[name] = (min(r[0] for r in ranges), max(r[1] for r in ranges))

    return calibrated


def quantize_chunk_aware(
    model: nn.Module,
    target_bits: float = 4.0,
    chunk_calibration: dict[str, tuple[float, float]] | None = None,
    action_head_bits: int = 8,
) -> nn.Module:
    """Quantize with chunk-aware bit allocation.

    Fractional target widths use round-half-up semantics and must remain in the
    supported 2-to-8-bit range.
    """
    if not math.isfinite(target_bits) or not 2.0 <= target_bits <= 8.0:
        raise ValueError("target_bits must be finite and between 2 and 8 inclusive")
    if isinstance(action_head_bits, bool) or not isinstance(action_head_bits, int):
        raise TypeError("action_head_bits must be an integer")
    if not 2 <= action_head_bits <= 8:
        raise ValueError("action_head_bits must be between 2 and 8 inclusive")
    model = copy.deepcopy(model)
    default_bits = math.floor(target_bits + 0.5)

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        if "action_head" in name:
            bits = action_head_bits
        elif "vision_encoder" in name or "lora" in name.lower():
            continue
        else:
            bits = default_bits

        for ch_idx in range(module.weight.shape[0]):
            module.weight.data[ch_idx] = _fake_quantize_channel(module.weight.data[ch_idx], bits=bits)

    return model


def measure_chunk_quantization_quality(
    fp_model: nn.Module,
    quantized_model: nn.Module,
    test_data: list[torch.Tensor],
    action_horizon: int = 8,
) -> ChunkQuantizationQuality:
    """Compare FP vs quantized model on temporal coherence."""
    fp_model.eval()
    quantized_model.eval()

    action_mses = []
    tc_deltas = []
    per_step_error_sums: list[float] = []
    per_step_error_counts: list[int] = []
    max_drifts = []

    with torch.no_grad():
        for images in test_data:
            if images.dim() == 3:
                images = images.unsqueeze(0)

            try:
                fp_out = fp_model(images)
                q_out = quantized_model(images)
            except Exception:
                continue

            fp_actions = fp_out["actions"]
            q_actions = q_out["actions"]
            if fp_actions.shape != q_actions.shape:
                raise ValueError(
                    "Full-precision and quantized action shapes must match: "
                    f"{tuple(fp_actions.shape)} != {tuple(q_actions.shape)}"
                )
            if fp_actions.ndim == 2:
                fp_actions = fp_actions.unsqueeze(1)
                q_actions = q_actions.unsqueeze(1)
            elif fp_actions.ndim != 3:
                raise ValueError(
                    "Action outputs must have shape (batch, dimension) or "
                    f"(batch, horizon, dimension), got {tuple(fp_actions.shape)}"
                )
            if fp_actions.shape[0] == 0 or fp_actions.shape[1] == 0 or fp_actions.shape[2] == 0:
                raise ValueError("Action outputs must be nonempty")
            if not torch.isfinite(fp_actions).all() or not torch.isfinite(q_actions).all():
                raise ValueError("Action outputs must contain only finite values")
            action_mses.append(F.mse_loss(fp_actions, q_actions).item())

            if fp_actions.shape[1] > 1:
                fp_tc = temporal_coherence_score(fp_actions)
                q_tc = temporal_coherence_score(q_actions)
                tc_deltas.append(q_tc - fp_tc)
            else:
                tc_deltas.append(0.0)
            step_errors = (fp_actions - q_actions).pow(2).mean(dim=-1)
            step_sums = step_errors.sum(dim=0).cpu().tolist()
            while len(per_step_error_sums) < len(step_sums):
                per_step_error_sums.append(0.0)
                per_step_error_counts.append(0)
            for index, value in enumerate(step_sums):
                per_step_error_sums[index] += float(value)
                per_step_error_counts[index] += int(step_errors.shape[0])
            max_drifts.append(step_errors.max().item())

    per_step_error = [
        total / count for total, count in zip(per_step_error_sums, per_step_error_counts, strict=True) if count
    ]

    result: ChunkQuantizationQuality = {
        "action_mse": sum(action_mses) / max(len(action_mses), 1),
        "temporal_coherence_delta": sum(tc_deltas) / max(len(tc_deltas), 1),
        "max_step_drift": max(max_drifts) if max_drifts else 0.0,
        "per_step_error": per_step_error,
    }
    return result
