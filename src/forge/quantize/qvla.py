"""QVLA and low-level fake quantization utilities."""

from __future__ import annotations

import copy
import logging
from decimal import Decimal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional

from forge.config import QuantConfig

logger = logging.getLogger(__name__)


def compute_channel_sensitivity(
    model: nn.Module,
    calibration_data: list[dict],
    n_samples: int = 50,
) -> dict[str, torch.Tensor]:
    """For each weight channel, measure how quantization affects action output."""
    sensitivities = {}
    model.eval()

    baseline_actions = []
    with torch.no_grad():
        for batch in calibration_data[:n_samples]:
            images = batch["image"].unsqueeze(0) if batch["image"].dim() == 3 else batch["image"]
            try:
                out = model(images)
                baseline_actions.append(out["actions"].clone())
            except Exception:
                continue

    if not baseline_actions:
        logger.warning("No baseline actions collected for sensitivity analysis")
        return {}

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if module.weight.shape[0] < 8:
            continue

        n_channels = module.weight.shape[0]
        channel_sens = torch.zeros(n_channels)

        with torch.no_grad():
            for batch_idx, batch in enumerate(calibration_data[: min(n_samples, 20)]):
                images = batch["image"].unsqueeze(0) if batch["image"].dim() == 3 else batch["image"]

                for ch in range(0, n_channels, max(1, n_channels // 16)):
                    original = module.weight.data[ch].clone()
                    module.weight.data[ch] = _fake_quantize_channel(original, bits=4)

                    try:
                        out = model(images)
                        if batch_idx < len(baseline_actions):
                            delta = functional.mse_loss(out["actions"], baseline_actions[batch_idx])
                            channel_sens[ch] += delta.item()
                    except Exception:
                        pass

                    module.weight.data[ch] = original

        sensitivities[name] = channel_sens / max(1, min(n_samples, 20))

    return sensitivities


def allocate_bits(
    sensitivities: dict[str, torch.Tensor],
    config: QuantConfig,
) -> dict[str, dict[int, int]]:
    """Assign more bits to action-sensitive channels and fewer to insensitive ones."""
    all_sens: list[float] = []
    channel_map: list[tuple[str, int]] = []

    for name, sens in sensitivities.items():
        for ch_idx in range(len(sens)):
            all_sens.append(sens[ch_idx].item())
            channel_map.append((name, ch_idx))

    if not all_sens:
        return {}

    all_sens_array = np.array(all_sens)
    s_min, s_max = all_sens_array.min(), all_sens_array.max()
    s_range = s_max - s_min
    if s_range < 1e-8:
        s_norm = np.ones_like(all_sens_array) * 0.5
    else:
        s_norm = (all_sens_array - s_min) / s_range

    if config.min_bits < 1:
        raise ValueError("min_bits must be >= 1")
    if config.min_bits > config.max_bits:
        raise ValueError("min_bits must be <= max_bits")
    if not config.min_bits <= config.target_avg_bits <= config.max_bits:
        raise ValueError("target_avg_bits must be between min_bits and max_bits")

    bit_range = config.max_bits - config.min_bits
    raw_bits = config.min_bits + s_norm * bit_range
    target_total_decimal = Decimal(str(config.target_avg_bits)) * len(raw_bits)
    if target_total_decimal != target_total_decimal.to_integral_value():
        raise ValueError(
            f"target_avg_bits={config.target_avg_bits} is not exactly realizable "
            f"across {len(raw_bits)} integer-valued channels"
        )
    target_total = int(target_total_decimal)

    def realize(scale: float) -> np.ndarray:
        return np.rint(np.clip(raw_bits * scale, config.min_bits, config.max_bits)).astype(int)

    # Re-solve the scale against the rounded integer total. Clipping and
    # rounding make the objective stepwise, so retain the closest candidate and
    # then distribute any residual one bit at a time by sensitivity.
    low = 0.0
    high = 1.0
    while realize(high).sum() < target_total:
        high *= 2.0

    candidates: list[np.ndarray] = []
    for _ in range(64):
        midpoint = (low + high) / 2.0
        candidate = realize(midpoint)
        candidates.append(candidate)
        if candidate.sum() < target_total:
            low = midpoint
        else:
            high = midpoint

    candidates.extend((realize(low), realize(high)))
    scaled_bits = min(candidates, key=lambda candidate: abs(int(candidate.sum()) - target_total)).copy()

    residual = target_total - int(scaled_bits.sum())
    if residual > 0:
        order = np.argsort(-s_norm, kind="stable")
        while residual:
            changed = False
            for index in order:
                if scaled_bits[index] >= config.max_bits:
                    continue
                scaled_bits[index] += 1
                residual -= 1
                changed = True
                if residual == 0:
                    break
            if not changed:  # Defensive: validated bounds make this unreachable.
                raise RuntimeError("Unable to realize target bit allocation")
    elif residual < 0:
        order = np.argsort(s_norm, kind="stable")
        while residual:
            changed = False
            for index in order:
                if scaled_bits[index] <= config.min_bits:
                    continue
                scaled_bits[index] -= 1
                residual += 1
                changed = True
                if residual == 0:
                    break
            if not changed:  # Defensive: validated bounds make this unreachable.
                raise RuntimeError("Unable to realize target bit allocation")

    allocation: dict[str, dict[int, int]] = {}
    for (name, ch_idx), bits in zip(channel_map, scaled_bits):
        if name not in allocation:
            allocation[name] = {}
        allocation[name][ch_idx] = int(bits)

    return allocation


def quantize_qvla(
    model: nn.Module,
    bit_allocation: dict[str, dict[int, int]] | None = None,
    uniform_bits: int | None = None,
    *,
    inplace: bool = False,
    row_chunk_size: int = 64,
) -> nn.Module:
    """Apply QVLA or uniform fake quantization to a model.

    ``inplace`` is intended for full-model compression where retaining a second
    copy can exceed a 24 GB deployment GPU. Public library callers retain the
    historical copy-on-quantize behavior by default.
    """
    if row_chunk_size < 1:
        raise ValueError("row_chunk_size must be >= 1")
    model = model if inplace else copy.deepcopy(model)

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        if bit_allocation and name in bit_allocation:
            alloc = bit_allocation[name]
            for ch_idx, bits in alloc.items():
                if ch_idx < module.weight.shape[0]:
                    module.weight.data[ch_idx] = _fake_quantize_channel(module.weight.data[ch_idx], bits=bits)
        elif uniform_bits:
            _fake_quantize_rows_inplace(
                module.weight.data,
                bits=uniform_bits,
                row_chunk_size=row_chunk_size,
            )

    return model


def _fake_quantize_rows_inplace(
    weight: torch.Tensor,
    *,
    bits: int,
    row_chunk_size: int,
) -> None:
    """Quantize a weight matrix with bounded temporary CUDA memory."""
    if weight.ndim < 2:
        weight.copy_(_fake_quantize_channel(weight, bits))
        return
    for start in range(0, weight.shape[0], row_chunk_size):
        rows = weight[start : start + row_chunk_size]
        rows.copy_(_fake_quantize_rows(rows, bits=bits))


def _fake_quantize_channel(weight: torch.Tensor, bits: int) -> torch.Tensor:
    """Simulate quantization on a single weight channel."""
    qmin = -(2 ** (bits - 1))
    qmax = (2 ** (bits - 1)) - 1

    w_min = weight.min()
    w_max = weight.max()

    if w_max - w_min < 1e-8:
        return weight

    scale = (w_max - w_min) / (qmax - qmin)
    zero_point = qmin - (w_min / scale).round()
    quantized = torch.clamp((weight / scale + zero_point).round(), qmin, qmax)
    return (quantized - zero_point) * scale


def _fake_quantize_rows(weight: torch.Tensor, bits: int) -> torch.Tensor:
    """Vectorized per-output-channel fake quantization for a weight matrix."""
    if weight.ndim < 2:
        return _fake_quantize_channel(weight, bits)
    qmin = -(2 ** (bits - 1))
    qmax = (2 ** (bits - 1)) - 1
    work = weight.float()
    row_min = work.amin(dim=-1, keepdim=True)
    row_max = work.amax(dim=-1, keepdim=True)
    dynamic_range = row_max - row_min
    safe_scale = torch.where(
        dynamic_range < 1e-8,
        torch.ones_like(dynamic_range),
        dynamic_range / (qmax - qmin),
    )
    zero_point = qmin - (row_min / safe_scale).round()
    quantized = torch.clamp((work / safe_scale + zero_point).round(), qmin, qmax)
    restored = (quantized - zero_point) * safe_scale
    restored = torch.where(dynamic_range < 1e-8, work, restored)
    return restored.to(dtype=weight.dtype)
