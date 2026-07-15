"""Public quantization API and method registry."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import torch.nn as nn
import torch.nn.functional as F

from forge.config import ForgeConfig, QuantConfig
from forge.quantize.qvla import quantize_qvla

QUANT_METHODS = {
    "qvla": "qvla",
    "turboquant-mse": "turboquant-mse",
    "turboquant-prod": "turboquant-prod",
    "polarquant": "polarquant",
}


def _describe_turboquant_path() -> str:
    expected_root = Path(__file__).resolve().parents[1] / "turboquant"
    return f"Forge stores TurboQuant in the in-tree module `forge/turboquant/`. Resolved path check: {expected_root}"


def _raise_turboquant_error(message: str, exc: Exception) -> None:
    raise RuntimeError(
        f"{message} {_describe_turboquant_path()}. "
        "Please run inside the active `uv` environment from this repository "
        "by reinstalling FORGE on Linux, then invoke via `uv run forge ...`."
    ) from exc


def _get_quantizer_cls(method: str) -> Any:
    if method not in QUANT_METHODS:
        raise ValueError(f"Unknown quantization method: {method}")

    try:
        turboquant = importlib.import_module("forge.turboquant")
    except ModuleNotFoundError as exc:
        _raise_turboquant_error(
            "Missing `forge.turboquant` module import while resolving quantizer backend.",
            exc,
        )
    except Exception as exc:  # pragma: no cover - exercised at call sites
        _raise_turboquant_error(
            "TurboQuant backend import failed during initialization. This is usually a dependency path "
            "or install mismatch.",
            exc,
        )

    if method == "polarquant":
        cls = getattr(turboquant, "PolarQuantizer", None)
        if cls is None:
            raise RuntimeError(
                "TurboQuant is installed but PolarQuantizer was not found. "
                "Confirm the in-tree module `forge.turboquant` is available and importable."
            )
        return cls

    cls = getattr(turboquant, "TurboQuantizer", None)
    if cls is None:
        raise RuntimeError(
            "TurboQuant module is importable, but `TurboQuantizer` symbol is missing from `forge.turboquant`."
        )
    return cls


def quantize_model(
    model: nn.Module,
    bit_allocation: dict[str, dict[int, int]] | None = None,
    uniform_bits: int | None = None,
    method: str = "qvla",
    bits: int | None = None,
    seed: int = 42,
    *,
    inplace: bool = False,
    row_chunk_size: int = 64,
) -> nn.Module:
    """Apply the selected quantization method to a model."""
    if method not in QUANT_METHODS:
        raise ValueError(f"Unknown quantization method: {method}")

    if method == "qvla":
        return quantize_qvla(
            model,
            bit_allocation=bit_allocation,
            uniform_bits=uniform_bits,
            inplace=inplace,
            row_chunk_size=row_chunk_size,
        )

    quant_bits = bits if bits is not None else uniform_bits
    if quant_bits is None:
        quant_bits = 4

    if method == "polarquant":
        quantizer = _get_quantizer_cls(method)(bits=quant_bits, seed=seed)
    else:
        mode = "prod" if method == "turboquant-prod" else "mse"
        quantizer = _get_quantizer_cls(method)(bits=quant_bits, mode=mode, seed=seed)

    if row_chunk_size < 1:
        raise ValueError("row_chunk_size must be >= 1")
    if inplace:
        quantized_model = model
    else:
        import copy

        quantized_model = copy.deepcopy(model)
    for module in quantized_model.modules():
        if isinstance(module, nn.Linear):
            weight = module.weight.data
            if weight.ndim < 2:
                weight.copy_(quantizer.quantize_weight(weight))
                continue
            for start in range(0, weight.shape[0], row_chunk_size):
                rows = weight[start : start + row_chunk_size]
                rows.copy_(quantizer.quantize_weight(rows))
    return quantized_model


def quantize_model_with_config(
    model: nn.Module,
    config: ForgeConfig | QuantConfig,
    bit_allocation: dict[str, dict[int, int]] | None = None,
    *,
    inplace: bool = False,
    row_chunk_size: int = 64,
) -> nn.Module:
    """Apply the quantizer selected in config."""
    quant_cfg = config.quant if isinstance(config, ForgeConfig) else config
    return quantize_model(
        model,
        bit_allocation=bit_allocation,
        uniform_bits=quant_cfg.bits,
        method=quant_cfg.method,
        bits=quant_cfg.bits,
        seed=quant_cfg.seed,
        inplace=inplace,
        row_chunk_size=row_chunk_size,
    )


def benchmark_quantization(
    model: nn.Module,
    method: str,
    bits: int,
    seed: int = 42,
    max_layers: int | None = None,
) -> dict[str, float | int | str]:
    """Measure basic distortion metrics for a quantization method."""
    linear_layers = [module for module in model.modules() if isinstance(module, nn.Linear)]
    if max_layers is not None:
        linear_layers = linear_layers[:max_layers]
    if not linear_layers:
        return {"method": method, "bits": bits, "layers": 0, "mse": 0.0, "mae": 0.0, "cosine": 1.0}

    total_mse = 0.0
    total_mae = 0.0
    total_cosine = 0.0

    for layer in linear_layers:
        original = layer.weight.data.float()
        quantized_layer = quantize_model(layer, uniform_bits=bits, method=method, bits=bits, seed=seed)
        if not isinstance(quantized_layer, nn.Linear):
            raise TypeError("Quantizing a linear layer must return a linear layer")
        quantized = quantized_layer.weight.data.float()
        total_mse += F.mse_loss(quantized, original).item()
        total_mae += (quantized - original).abs().mean().item()
        total_cosine += F.cosine_similarity(quantized.reshape(1, -1), original.reshape(1, -1), dim=-1).item()

    n_layers = len(linear_layers)
    return {
        "method": method,
        "bits": bits,
        "layers": n_layers,
        "mse": total_mse / n_layers,
        "mae": total_mae / n_layers,
        "cosine": total_cosine / n_layers,
    }
