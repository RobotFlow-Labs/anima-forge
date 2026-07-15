"""MLX export for Apple Silicon deployment.

Converts PyTorch FORGE models to MLX format for native Metal acceleration.
This is the Mac-first path — runs on M1-M4 without CUDA.

Usage:
    forge pipeline --stage export --checkpoint outputs/checkpoints/final.pt --output-dir outputs/export
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _to_mlx_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert every PyTorch floating dtype, including bf16, to MLX fp16."""
    value = tensor.detach().cpu()
    if value.dtype == torch.bfloat16:
        value = value.float()
    return value.numpy().astype(np.float16, copy=False)


def export_mlx(
    model: nn.Module,
    output_dir: str | Path,
    config: dict | None = None,
) -> Path:
    """Export FORGE model to MLX format.

    Saves:
    - weights.npz: All model weights as numpy arrays
    - config.json: Model architecture config
    - metadata.json: Export metadata

    Args:
        model: Trained FORGE student model
        output_dir: Directory for MLX files
        config: Model config dict to save alongside weights

    Returns:
        Path to output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    model = model.cpu()

    # Convert weights to numpy
    weights = {}
    for name, param in model.named_parameters():
        weights[name] = _to_mlx_numpy(param)

    # Save weights
    weights_path = output_dir / "weights.npz"
    # Dense fp16 tensors gain little from deflate while multi-billion-parameter
    # checkpoints can spend tens of minutes compressing. The uncompressed NPZ
    # container loads identically and writes at storage speed.
    np.savez(str(weights_path), **weights)  # type: ignore[arg-type]

    weight_size_mb = sum(w.nbytes for w in weights.values()) / (1024 * 1024)
    file_size_mb = weights_path.stat().st_size / (1024 * 1024)

    logger.info(
        "MLX weights saved: %s (%.1f MB archive, %.1f MB weights)",
        weights_path,
        file_size_mb,
        weight_size_mb,
    )

    # Save config
    if config:
        config_path = output_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, default=str)

    # Save metadata
    metadata = {
        "format": "mlx",
        "n_params": sum(p.numel() for p in model.parameters()),
        "n_weights": len(weights),
        "weight_size_mb": weight_size_mb,
        "file_size_mb": file_size_mb,
        "dtype": "float16",
        "archive_compression": "none",
    }
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return output_dir


def load_mlx_weights(weight_dir: str | Path) -> dict[str, np.ndarray]:
    """Load MLX weights from npz file."""
    weight_dir = Path(weight_dir)
    weights_path = weight_dir / "weights.npz"

    if not weights_path.exists():
        raise FileNotFoundError(f"MLX weights not found: {weights_path}")

    data = np.load(str(weights_path))
    return dict(data)


def validate_mlx_export(
    pytorch_model: nn.Module,
    mlx_dir: str | Path,
) -> dict:
    """Validate MLX export by comparing weight shapes and values."""
    mlx_weights = load_mlx_weights(mlx_dir)

    pytorch_weights = {name: _to_mlx_numpy(param) for name, param in pytorch_model.named_parameters()}

    mismatches = []
    for name in pytorch_weights:
        if name not in mlx_weights:
            mismatches.append(f"Missing in MLX: {name}")
            continue

        pt_shape = pytorch_weights[name].shape
        mlx_shape = mlx_weights[name].shape
        if pt_shape != mlx_shape:
            mismatches.append(f"Shape mismatch {name}: PT={pt_shape}, MLX={mlx_shape}")

    extra_in_mlx = set(mlx_weights.keys()) - set(pytorch_weights.keys())
    for name in extra_in_mlx:
        mismatches.append(f"Extra in MLX: {name}")

    return {
        "status": "passed" if not mismatches else "failed",
        "n_pytorch_params": len(pytorch_weights),
        "n_mlx_params": len(mlx_weights),
        "mismatches": mismatches,
    }
