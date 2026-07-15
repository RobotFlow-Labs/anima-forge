"""Backend abstraction — seamless CUDA / MLX / CPU switching.

This is the core abstraction that makes FORGE dual-platform.
All code imports from here instead of directly from torch/mlx.

Usage:
    from forge.backend import get_backend, BackendType

    backend = get_backend()  # Auto-detects: CUDA > MLX > CPU
    tensor = backend.zeros(3, 4)
    tensor = backend.to_device(tensor)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


class BackendType(Enum):
    CUDA = "cuda"
    MLX = "mlx"
    CPU = "cpu"


@dataclass
class DeviceInfo:
    backend: BackendType
    device_name: str
    vram_gb: float
    compute_capability: str | None = None


def detect_backend() -> BackendType:
    """Auto-detect best available backend. Respects FORGE_DEVICE env var."""
    forced = os.environ.get("FORGE_DEVICE", "").lower()
    if forced == "cuda":
        return BackendType.CUDA
    if forced == "mlx":
        return BackendType.MLX
    if forced == "cpu":
        return BackendType.CPU

    # Auto-detect
    try:
        import torch

        if torch.cuda.is_available():
            return BackendType.CUDA
    except ImportError:
        pass

    try:
        import mlx.core  # type: ignore[import-not-found]  # noqa: F401

        return BackendType.MLX
    except ImportError:
        pass

    return BackendType.CPU


class TorchBackend:
    """PyTorch backend for CUDA and CPU."""

    def __init__(self, device: str = "cpu"):
        import torch

        self.torch = torch
        self.device = torch.device(device)
        self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32

    @property
    def name(self) -> str:
        return f"torch:{self.device}"

    def zeros(self, *shape: int) -> Any:
        return self.torch.zeros(*shape, dtype=self.dtype, device=self.device)

    def from_numpy(self, arr: np.ndarray) -> Any:
        return self.torch.from_numpy(arr).to(device=self.device, dtype=self.dtype)

    def to_numpy(self, tensor: Any) -> np.ndarray:
        return tensor.detach().cpu().float().numpy()

    def to_device(self, tensor: Any) -> Any:
        return tensor.to(device=self.device, dtype=self.dtype)

    def get_device_info(self) -> DeviceInfo:
        if self.device.type == "cuda":
            props = self.torch.cuda.get_device_properties(self.device)
            return DeviceInfo(
                backend=BackendType.CUDA,
                device_name=props.name,
                vram_gb=props.total_memory / (1024**3),
                compute_capability=f"{props.major}.{props.minor}",
            )
        return DeviceInfo(backend=BackendType.CPU, device_name="CPU", vram_gb=0)

    def save(self, obj: Any, path: str) -> None:
        self.torch.save(obj, path)

    def load(self, path: str) -> Any:
        return self.torch.load(path, map_location=self.device, weights_only=True)


class MLXBackend:
    """Apple Silicon MLX backend."""

    def __init__(self):
        import mlx.core as mx

        self.mx = mx
        self.dtype = mx.float16

    @property
    def name(self) -> str:
        return "mlx"

    def zeros(self, *shape: int) -> Any:
        return self.mx.zeros(shape, dtype=self.dtype)

    def from_numpy(self, arr: np.ndarray) -> Any:
        return self.mx.array(arr, dtype=self.dtype)

    def to_numpy(self, tensor: Any) -> np.ndarray:
        return np.array(tensor, dtype=np.float32)

    def to_device(self, tensor: Any) -> Any:
        return tensor  # MLX handles device placement automatically

    def get_device_info(self) -> DeviceInfo:
        import subprocess

        result = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=False)
        total_ram = int(result.stdout.strip()) / (1024**3) if result.returncode == 0 else 0
        return DeviceInfo(
            backend=BackendType.MLX,
            device_name="Apple Silicon",
            vram_gb=total_ram,  # Unified memory
        )

    def save(self, obj: dict, path: str) -> None:
        self.mx.savez(path, **obj)

    def load(self, path: str) -> dict:
        return dict(self.mx.load(path))


_backend_instance: TorchBackend | MLXBackend | None = None


def get_backend() -> TorchBackend | MLXBackend:
    """Get the singleton backend instance."""
    global _backend_instance
    if _backend_instance is None:
        backend_type = detect_backend()
        if backend_type == BackendType.CUDA:
            _backend_instance = TorchBackend("cuda")
        elif backend_type == BackendType.MLX:
            _backend_instance = MLXBackend()
        else:
            _backend_instance = TorchBackend("cpu")
    return _backend_instance


def get_quantization_backend(
    quant_method: str,
    bits: int,
    seed: int = 42,
    group_size: int = 128,
) -> object | None:
    """Return a backend-specific quantization helper when needed."""
    if detect_backend() == BackendType.MLX and quant_method.startswith("turboquant"):
        try:
            from forge.turboquant.mlx_backend import MLXTurboQuantizer
        except Exception as exc:
            raise RuntimeError(
                "MLX TurboQuant backend requested but `forge.turboquant` dependencies are unavailable. "
                "Run from an `uv` environment with CUDA/MLX deps synchronized."
            ) from exc

        return MLXTurboQuantizer(bits=bits, method=quant_method, group_size=group_size, seed=seed)
    return None


def reset_backend() -> None:
    """Reset backend (for testing)."""
    global _backend_instance
    _backend_instance = None
