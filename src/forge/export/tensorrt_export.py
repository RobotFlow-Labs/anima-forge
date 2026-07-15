"""TensorRT export for Jetson deployment.

Converts ONNX → TensorRT engine with INT8 calibration.
Requires NVIDIA GPU and TensorRT SDK (not available on Mac).

Usage:
    forge pipeline --stage export --checkpoint outputs/checkpoints/final.pt --output-dir outputs/export
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


def write_tensorrt_calibration_archive(
    images: torch.Tensor,
    language_ids: torch.Tensor,
    output_path: str | Path,
) -> Path:
    """Write provenance-selected runtime inputs for TensorRT INT8 calibration."""
    import numpy as np

    if images.ndim != 4 or images.shape[1:] != (3, 384, 384):
        raise ValueError(f"TensorRT calibration images must be Nx3x384x384, got {tuple(images.shape)}")
    if language_ids.ndim != 2 or language_ids.shape[0] != images.shape[0]:
        raise ValueError("TensorRT calibration language_ids must be NxS and align with images")
    if images.shape[0] < 1:
        raise ValueError("TensorRT calibration requires at least one real observation")
    if not torch.isfinite(images).all():
        raise ValueError("TensorRT calibration images contain non-finite values")

    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        images=images.detach().cpu().to(torch.float32).numpy(),
        language_ids=language_ids.detach().cpu().to(torch.int64).numpy(),
    )
    return path


def get_tensorrt_status() -> dict[str, object]:
    """Return runtime TensorRT availability metadata."""
    status: dict[str, object] = {
        "available": False,
        "version": None,
        "cuda_available": torch.cuda.is_available(),
        "error": None,
    }
    try:
        import tensorrt as trt  # type: ignore[import-untyped]

        status["available"] = True
        status["version"] = getattr(trt, "__version__", "unknown")
    except Exception as exc:
        status["error"] = str(exc)
    return status


def export_tensorrt(
    onnx_path: str | Path,
    output_path: str | Path,
    precision: str = "fp16",
    workspace_mb: int = 2048,
    calibration_data: str | None = None,
    device: str = "cuda",
) -> Path:
    """Export an ONNX model on the explicitly selected CUDA device."""
    selected_device = torch.device(device)
    if selected_device.type != "cuda":
        raise ValueError(f"TensorRT export requires a CUDA device, got {device!r}")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "TensorRT export requires CUDA runtime and hardware. Run this command on a CUDA-enabled machine."
        )
    with torch.cuda.device(selected_device):
        return _export_tensorrt_on_selected_device(
            onnx_path,
            output_path,
            precision=precision,
            workspace_mb=workspace_mb,
            calibration_data=calibration_data,
            device=str(selected_device),
        )


def _export_tensorrt_on_selected_device(
    onnx_path: str | Path,
    output_path: str | Path,
    precision: str = "fp16",
    workspace_mb: int = 2048,
    calibration_data: str | None = None,
    device: str = "cuda",
) -> Path:
    """Export ONNX model to TensorRT engine.

    REQUIRES: NVIDIA GPU + TensorRT SDK
    This function will fail on Mac — use on CUDA machine only.

    Args:
        onnx_path: Path to ONNX model
        output_path: Path for TensorRT .engine file
        precision: "fp16" or "int8"
        workspace_mb: TensorRT workspace memory
        calibration_data: Path to calibration data (required for INT8)

    Returns:
        Path to TensorRT engine
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx_path = Path(onnx_path)

    if precision not in {"fp16", "int8"}:
        raise ValueError("precision must be 'fp16' or 'int8'")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "TensorRT export requires CUDA runtime and hardware. Run this command on a CUDA-enabled machine."
        )

    if precision == "int8" and not calibration_data:
        raise ValueError("TensorRT INT8 export requires calibration_data path")

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX input not found: {onnx_path}")

    if calibration_data is not None and not Path(calibration_data).exists():
        raise FileNotFoundError(f"TensorRT INT8 calibration file not found: {calibration_data}")

    try:
        import tensorrt as trt  # type: ignore[import-untyped]
    except ImportError:
        status = get_tensorrt_status()
        logger.error("TensorRT unavailable: %s", status)
        raise RuntimeError(
            "TensorRT runtime package could not be imported. Reinstall FORGE on Linux and run via `uv run forge ...`."
        ) from None
    except Exception as exc:
        logger.error("TensorRT import failed: %s", exc)
        raise RuntimeError(
            "TensorRT import succeeded for presence check but runtime initialization failed. "
            "Confirm native TensorRT libraries match installed TensorRT Python package."
        ) from exc

    logger.info(f"Building TensorRT engine: {onnx_path} → {output_path} (precision={precision})")
    workspace_limit_mb = _resolve_workspace_budget(workspace_mb, device=device)

    trt_logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, trt_logger)

    # Use TensorRT's path-aware parser. Large torch ONNX exports keep weights in
    # a relative external-data sidecar (for example ``forge.onnx.data``). Feeding
    # only graph bytes to ``parser.parse`` loses the model directory, so
    # TensorRT looks for the sidecar in the process working directory.
    if not _parse_onnx_file(parser, onnx_path):
        for i in range(parser.num_errors):
            logger.error(f"TensorRT parse error: {parser.get_error(i)}")
        raise RuntimeError("Failed to parse ONNX model")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_limit_mb * (1 << 20))
    _add_dynamic_shape_profile(builder, network, config)

    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        # TensorRT applies INT8 only where calibration and an INT8 kernel are
        # available.  Permit FP16 kernels for the remaining layers so they do
        # not inflate the engine by falling back to FP32.
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.INT8)
        if calibration_data:
            config.int8_calibrator = _create_calibrator(calibration_data, device=device)

    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError("Failed to build TensorRT engine")

    with open(str(output_path), "wb") as f:
        f.write(engine)

    logger.info(f"TensorRT engine saved: {output_path} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")
    return output_path


def _parse_onnx_file(parser: Any, onnx_path: Path) -> bool:
    """Parse an ONNX graph while preserving relative external-data paths."""
    return bool(parser.parse_from_file(str(onnx_path.resolve())))


def _add_dynamic_shape_profile(builder: Any, network: Any, config: Any) -> None:
    """Attach a bounded profile when an ONNX input has dynamic dimensions."""
    dynamic_inputs = []
    for index in range(network.num_inputs):
        tensor = network.get_input(index)
        shape = tuple(int(dim) for dim in tensor.shape)
        if any(dim < 0 for dim in shape):
            dynamic_inputs.append((tensor.name, shape))

    if not dynamic_inputs:
        return

    profile = builder.create_optimization_profile()
    for name, shape in dynamic_inputs:
        minimum = tuple(1 if dim < 0 else dim for dim in shape)
        optimum = tuple(128 if dim < 0 and axis > 0 else 1 if dim < 0 else dim for axis, dim in enumerate(shape))
        maximum = tuple(256 if dim < 0 and axis > 0 else 4 if dim < 0 else dim for axis, dim in enumerate(shape))
        # TensorRT 10 returns None on success and raises ValueError for an
        # inconsistent shape tuple.
        profile.set_shape(name, minimum, optimum, maximum)

    if config.add_optimization_profile(profile) < 0:
        raise RuntimeError("Failed to attach TensorRT optimization profile")


def _resolve_workspace_budget(requested_mb: int, *, device: str = "cuda") -> int:
    """Resolve TensorRT workspace budget with small-GPU caps."""
    requested = max(int(requested_mb), 64)

    if torch.cuda.is_available():
        try:
            total_mem_mb = torch.cuda.get_device_properties(device).total_memory // (1 << 20)
            cap_mb = max(128, total_mem_mb // 4)  # avoid overcommitting small cards
            if requested > cap_mb:
                logger.warning(
                    "TensorRT workspace too high for current GPU; capping from %s MB to %s MB",
                    requested,
                    cap_mb,
                )
                return cap_mb
        except Exception as exc:
            logger.warning(f"Could not resolve CUDA memory for workspace cap: {exc}")

    return requested


def _create_calibrator(data_path: str, *, device: str = "cuda"):
    """Create an entropy calibrator backed by saved real observations."""
    import numpy as np
    import tensorrt as trt  # type: ignore[import-untyped]

    path = Path(data_path).expanduser().resolve()
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != {"images", "language_ids"}:
            raise ValueError("TensorRT calibration archive must contain images and language_ids")
        images = np.asarray(payload["images"], dtype=np.float32)
        language_ids = np.asarray(payload["language_ids"], dtype=np.int64)
    if images.ndim != 4 or images.shape[1:] != (3, 384, 384):
        raise ValueError(f"Invalid TensorRT calibration image shape: {images.shape}")
    if language_ids.ndim != 2 or language_ids.shape[0] != images.shape[0]:
        raise ValueError("TensorRT calibration language_ids do not align with images")

    class RealEntropyCalibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self) -> None:
            super().__init__()
            self._offset = 0
            self._device_inputs: dict[str, torch.Tensor] = {}
            self._cache_path = path.with_suffix(".cache")

        def get_batch_size(self) -> int:
            return 1

        def get_batch(self, names: list[str]) -> list[int] | None:
            if self._offset >= images.shape[0]:
                return None
            self._device_inputs = {
                "images": torch.from_numpy(images[self._offset : self._offset + 1]).to(device).contiguous(),
                "language_ids": torch.from_numpy(language_ids[self._offset : self._offset + 1]).to(device).contiguous(),
            }
            self._offset += 1
            unknown = [name for name in names if name not in self._device_inputs]
            if unknown:
                raise ValueError(f"Unsupported TensorRT calibration inputs: {unknown}")
            return [self._device_inputs[name].data_ptr() for name in names]

        def read_calibration_cache(self) -> bytes | None:
            # Release validation must execute the supplied real observations on
            # every build; an older cache may belong to another ONNX graph.
            return None

        def write_calibration_cache(self, cache: bytes) -> None:
            self._cache_path.write_bytes(cache)

    return RealEntropyCalibrator()


def benchmark_tensorrt_runtime(
    engine_path: str | Path,
    *,
    n_warmup: int = 5,
    n_runs: int = 50,
    image_size: int = 384,
    sequence_length: int = 128,
    images: torch.Tensor | None = None,
    language_ids: torch.Tensor | None = None,
    precision: str = "unknown",
    device: str = "cuda",
) -> dict[str, object]:
    """Deserialize and measure an engine on the explicitly selected CUDA device."""
    if n_warmup < 0 or n_runs < 1:
        raise ValueError("TensorRT benchmark run counts must be positive")
    path = Path(engine_path).expanduser().resolve()
    if not path.is_file():
        return {"status": "failed", "reason": f"TensorRT engine not found: {path}"}
    if not torch.cuda.is_available():
        return {"status": "skipped", "reason": "CUDA is unavailable"}
    selected_device = torch.device(device)
    if selected_device.type != "cuda":
        return {"status": "failed", "reason": f"TensorRT benchmark requires a CUDA device, got {device!r}"}
    with torch.cuda.device(selected_device):
        return _benchmark_tensorrt_runtime_on_selected_device(
            engine_path,
            n_warmup=n_warmup,
            n_runs=n_runs,
            image_size=image_size,
            sequence_length=sequence_length,
            images=images,
            language_ids=language_ids,
            precision=precision,
            device=str(selected_device),
        )


def _benchmark_tensorrt_runtime_on_selected_device(
    engine_path: str | Path,
    *,
    n_warmup: int = 5,
    n_runs: int = 50,
    image_size: int = 384,
    sequence_length: int = 128,
    images: torch.Tensor | None = None,
    language_ids: torch.Tensor | None = None,
    precision: str = "unknown",
    device: str = "cuda",
) -> dict[str, object]:
    """Deserialize and measure a FORGE TensorRT engine on the selected GPU."""
    if n_warmup < 0 or n_runs < 1:
        raise ValueError("TensorRT benchmark run counts must be positive")

    path = Path(engine_path).expanduser().resolve()
    if not path.is_file():
        return {"status": "failed", "reason": f"TensorRT engine not found: {path}"}
    if not torch.cuda.is_available():
        return {"status": "skipped", "reason": "CUDA is unavailable"}

    try:
        import numpy as np
        import tensorrt as trt  # type: ignore[import-untyped]

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(path.read_bytes())
        if engine is None:
            return {"status": "failed", "reason": "TensorRT engine deserialization failed"}
        context = engine.create_execution_context()
        if context is None:
            return {"status": "failed", "reason": "TensorRT execution context creation failed"}

        tensors: dict[str, torch.Tensor] = {}
        for index in range(engine.num_io_tensors):
            name = engine.get_tensor_name(index)
            if engine.get_tensor_mode(name) != trt.TensorIOMode.INPUT:
                continue
            shape: tuple[int, ...]
            if name == "images":
                shape = (1, 3, image_size, image_size)
                tensor = (
                    torch.zeros(shape, device=device, dtype=torch.float32)
                    if images is None
                    else images[:1].to(device=device, dtype=torch.float32)
                )
            elif name == "language_ids":
                shape = (1, sequence_length)
                tensor = (
                    torch.zeros(shape, device=device, dtype=torch.int64)
                    if language_ids is None
                    else language_ids[:1].to(device=device, dtype=torch.int64)
                )
            else:
                return {"status": "failed", "reason": f"Unsupported TensorRT input: {name}"}
            if any(int(dim) < 0 for dim in engine.get_tensor_shape(name)):
                if not context.set_input_shape(name, shape):
                    return {"status": "failed", "reason": f"Failed to set TensorRT input shape: {name}"}
            tensors[name] = tensor

        for index in range(engine.num_io_tensors):
            name = engine.get_tensor_name(index)
            if engine.get_tensor_mode(name) != trt.TensorIOMode.OUTPUT:
                continue
            shape = tuple(int(dim) for dim in context.get_tensor_shape(name))
            if any(dim < 0 for dim in shape):
                return {"status": "failed", "reason": f"Unresolved TensorRT output shape: {name} {shape}"}
            dtype = _torch_dtype_for_tensorrt(trt, engine.get_tensor_dtype(name))
            tensors[name] = torch.empty(shape, device=device, dtype=dtype)

        for name, tensor in tensors.items():
            if not context.set_tensor_address(name, tensor.data_ptr()):
                return {"status": "failed", "reason": f"Failed to bind TensorRT tensor: {name}"}

        stream = torch.cuda.Stream(device=device)
        for _ in range(n_warmup):
            if not context.execute_async_v3(stream.cuda_stream):
                return {"status": "failed", "reason": "TensorRT warmup execution failed"}
        stream.synchronize()

        latencies_ms: list[float] = []
        for _ in range(n_runs):
            start = time.perf_counter()
            if not context.execute_async_v3(stream.cuda_stream):
                return {"status": "failed", "reason": "TensorRT measured execution failed"}
            stream.synchronize()
            latencies_ms.append((time.perf_counter() - start) * 1000)

        actions = tensors.get("actions")
        if actions is None:
            return {"status": "failed", "reason": "TensorRT engine has no actions output"}
        actions_finite = bool(torch.isfinite(actions).all().item())
        if not actions_finite:
            return {"status": "failed", "reason": "TensorRT actions contain non-finite values"}

        mean_ms = float(np.mean(latencies_ms))
        return {
            "status": "success",
            "provider": "TensorRT",
            "version": trt.__version__,
            "precision": precision,
            "mean_ms": mean_ms,
            "p50_ms": float(np.percentile(latencies_ms, 50)),
            "p95_ms": float(np.percentile(latencies_ms, 95)),
            "fps": 1000.0 / mean_ms,
            "warmup_runs": n_warmup,
            "measured_runs": n_runs,
            "engine_path": str(path),
            "engine_size_mb": path.stat().st_size / 1e6,
            "io_tensors": engine.num_io_tensors,
            "actions_shape": list(actions.shape),
            "actions_finite": actions_finite,
        }
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        return {"status": "failed", "reason": str(exc)}


def _torch_dtype_for_tensorrt(trt: object, dtype: object) -> torch.dtype:
    """Translate TensorRT output dtypes to torch allocation dtypes."""
    mapping = {
        trt_dtype: torch_dtype
        for name, torch_dtype in (
            ("float32", torch.float32),
            ("float16", torch.float16),
            ("bfloat16", torch.bfloat16),
            ("int8", torch.int8),
            ("int32", torch.int32),
            ("int64", torch.int64),
            ("bool", torch.bool),
            ("uint8", torch.uint8),
        )
        if (trt_dtype := getattr(trt, name, None)) is not None
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported TensorRT tensor dtype: {dtype}")
    return mapping[dtype]


def check_tensorrt_available() -> bool:
    """Check if TensorRT is available on this system."""
    return bool(get_tensorrt_available())


def get_tensorrt_available() -> bool:
    """Compatibility helper returning TensorRT availability."""
    return get_tensorrt_status().get("available", False) is True
