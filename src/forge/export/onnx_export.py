"""ONNX export for FORGE models.

Universal format: works on any CPU/GPU via ONNX Runtime.
Also serves as intermediate format for TensorRT conversion.

Usage:
    forge pipeline --stage export --checkpoint outputs/checkpoints/final.pt --output-dir outputs/export
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn

from forge.export.onnx_artifacts import resolve_onnx_artifact_family

logger = logging.getLogger(__name__)


class _ActionOnlyExportWrapper(nn.Module):
    """Expose the deployment action tensor instead of training-only outputs."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor, language_ids: torch.Tensor) -> torch.Tensor:
        output = self.model(images, language_ids=language_ids)
        if not isinstance(output, dict) or "actions" not in output:
            raise RuntimeError("ONNX export model must return a mapping containing 'actions'")
        return cast(torch.Tensor, output["actions"])


def _onnx_artifact_files(onnx_path: Path) -> list[Path]:
    """Return the graph plus every local external-data file it references."""
    return list(resolve_onnx_artifact_family(onnx_path).values())


def export_onnx(
    model: nn.Module,
    output_path: str | Path,
    image_size: int = 384,
    max_seq_len: int = 128,
    opset_version: int = 19,
    optimize: bool = True,
) -> Path:
    """Export FORGE model to ONNX format.

    Args:
        model: Trained FORGE student model
        output_path: Path for the .onnx file
        image_size: Input image size
        max_seq_len: Max language token sequence length
        opset_version: ONNX opset version
        optimize: Apply ONNX graph optimizations

    Returns:
        Path to exported ONNX file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    # ONNX Runtime's CPU provider does not implement every bfloat16 kernel
    # emitted by the v3 backbones (notably elementwise Mul).  ONNX is the
    # universal CPU/GPU artifact, so normalize floating parameters and buffers
    # to float32 before tracing instead of producing a graph that serializes but
    # cannot be executed on CPU.
    model = _ActionOnlyExportWrapper(model.cpu().float()).eval()

    # Dummy inputs
    # A sample size greater than one keeps the dynamo exporter from specializing
    # the batch dimension before it applies the dynamic-shape constraints.
    dummy_images = torch.zeros(2, 3, image_size, image_size)
    dummy_lang = torch.zeros((2, max_seq_len), dtype=torch.int64)

    batch = torch.export.Dim("batch", min=1)
    seq_len = torch.export.Dim("seq_len", min=1)
    dynamic_shapes = {
        "images": {0: batch},
        "language_ids": {0: batch, 1: seq_len},
    }

    logger.info(f"Exporting to ONNX: {output_path}")

    requested_opset = max(int(opset_version), 18)
    candidates = [requested_opset]
    if requested_opset > 18:
        candidates.append(18)
    elif requested_opset != 19:
        candidates.append(19)
        candidates.append(18)

    export_error: Exception | None = None
    for candidate_opset in dict.fromkeys(candidates):
        try:
            torch.onnx.export(
                model,
                (dummy_images, dummy_lang),
                str(output_path),
                input_names=["images", "language_ids"],
                output_names=["actions"],
                dynamo=True,
                dynamic_shapes=dynamic_shapes,
                opset_version=candidate_opset,
                do_constant_folding=True,
            )
            logger.info(f"ONNX export complete with opset {candidate_opset}")
            break
        except Exception as exc:
            logger.warning(f"ONNX export failed at opset {candidate_opset}: {exc}")
            export_error = exc
            continue
    else:
        raise RuntimeError(f"ONNX export failed for all candidate opsets: {export_error}") from export_error

    logger.info(f"ONNX export complete: {output_path} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")

    if optimize:
        optimized_path = _optimize_onnx(output_path)
        return optimized_path

    return output_path


def _optimize_onnx(onnx_path: Path) -> Path:
    """Apply ONNX Runtime graph optimizations."""
    try:
        import onnxruntime as ort  # type: ignore[import-untyped]

        optimized_path = onnx_path.with_suffix(".optimized.onnx")
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.optimized_model_filepath = str(optimized_path)

        ort.InferenceSession(str(onnx_path), sess_options)
        logger.info(f"Optimized ONNX: {optimized_path}")
        return optimized_path

    except ImportError:
        logger.warning("onnxruntime not installed, skipping optimization")
        return onnx_path
    except Exception as e:
        logger.warning(f"ONNX optimization failed: {e}")
        return onnx_path


def validate_onnx(
    pytorch_model: nn.Module,
    onnx_path: str | Path,
    n_samples: int = 10,
    tolerance: float = 0.01,
    images: torch.Tensor | None = None,
    language_ids: torch.Tensor | None = None,
) -> dict[str, object]:
    """Validate deterministic pointwise parity or stochastic distribution parity."""
    try:
        import onnxruntime as ort
    except ImportError:
        logger.warning("onnxruntime not installed, skipping validation")
        return {"status": "skipped", "reason": "onnxruntime not installed"}

    import numpy as np

    session = ort.InferenceSession(str(onnx_path))
    pytorch_model.eval()
    pytorch_model = pytorch_model.cpu()

    max_diff = 0.0
    sample_images = torch.zeros(1, 3, 384, 384) if images is None else images.detach().cpu().float()
    sample_language = (
        torch.zeros((len(sample_images), 128), dtype=torch.int64)
        if language_ids is None
        else language_ids.detach().cpu().to(torch.int64)
    )
    probe_inputs = {
        "images": sample_images[:1].numpy(),
        "language_ids": sample_language[:1].numpy(),
    }
    first_probe = session.run(["actions"], probe_inputs)[0]
    second_probe = session.run(["actions"], probe_inputs)[0]
    stochastic_graph = not np.array_equal(first_probe, second_probe)
    validation_samples = max(n_samples, 32) if stochastic_graph else n_samples
    pytorch_samples: list[np.ndarray[Any, np.dtype[Any]]] = []
    onnx_samples: list[np.ndarray[Any, np.dtype[Any]]] = []

    for index in range(validation_samples):
        sample_index = index % len(sample_images)
        image = sample_images[sample_index : sample_index + 1]
        lang = sample_language[sample_index : sample_index + 1]

        # PyTorch
        with torch.no_grad():
            pt_out = pytorch_model(image, language_ids=lang)
            pt_actions = pt_out["actions"].numpy()

        # ONNX
        ort_inputs = {
            "images": image.numpy(),
            "language_ids": lang.numpy(),
        }
        ort_actions = session.run(["actions"], ort_inputs)[0]

        if pt_actions.shape != ort_actions.shape:
            return {
                "status": "failed",
                "validation_mode": ("stochastic_runtime_contract" if stochastic_graph else "pointwise_parity"),
                "reason": f"output shape mismatch: PyTorch {pt_actions.shape}, ONNX {ort_actions.shape}",
                "n_samples": index + 1,
            }
        if not np.isfinite(pt_actions).all() or not np.isfinite(ort_actions).all():
            return {
                "status": "failed",
                "validation_mode": ("stochastic_runtime_contract" if stochastic_graph else "pointwise_parity"),
                "reason": "non-finite action output",
                "n_samples": index + 1,
            }

        diff = np.abs(pt_actions - ort_actions).max()
        max_diff = max(max_diff, diff)
        if stochastic_graph:
            pytorch_samples.append(pt_actions)
            onnx_samples.append(ort_actions)

    if stochastic_graph:
        pytorch_distribution = np.stack(pytorch_samples)
        onnx_distribution = np.stack(onnx_samples)
        pytorch_mean = pytorch_distribution.mean(axis=0)
        onnx_mean = onnx_distribution.mean(axis=0)
        pytorch_std = pytorch_distribution.std(axis=0, ddof=1)
        onnx_std = onnx_distribution.std(axis=0, ddof=1)

        mean_difference = np.abs(pytorch_mean - onnx_mean)
        std_difference = np.abs(pytorch_std - onnx_std)
        # The two runtimes draw independent noise. Compare moments using a
        # four-standard-error envelope plus the caller's numerical tolerance.
        mean_limit = tolerance + 4.0 * np.sqrt((pytorch_std**2 + onnx_std**2) / validation_samples)
        std_limit = tolerance + 4.0 * np.sqrt((pytorch_std**2 + onnx_std**2) / (2.0 * (validation_samples - 1)))
        means_match = bool(np.all(mean_difference <= mean_limit))
        stds_match = bool(np.all(std_difference <= std_limit))
        passed = means_match and stds_match
        return {
            "status": "passed" if passed else "failed",
            "validation_mode": "stochastic_runtime_contract",
            "pointwise_comparable": False,
            "max_diff": None,
            "max_observed_difference": float(max_diff),
            "max_mean_difference": float(mean_difference.max()),
            "max_std_difference": float(std_difference.max()),
            "tolerance": tolerance,
            "n_samples": validation_samples,
            "checks": {
                "output_shape_matches": True,
                "pytorch_actions_finite": True,
                "onnx_actions_finite": True,
                "per_dimension_means_match": means_match,
                "per_dimension_stds_match": stds_match,
            },
        }

    passed = max_diff < tolerance
    return {
        "status": "passed" if passed else "failed",
        "validation_mode": "pointwise_parity",
        "pointwise_comparable": True,
        "max_diff": float(max_diff),
        "tolerance": tolerance,
        "n_samples": n_samples,
    }


def benchmark_onnx_runtime(
    onnx_path: str | Path,
    *,
    device: str = "cpu",
    n_warmup: int = 5,
    n_runs: int = 50,
    image_size: int = 384,
    sequence_length: int = 128,
    images: torch.Tensor | None = None,
    language_ids: torch.Tensor | None = None,
) -> dict[str, object]:
    """Measure an exported model with an explicit ONNX Runtime provider."""
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError:
        return {"status": "skipped", "reason": "onnxruntime not installed"}

    onnx_path = Path(onnx_path).expanduser().resolve()
    available = ort.get_available_providers()
    selected_device = torch.device(device)
    wants_cuda = selected_device.type == "cuda"
    provider = "CUDAExecutionProvider" if wants_cuda else "CPUExecutionProvider"
    if provider not in available:
        return {
            "status": "skipped",
            "reason": f"{provider} is unavailable",
            "available_providers": available,
        }

    provider_device_id: int | None = None
    if wants_cuda:
        provider_device_id = selected_device.index if selected_device.index is not None else 0
        session = ort.InferenceSession(
            str(onnx_path),
            providers=[provider],
            provider_options=[{"device_id": provider_device_id}],
        )
    else:
        session = ort.InferenceSession(str(onnx_path), providers=[provider])
    actual_provider = session.get_providers()[0]
    if actual_provider != provider:
        return {
            "status": "failed",
            "reason": f"requested {provider}, session selected {actual_provider}",
        }

    inputs: dict[str, np.ndarray[Any, np.dtype[Any]]] = {}
    for model_input in session.get_inputs():
        if model_input.name == "images":
            inputs[model_input.name] = (
                np.zeros((1, 3, image_size, image_size), dtype=np.float32)
                if images is None
                else images[:1].detach().cpu().numpy().astype(np.float32, copy=False)
            )
        elif model_input.name == "language_ids":
            inputs[model_input.name] = (
                np.zeros((1, sequence_length), dtype=np.int64)
                if language_ids is None
                else language_ids[:1].detach().cpu().numpy().astype(np.int64, copy=False)
            )
        else:
            raise ValueError(f"Unsupported ONNX benchmark input: {model_input.name}")

    output_names = [model_output.name for model_output in session.get_outputs()]
    if "actions" not in output_names:
        return {"status": "failed", "reason": "ONNX runtime output does not contain actions"}
    actions_index = output_names.index("actions")

    for _ in range(n_warmup):
        session.run(None, inputs)

    timings_ms: list[float] = []
    actions_shape: list[int] = []
    action_samples = 0
    for _ in range(n_runs):
        started = time.perf_counter()
        outputs = session.run(None, inputs)
        timings_ms.append((time.perf_counter() - started) * 1000)
        actions = np.asarray(outputs[actions_index])
        if actions.size < 1 or not np.isfinite(actions).all():
            return {"status": "failed", "reason": "ONNX runtime actions are empty or non-finite"}
        actions_shape = [int(dimension) for dimension in actions.shape]
        action_samples += int(actions.shape[0]) if actions.ndim > 0 else 1

    values = np.asarray(timings_ms, dtype=np.float64)
    mean_ms = float(values.mean())
    artifact_files = _onnx_artifact_files(onnx_path)
    artifact_size_bytes = sum(path.stat().st_size for path in artifact_files)
    return {
        "status": "success",
        "provider": actual_provider,
        "device": str(selected_device),
        "provider_device_id": provider_device_id,
        "mean_ms": mean_ms,
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "fps": 1000.0 / mean_ms,
        "warmup_runs": n_warmup,
        "measured_runs": n_runs,
        "onnx_path": str(onnx_path),
        "onnx_size_mb": artifact_size_bytes / 1e6,
        "onnx_graph_size_mb": onnx_path.stat().st_size / 1e6,
        "artifact_files": [str(path) for path in artifact_files],
        "actions_finite": True,
        "actions_shape": actions_shape,
        "action_samples": action_samples,
    }
