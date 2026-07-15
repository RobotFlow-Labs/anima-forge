"""Quantization CLI commands."""

from __future__ import annotations

import hashlib
import importlib
import io
import os
import tempfile
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from forge.cli_commands.shared import (
    emit_cli_error,
    emit_json,
    load_forge_config,
    resolve_runtime_device,
)

console = Console()
quantize_app = typer.Typer(name="quantize", help="Quantization backends")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _publish_torch_artifact(payload: dict[str, Any], destination: Path) -> None:
    """Serialize, sync, and atomically publish one standalone Torch artifact."""
    import torch

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


@contextmanager
def _suppress_stdout_for_json(enabled: bool):
    if not enabled:
        yield
        return
    with redirect_stdout(io.StringIO()):
        yield


def _warn(msg: str, output_json: bool) -> None:
    if output_json:
        return
    console.print(f"[yellow]{msg}[/yellow]")


def _is_cuda_oom(exc: Exception) -> bool:
    return "out of memory" in str(exc).lower()


def _require_quant_backend(method: str, *, output_json: bool) -> None:
    if method in {"turboquant-mse", "turboquant-prod", "polarquant"}:
        try:
            importlib.import_module("forge.turboquant")  # noqa: F401
        except Exception as exc:
            emit_cli_error(
                "TurboQuant backend is unavailable. Reinstall FORGE on Linux and "
                "invoke commands with `uv run ...` (repo-local module path is `forge/turboquant`). "
                f"Cause: {exc}",
                output_json=output_json,
                exit_code=2,
            )


def load_student_for_quant(
    config_path: str,
    checkpoint: str | None = None,
    *,
    allow_mock: bool = False,
    require_trained_checkpoint: bool = False,
    protected_action: str = "quantize",
):
    """Load a quantization model through the protected checkpoint boundary."""
    import torch

    from forge.checkpoint_compat import (
        apply_checkpoint_structure,
        extract_checkpoint_state_dict,
        load_checkpoint_payload,
        load_model_weights_with_compatibility,
    )
    from forge.config import apply_checkpoint_student_config
    from forge.provenance import build_provenance, validate_provenance
    from forge.student import FORGEStudent

    config = load_forge_config(config_path, required=True)
    effective_allow_mock = bool(allow_mock or config.student.allow_mock)
    config.student.allow_mock = effective_allow_mock

    if checkpoint is None and require_trained_checkpoint and not effective_allow_mock:
        action = "quantize run" if protected_action == "quantize" else protected_action
        raise ValueError(
            f"{action} requires a trained --checkpoint. Refusing to use an untrained model; "
            "use --allow-mock only for explicit test workflows."
        )

    payload = None
    source_provenance = None
    if checkpoint is not None:
        payload = load_checkpoint_payload(
            checkpoint,
            map_location="cpu",
            verify_provenance_for=protected_action,
            allow_mock=effective_allow_mock,
        )
        if payload is None:
            raise ValueError(f"Checkpoint payload is unreadable: {checkpoint}")
        source_provenance = payload.get("provenance")
        if source_provenance is None and require_trained_checkpoint and not effective_allow_mock:
            raise ValueError(
                "Checkpoint has no provenance, so its real-weight lineage cannot be verified. "
                "Use a PRD-36 checkpoint or --allow-mock only for explicit test workflows."
            )

    if payload is not None:
        apply_checkpoint_student_config(config.student, payload)
    model = FORGEStudent(config.student, model_dir=config.paths.model_dir)
    if payload is not None:
        apply_checkpoint_structure(model, payload)
        pruning_metadata = payload.get("pruning")
        if isinstance(pruning_metadata, dict):
            setattr(model, "_forge_pruning_metadata", dict(pruning_metadata))
        state_dict, _ = extract_checkpoint_state_dict(payload)
        if state_dict is None and payload and all(torch.is_tensor(value) for value in payload.values()):
            state_dict = payload
        if state_dict is None:
            raise ValueError(f"Checkpoint contains no model state dictionary: {checkpoint}")
        load_model_weights_with_compatibility(
            model,
            state_dict,
            context=f"{protected_action}:{checkpoint}",
            minimum_coverage=0.8,
        )

    if source_provenance is not None:
        provenance = validate_provenance(source_provenance)
    else:
        # An untrained or legacy input has no trustworthy label lineage. It is
        # deliberately marked mock so downstream protected operations fail closed.
        provenance = build_provenance(
            student=model,
            config=config,
            labels="mock",
        )
    return config, model, provenance


@quantize_app.command("run")
def quantize_run(
    config: str = typer.Option("configs/forge_nano.yaml", help="Config file path"),
    method: str = typer.Option("turboquant-mse", help="qvla|turboquant-mse|turboquant-prod|polarquant"),
    bits: int = typer.Option(4, min=1, max=8, help="Quantization bit-width (packed artifacts: 4 or 8)"),
    device: str = typer.Option(None, help="Quantization device: auto|cuda|cpu"),
    allow_cpu_fallback: bool = typer.Option(
        False,
        "--allow-cpu-fallback",
        help="Allow automatic CPU fallback if CUDA OOM occurs.",
    ),
    checkpoint: str = typer.Option(None, help="Optional checkpoint to quantize"),
    output: str = typer.Option(None, help="Output model path"),
    allow_mock: bool = typer.Option(
        False,
        "--allow-mock",
        help="Allow an untrained or mock-provenance input for explicit test workflows.",
    ),
    output_json: bool = typer.Option(False, "--json", help="JSON output"),
):
    """Quantize a FORGE student model with the selected backend."""
    import torch

    from forge.quantize import QUANT_METHODS as _SUPPORTED_QUANT_METHODS
    from forge.quantize import create_quant_profile, quantize_model_with_config
    from forge.quantize.serialization import PACKED_STATE_KEY, pack_state_dict

    if method not in _SUPPORTED_QUANT_METHODS:
        choices = ", ".join(sorted(_SUPPORTED_QUANT_METHODS))
        emit_cli_error(
            f"Unknown quantization method {method!r}. Choose one of: {choices}",
            output_json=output_json,
            exit_code=2,
        )
    checkpoint_path = Path(checkpoint).expanduser().resolve() if checkpoint else None
    if checkpoint_path is not None and not checkpoint_path.is_file():
        emit_cli_error(
            f"Checkpoint not found: {checkpoint_path}",
            output_json=output_json,
            exit_code=2,
        )
    if bits not in {4, 8}:
        emit_cli_error(
            f"Quantized checkpoint serialization supports --bits 4 or 8, got {bits}.",
            output_json=output_json,
            exit_code=2,
        )
    _require_quant_backend(method, output_json=output_json)
    requested_device = device or "auto"
    source_checkpoint_sha256: str | None = None
    try:
        if checkpoint_path is not None:
            source_checkpoint_sha256 = _sha256_file(checkpoint_path)
        device = resolve_runtime_device(device=device, command="quantize", default="auto", strict=True)
        with _suppress_stdout_for_json(output_json):
            cfg, model, provenance = load_student_for_quant(
                config,
                checkpoint=str(checkpoint_path) if checkpoint_path is not None else None,
                allow_mock=allow_mock,
                require_trained_checkpoint=True,
                protected_action="quantize",
            )
    except Exception as exc:
        emit_cli_error(str(exc), output_json=output_json, exit_code=2)
    fallback_device: str | None = None
    fallbacks: list[dict[str, str]] = []

    try:
        with _suppress_stdout_for_json(output_json):
            model = model.to(device)
            cfg.quant.method = method
            cfg.quant.bits = bits
            quantized = quantize_model_with_config(
                model,
                cfg,
                inplace=not allow_cpu_fallback,
            )
    except RuntimeError as exc:
        # A frequent production issue on 24GB+ consumer/edge GPUs is OOM during
        # deepcopy+quantization. Keep this command usable by falling back to CPU.
        is_cuda_oom = _is_cuda_oom(exc)
        if is_cuda_oom and device.startswith("cuda") and allow_cpu_fallback:
            fallback_device = "cpu"
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
            fallbacks.append(
                {
                    "stage": "quantize",
                    "from": device,
                    "to": "cpu",
                    "reason": "RuntimeError: out-of-memory",
                }
            )
            _warn("CUDA OOM during quantization. Retrying on CPU.", output_json)
            try:
                with _suppress_stdout_for_json(output_json):
                    model = model.to("cpu")
                    cfg.quant.method = method
                    cfg.quant.bits = bits
                    quantized = quantize_model_with_config(model, cfg, inplace=True)
            except Exception as cpu_exc:
                emit_cli_error(
                    "Quantization failed on CUDA and CPU fallback. "
                    "Reduce bits/model size, use a smaller checkpoint, or retry on a larger GPU. "
                    f"Cause: {cpu_exc}",
                    output_json=output_json,
                    exit_code=2,
                )
        elif is_cuda_oom and device.startswith("cuda"):
            emit_cli_error(
                "CUDA OOM during quantization and --allow-cpu-fallback is false. "
                "Retry with --allow-cpu-fallback or use a smaller model/quantization config.",
                output_json=output_json,
                exit_code=2,
            )
        else:
            emit_cli_error(str(exc), output_json=output_json, exit_code=2)
    except Exception as exc:
        emit_cli_error(str(exc), output_json=output_json, exit_code=2)

    try:
        profile = create_quant_profile(
            quantized,
            {},
            name=f"{method}_{bits}bit",
            uniform_bits=bits,
        )

        if output is None:
            output_dir = Path(cfg.paths.output_dir) / "compressed"
            output_dir.mkdir(parents=True, exist_ok=True)
            output = str(output_dir / f"{method}_{bits}bit.pt")

        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        packed_state, packing = pack_state_dict(quantized.state_dict(), bits=bits)
        quantization = {
            **packing,
            "method": method,
        }
        artifact = {
            PACKED_STATE_KEY: packed_state,
            "quantization": quantization,
            "provenance": provenance,
        }
        config_sha256 = getattr(cfg, "_forge_config_sha256", None)
        if isinstance(config_sha256, str):
            artifact["config_sha256"] = config_sha256
        if source_checkpoint_sha256 is not None:
            artifact["source_checkpoint_sha256"] = source_checkpoint_sha256
        pruning_metadata = getattr(quantized, "_forge_pruning_metadata", None)
        if isinstance(pruning_metadata, dict):
            artifact["pruning"] = pruning_metadata
        _publish_torch_artifact(artifact, output_path)
        artifact_sha256 = _sha256_file(output_path)
        artifact_size_mb = output_path.stat().st_size / 1e6
    except Exception as exc:
        emit_cli_error(str(exc), output_json=output_json, exit_code=2)

    result = {
        "method": method,
        "bits": bits,
        "device": fallback_device or device,
        "requested_device": requested_device,
        "fallbacks": fallbacks,
        "output": str(output_path),
        "avg_bits": bits,
        "estimated_compressed_size_mb": profile.compressed_size_mb,
        "compressed_size_mb": artifact_size_mb,
        "compression_ratio": packing["compression_ratio"],
        "serialization_schema": packing["schema"],
        "artifact_sha256": artifact_sha256,
        "provenance": provenance,
    }
    if source_checkpoint_sha256 is not None:
        result["source_checkpoint_sha256"] = source_checkpoint_sha256
    if isinstance(config_sha256, str):
        result["config_sha256"] = config_sha256
    if output_json:
        emit_json(result)
        return

    table = Table(title="Quantization Result")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    for key, value in result.items():
        table.add_row(key, str(value))
    console.print(table)


@quantize_app.command("bench")
def quantize_bench(
    config: str = typer.Option("configs/forge_nano.yaml", help="Config file path"),
    method: str = typer.Option("turboquant-mse", help="Quantization method"),
    bits: int = typer.Option(3, min=1, max=8, help="Quantization bit-width"),
    device: str = typer.Option(None, help="Benchmark device: auto|cuda|cpu"),
    allow_cpu_fallback: bool = typer.Option(
        False,
        "--allow-cpu-fallback",
        help="Allow automatic CPU fallback if CUDA OOM occurs.",
    ),
    checkpoint: str = typer.Option(None, help="Optional checkpoint to benchmark"),
    max_layers: int = typer.Option(8, help="Limit benchmark to first N linear layers"),
    allow_mock: bool = typer.Option(
        False,
        "--allow-mock",
        help="Allow a mock-provenance checkpoint for explicit test workflows.",
    ),
    output_json: bool = typer.Option(False, "--json", help="JSON output"),
):
    """Benchmark quantization distortion on a FORGE student."""
    import torch

    requested_device = device or "auto"
    from forge.quantize import benchmark_quantization

    if checkpoint and not Path(checkpoint).is_file():
        emit_cli_error(
            f"Checkpoint not found: {checkpoint}",
            output_json=output_json,
            exit_code=2,
        )
    _require_quant_backend(method, output_json=output_json)
    try:
        device = resolve_runtime_device(device=device, command="quantize", default="auto", strict=True)
        with _suppress_stdout_for_json(output_json):
            _, model, provenance = load_student_for_quant(
                config,
                checkpoint=checkpoint,
                allow_mock=allow_mock,
                protected_action="benchmark",
            )
    except Exception as exc:
        emit_cli_error(str(exc), output_json=output_json, exit_code=2)
    fallback_device: str | None = None
    fallbacks: list[dict[str, str]] = []

    try:
        with _suppress_stdout_for_json(output_json):
            model = model.to(device)
            report = benchmark_quantization(model, method=method, bits=bits, max_layers=max_layers)
    except RuntimeError as exc:
        is_cuda_oom = _is_cuda_oom(exc)
        if is_cuda_oom and device.startswith("cuda") and allow_cpu_fallback:
            fallback_device = "cpu"
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
            fallbacks.append(
                {
                    "stage": "bench",
                    "from": device,
                    "to": "cpu",
                    "reason": "RuntimeError: out-of-memory",
                }
            )
            _warn("CUDA OOM during quantization benchmark. Retrying on CPU.", output_json)
            try:
                with _suppress_stdout_for_json(output_json):
                    model = model.to("cpu")
                    report = benchmark_quantization(model, method=method, bits=bits, max_layers=max_layers)
            except Exception as cpu_exc:
                emit_cli_error(
                    f"Quantization benchmark failed on CUDA and CPU fallback. Cause: {cpu_exc}",
                    output_json=output_json,
                    exit_code=2,
                )
        elif is_cuda_oom and device.startswith("cuda"):
            emit_cli_error(
                "Benchmark failed from CUDA OOM and --allow-cpu-fallback is false. "
                "Retry with --allow-cpu-fallback or use smaller benchmark scope.",
                output_json=output_json,
                exit_code=2,
            )
        else:
            emit_cli_error(str(exc), output_json=output_json, exit_code=2)
    except Exception as exc:
        emit_cli_error(str(exc), output_json=output_json, exit_code=2)

    report_payload: dict[str, Any] = dict(report)
    report_payload.update(
        {
            "requested_device": requested_device,
            "device": fallback_device or device,
            "fallbacks": fallbacks,
            "provenance": provenance,
        }
    )

    if output_json:
        emit_json(report_payload)
        return

    table = Table(title="Quantization Benchmark")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    for key, value in report_payload.items():
        table.add_row(str(key), str(value))
    console.print(table)
