"""FORGE End-to-End Pipeline Runner.

Chains all 4 stages into a single command:
  Stage 1: Teacher Label Generation (PRD-01)
  Stage 2: Student Init + Knowledge Distillation (PRD-02/03)
  Stage 3: Compression — Pruning + Quantization (PRD-04/05)
  Stage 4: Export + Validation (PRD-06/07)

Usage:
    forge pipeline run --config configs/forge_nano.yaml
    forge pipeline run --config configs/forge_nano.yaml --skip-labels  # If labels already generated
    forge pipeline run --config configs/forge_nano.yaml --stage export  # Run single stage
"""

from __future__ import annotations

import hashlib
import logging
import os
import platform
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from forge.checkpoint_compat import (
    CheckpointLoadReport,
    apply_checkpoint_structure,
    extract_checkpoint_state_dict,
    load_checkpoint_payload,
    load_model_weights_with_compatibility,
    summarize_checkpoint_report,
)
from forge.config import ForgeConfig
from forge.export.onnx_artifacts import resolve_onnx_artifact_family
from forge.json_artifacts import write_json_artifact
from forge.provenance import build_provenance, checkpoint_provenance, current_git_sha

logger = logging.getLogger(__name__)


def _latest_checkpoint(output_dir: Path) -> Path | None:
    """Find the newest checkpoint in compression/checkpoint directories."""
    candidates: list[Path] = []
    for folder in [output_dir / "compressed", output_dir / "checkpoints", output_dir / "train-runs"]:
        if folder.exists():
            candidates.extend(path for path in folder.rglob("*.pt") if path.is_file())

    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _checkpoint_sha256(path: Path) -> str:
    """Flush and hash a completed checkpoint so summaries bind durable bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        if path.stat().st_size <= 0:
            raise ValueError(f"Final checkpoint is empty: {path}")
        # torch.save has closed the writer before this helper runs. fsync makes
        # that completed file durable before its digest is published.
        stream.flush()
        os.fsync(stream.fileno())
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_family_sha256(path: Path) -> dict[str, str]:
    """Hash the exact graph-plus-external-data family declared by an ONNX model."""
    return {name: _checkpoint_sha256(artifact) for name, artifact in resolve_onnx_artifact_family(path).items()}


def _load_export_runtime_inputs(
    student: torch.nn.Module,
    config: ForgeConfig,
    *,
    max_samples: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Load real label observations and tokenize their actual instructions."""
    from forge.data.teacher_dataset import TeacherLabelDataset

    label_dir = Path(config.paths.data_dir) / "teacher_labels"
    dataset = TeacherLabelDataset(label_dir, sample_timestep="first")
    try:
        provenance = dataset.labels_provenance
        if provenance != "real" and not config.student.allow_mock:
            raise ValueError(f"Export runtime inputs are not provenance-verified real labels: {label_dir}")
        count = min(len(dataset), max_samples)
        if count < 1:
            raise ValueError(f"Export requires at least one label observation: {label_dir}")
        samples = [dataset[index] for index in range(count)]
    finally:
        dataset.close()

    images = torch.stack([sample["image"] for sample in samples]).to(torch.float32)
    tokenizer = getattr(student, "tokenizer", None)
    if tokenizer is None:
        if not config.student.allow_mock:
            raise RuntimeError("Real export requires the student's local tokenizer")
        language_ids = torch.zeros((count, 128), dtype=torch.int64)
    else:
        tokenized = tokenizer(
            [str(sample["language_instruction"]) for sample in samples],
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        language_ids = tokenized["input_ids"].to(torch.int64)
    return images, language_ids, provenance


def _persist_export_runtime_inputs(
    images: torch.Tensor,
    language_ids: torch.Tensor,
    *,
    output_dir: Path,
    labels_provenance: str,
) -> dict[str, Any]:
    """Persist the exact real inputs used by every export runtime."""
    from forge.export.tensorrt_export import write_tensorrt_calibration_archive

    path = write_tensorrt_calibration_archive(
        images,
        language_ids,
        output_dir / "tensorrt_calibration.npz",
    )
    return {
        "status": "success",
        "samples": len(images),
        "labels_provenance": labels_provenance,
        "path": str(path),
        "sha256": _checkpoint_sha256(path),
    }


def _packed_compression_payload(
    *,
    packed_state: dict[str, Any],
    quantization: dict[str, Any],
    pruning: dict[str, Any],
    provenance: dict[str, Any],
    source_checkpoint_sha256: str,
    config_sha256: str | None = None,
) -> dict[str, Any]:
    """Build a packed checkpoint bound to the exact pruned source bytes."""
    from forge.quantize.serialization import PACKED_STATE_KEY

    payload = {
        PACKED_STATE_KEY: packed_state,
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "quantization": quantization,
        "pruning": pruning,
        "provenance": provenance,
    }
    if config_sha256 is not None:
        payload["config_sha256"] = config_sha256
    return payload


def _extract_checkpoint_state_dict(ckpt: dict[str, Any]) -> dict[str, object] | None:
    """Extract a model state-dict payload from FORGE checkpoint formats."""
    for key in ("model_state_dict", "student_state_dict", "state_dict"):
        payload = ckpt.get(key)
        if isinstance(payload, dict):
            return payload

    model = ckpt.get("model")
    if isinstance(model, dict):
        return model
    return None


def _load_checkpoint_if_available(
    student: torch.nn.Module,
    ckpt_path: Path | None,
    *,
    verify_for: str | None = None,
    allow_mock: bool = False,
) -> dict[str, str] | None:
    """Load compatible weights and return any stamped provenance."""
    if not ckpt_path:
        return None

    ckpt = load_checkpoint_payload(
        str(ckpt_path),
        map_location="cpu",
        verify_provenance_for=verify_for,
        allow_mock=allow_mock,
    )
    if ckpt is None:
        if verify_for is not None:
            raise ValueError(f"Checkpoint payload is unreadable: {ckpt_path}")
        logger.warning("Ignoring checkpoint %s: payload is not readable", ckpt_path)
        return None

    state_dict, extracted_key = extract_checkpoint_state_dict(ckpt)
    if state_dict is None:
        if verify_for is not None:
            raise ValueError(f"Checkpoint has no usable model state: {ckpt_path}")
        logger.warning("Ignoring checkpoint %s: payload is not a dict", ckpt_path)
        return None

    if not isinstance(state_dict, dict):
        if verify_for is not None:
            raise ValueError(f"Checkpoint model state is invalid: {ckpt_path}")
        logger.warning(
            "Ignoring checkpoint %s: extracted state dict is not a dict (source=%s)",
            ckpt_path,
            extracted_key,
        )
        return None

    apply_checkpoint_structure(student, ckpt)
    report = CheckpointLoadReport(
        source=str(ckpt_path),
        extracted_key=extracted_key,
    )
    try:
        missing, report = load_model_weights_with_compatibility(
            student,
            state_dict,
            context=f"pipeline:{ckpt_path}",
            minimum_coverage=0.8 if verify_for is not None else 0.0,
        )
    except RuntimeError as exc:
        if verify_for is not None:
            raise
        logger.warning("Could not load checkpoint %s: %s", ckpt_path, exc)
        return None

    for warning in report.warnings:
        logger.warning("%s", warning)
    logger.info(summarize_checkpoint_report("pipeline", report))

    if missing.unexpected_keys:
        logger.warning(
            "Ignored unexpected checkpoint keys: %s",
            ", ".join(missing.unexpected_keys[:8]),
        )
    if missing.missing_keys:
        logger.warning(
            "Missing checkpoint keys: %s",
            ", ".join(missing.missing_keys[:8]),
        )
    logger.info("Loaded checkpoint: %s", ckpt_path)
    provenance = checkpoint_provenance(ckpt)
    return dict(provenance) if provenance is not None else None


def _quantize_student(student: torch.nn.Module, cfg: ForgeConfig) -> torch.nn.Module:
    """Quantize on the caller-selected device and propagate runtime failures."""
    from forge.quantize import quantize_model_with_config

    return quantize_model_with_config(student, cfg, inplace=True)


def _create_quant_profile(student: torch.nn.Module, config: ForgeConfig):
    """Profile the pipeline's uniform quantization at its configured width."""
    from forge.quantize import create_quant_profile

    method = config.quant.method.replace("turboquant-", "tq-")
    return create_quant_profile(
        student,
        {},
        name=f"{method}_{config.quant.bits}bit_{config.student.variant}",
        uniform_bits=config.quant.bits,
    )


def _prepare_student_for_compression(student: torch.nn.Module, device: str) -> tuple[torch.nn.Module, str]:
    """Move the compression student to the explicitly selected device."""
    is_cuda = str(device).startswith("cuda")
    if not is_cuda or not torch.cuda.is_available():
        return student.to("cpu"), "cpu"
    return student.to(device), device


def _normalize_device(device: str | None) -> str:
    if not device:
        return "cuda" if torch.cuda.is_available() else "cpu"
    dev = device.lower().strip()
    if dev.startswith("cuda"):
        return dev
    return dev


def run_pipeline(
    config: ForgeConfig,
    device: str | None = None,
    skip_labels: bool = False,
    stage: str | None = None,
    checkpoint_path: str | Path | None = None,
    max_label_episodes: int | None = None,
    max_distill_steps: int | None = None,
    max_recovery_steps: int | None = None,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, Any]:
    """Run the complete FORGE distillation pipeline.

    Args:
        config: FORGE configuration
        device: Override device (cuda/cpu/mps)
        skip_labels: Skip teacher label generation (use existing)
        stage: Run only this stage (labels/distill/compress/export/validate)
        checkpoint_path: Explicit trained checkpoint for compress/export/validate
        max_label_episodes: Bound real or mock teacher-label episode generation
        max_distill_steps: Override max distillation steps
        max_recovery_steps: Override max recovery fine-tune steps
        progress_callback: Optional stage lifecycle callback for live CLI updates

    Returns:
        Pipeline summary with results from each stage
    """
    device = _normalize_device(device)

    output_dir = Path(config.paths.output_dir)
    source_checkpoint = Path(checkpoint_path).expanduser().resolve() if checkpoint_path is not None else None

    from forge import __version__

    results: dict[str, Any] = {
        "device": device,
        "config": config.student.variant,
        "execution": {
            "schema": "forge.pipeline-execution.v1",
            "started_at": datetime.now(UTC).isoformat(),
            "git_sha": current_git_sha(),
            "forge_version": __version__,
            "torch_version": str(torch.__version__),
            "python_version": platform.python_version(),
            "requested_stage": stage or "all",
            "device": device,
        },
    }
    config_path = getattr(config, "_forge_config_path", None)
    config_sha256 = getattr(config, "_forge_config_sha256", None)
    if isinstance(config_path, str) and isinstance(config_sha256, str):
        results["config_path"] = config_path
        results["config_sha256"] = config_sha256
    if source_checkpoint is not None:
        results["source_checkpoint"] = str(source_checkpoint)
        results["source_checkpoint_sha256"] = _checkpoint_sha256(source_checkpoint)
    export_student: torch.nn.Module | None = None
    student: torch.nn.Module
    t_start = time.time()
    stage_starts: dict[str, float] = {}

    def begin_stage(name: str, title: str) -> None:
        stage_starts[name] = time.perf_counter()
        if progress_callback is not None:
            progress_callback({"stage": name, "title": title, "status": "started"})

    def finish_stage(name: str, value: object) -> None:
        elapsed = time.perf_counter() - stage_starts[name]
        timings = results.setdefault("stage_timings_seconds", {})
        assert isinstance(timings, dict)
        timings[name] = round(elapsed, 3)
        failed = isinstance(value, dict) and value.get("status") == "failed"
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": name,
                    "status": "failed" if failed else "completed",
                    "elapsed_seconds": elapsed,
                }
            )

    logger.info("=" * 60)
    logger.info("FORGE DISTILLATION PIPELINE")
    logger.info(f"Device: {device}")
    logger.info(f"Student: FORGE-{config.student.variant}")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 60)

    # ─── STAGE 1: Teacher Labels ───
    if stage in (None, "labels") and not skip_labels:
        begin_stage("labels", "Teacher Label Generation")
        logger.info("\n>>> STAGE 1: Teacher Label Generation")
        try:
            from forge.teacher import generate_teacher_labels

            results["labels"] = generate_teacher_labels(
                config,
                device=device,
                max_episodes=(
                    max_label_episodes if max_label_episodes is not None else (10 if device == "cpu" else None)
                ),
            )
            logger.info(f"Labels: {results['labels']['total_episodes']} episodes generated")
        except Exception as e:
            logger.error(f"Stage 1 failed: {e}")
            results["labels"] = {"status": "failed", "error": str(e)}

        finish_stage("labels", results["labels"])
        if stage == "labels" or _contains_failure(results["labels"]):
            return _finalize(results, t_start, output_dir, config)

    # ─── STAGE 2: Student Init + Distillation ───
    if stage in (None, "distill"):
        begin_stage("distill", "Knowledge Distillation")
        logger.info("\n>>> STAGE 2: Knowledge Distillation")
        try:
            from forge.distill import train_forge

            def report_distill_progress(event: dict[str, float | int]) -> None:
                if progress_callback is not None:
                    progress_callback(
                        {
                            "stage": "distill",
                            "status": "progress",
                            **event,
                        }
                    )

            results["distill"] = train_forge(
                config,
                device=device,
                max_steps=max_distill_steps or (100 if device == "cpu" else config.distill.max_steps),
                checkpoint_dir=str(output_dir),
                progress_callback=report_distill_progress,
            )
            final_checkpoint = (output_dir / "checkpoints" / "final.pt").resolve()
            if not final_checkpoint.is_file():
                raise FileNotFoundError(
                    f"Distillation completed without its durable final checkpoint: {final_checkpoint}"
                )
            results["distill"]["checkpoint_path"] = str(final_checkpoint)
            results["distill"]["checkpoint_sha256"] = _checkpoint_sha256(final_checkpoint)
            distill_provenance = results["distill"].get("provenance")
            if isinstance(distill_provenance, dict):
                results["provenance"] = distill_provenance
            logger.info(f"Distillation: loss={results['distill']['final_loss']:.4f}")
        except Exception as e:
            logger.error(f"Stage 2 failed: {e}")
            results["distill"] = {"status": "failed", "error": str(e)}

        finish_stage("distill", results["distill"])
        if stage == "distill" or _contains_failure(results["distill"]):
            return _finalize(results, t_start, output_dir, config)

    # ─── STAGE 3: Compression ───
    if stage in (None, "compress"):
        begin_stage("compress", "Compression (Pruning + Quantization)")
        logger.info("\n>>> STAGE 3: Compression (Pruning + Quantization)")
        try:
            from forge.student import FORGEStudent

            # Load trained student (prefer the canonical final checkpoint).
            preferred_checkpoint = output_dir / "checkpoints" / "final.pt"
            selected_checkpoint = source_checkpoint
            if selected_checkpoint is None:
                selected_checkpoint = (
                    preferred_checkpoint if preferred_checkpoint.is_file() else _latest_checkpoint(output_dir)
                )
            if selected_checkpoint is None and not config.student.allow_mock:
                raise FileNotFoundError(
                    "Compression requires a trained checkpoint. Run the distill stage first or use "
                    "--checkpoint to select one, or use --allow-mock only for an explicit synthetic workflow."
                )
            student = FORGEStudent(config.student, model_dir=config.paths.model_dir)
            loaded_provenance = None
            if selected_checkpoint is not None:
                loaded_provenance = _load_checkpoint_if_available(
                    student,
                    selected_checkpoint,
                    verify_for="compress",
                    allow_mock=config.student.allow_mock,
                )

            if device.startswith("cuda"):
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
            student, student_device = _prepare_student_for_compression(student, device)

            # 3a: Pruning
            from forge.data.teacher_dataset import TeacherLabelDataset
            from forge.prune import _find_transformer_layers, compute_activation_layer_importance, prune_layers

            label_dir = Path(config.paths.data_dir) / "teacher_labels"
            calibration_dataset = TeacherLabelDataset(label_dir)
            if calibration_dataset.labels_provenance != "real" and not config.student.allow_mock:
                raise ValueError(f"Compression calibration labels are not real: {label_dir}")
            calibration_data = [
                {"image": calibration_dataset[index]["image"].to(student_device)}
                for index in range(min(len(calibration_dataset), 8))
            ]
            calibration_dataset.close()
            scores = compute_activation_layer_importance(student, calibration_data)
            pre_prune_layer_count = len(_find_transformer_layers(student))
            pruned_student, removed = prune_layers(student, scores, config.pruning)
            quant_device = student_device

            results["pruning"] = {
                "status": "success",
                "removed_layers": removed,
                "n_removed": len(removed),
                "pre_prune_layer_count": pre_prune_layer_count,
            }

            compressed_path = output_dir / "compressed"
            compressed_path.mkdir(parents=True, exist_ok=True)
            pruning_metadata = {
                "removed_layers": removed,
                "pre_prune_layer_count": pre_prune_layer_count,
                "target_layers": config.pruning.target_layers,
                "calibration_provenance": calibration_dataset.labels_provenance,
            }
            provenance = loaded_provenance or results.get("provenance")
            if not isinstance(provenance, dict):
                provenance = build_provenance(
                    student=pruned_student,
                    config=config,
                    labels=calibration_dataset.labels_provenance,
                )
            results["provenance"] = provenance
            pruned_file = compressed_path / "pruned.pt"
            torch.save(
                {
                    "model_state_dict": pruned_student.state_dict(),
                    "pruning": pruning_metadata,
                    "provenance": provenance,
                },
                pruned_file,
            )
            results["pruning"]["path"] = str(pruned_file)
            results["pruning"]["size_mb"] = pruned_file.stat().st_size / 1e6
            pruned_sha256 = _checkpoint_sha256(pruned_file)
            results["pruning"]["sha256"] = pruned_sha256
            setattr(pruned_student, "_forge_pruning_metadata", pruning_metadata)

            # 3b: Quantization
            from forge.quantize.serialization import pack_state_dict

            quantized = _quantize_student(pruned_student.to(quant_device), config)
            method = config.quant.method.replace("turboquant-", "tq-")
            profile = _create_quant_profile(quantized, config)
            packed_state, packing = pack_state_dict(
                quantized.state_dict(),
                bits=config.quant.bits,
            )
            results["quantization"] = {
                "status": "success",
                "method": config.quant.method,
                "bits": config.quant.bits,
                "device": quant_device,
                "avg_bits": config.quant.bits,
                "estimated_compressed_size_mb": profile.compressed_size_mb,
                "compression_ratio": packing["compression_ratio"],
                "serialization_schema": packing["schema"],
            }

            # Save compressed model
            compressed_file = compressed_path / f"{method}_{config.quant.bits}bit.pt"
            torch.save(
                _packed_compression_payload(
                    packed_state=packed_state,
                    source_checkpoint_sha256=pruned_sha256,
                    config_sha256=config_sha256 if isinstance(config_sha256, str) else None,
                    quantization={
                        **packing,
                        "method": config.quant.method,
                    },
                    pruning=pruning_metadata,
                    provenance=provenance,
                ),
                compressed_file,
            )
            results["quantization"]["compressed_size_mb"] = compressed_file.stat().st_size / 1e6
            export_student = quantized
            results["compression"] = {
                "path": str(compressed_file),
                "sha256": _checkpoint_sha256(compressed_file),
                "status": "success",
            }

            logger.info(
                "Compression: %.1f MB artifact, %s-bit packed state",
                results["quantization"]["compressed_size_mb"],
                config.quant.bits,
            )

        except Exception as e:
            logger.error(f"Stage 3 failed: {e}")
            results["compression"] = {"status": "failed", "error": str(e)}

        finish_stage("compress", results["compression"])
        if stage == "compress" or _contains_failure(results["compression"]):
            return _finalize(results, t_start, output_dir, config)

    # ─── STAGE 4: Export + Validation ───
    if stage in (None, "export"):
        begin_stage("export", "Export")
        logger.info("\n>>> STAGE 4: Export")
        requested_formats = {
            str(fmt).strip().lower() for fmt in getattr(config.export, "formats", ["onnx", "mlx"]) if str(fmt).strip()
        } or {"onnx", "mlx"}
        try:
            from forge.export.mlx_export import export_mlx
            from forge.export.onnx_export import benchmark_onnx_runtime, export_onnx
            from forge.export.tensorrt_export import (
                benchmark_tensorrt_runtime,
                check_tensorrt_available,
                export_tensorrt,
            )
            from forge.student import FORGEStudent

            onnx_path = output_dir / "forge.onnx"
            engine_path = output_dir / "forge.engine"
            _clear_runtime_export_artifacts(onnx_path, engine_path)

            selected_checkpoint = source_checkpoint or _latest_checkpoint(output_dir)
            if selected_checkpoint is None and export_student is None and not config.student.allow_mock:
                raise FileNotFoundError(
                    "Export requires a trained checkpoint. Run the distill stage first or use "
                    "--checkpoint to select one, or use --allow-mock only for an explicit synthetic workflow."
                )
            student = export_student or FORGEStudent(config.student, model_dir=config.paths.model_dir)
            student = student.to("cpu")
            loaded_provenance = None
            if export_student is None:
                loaded_provenance = _load_checkpoint_if_available(
                    student,
                    selected_checkpoint,
                    verify_for="export",
                    allow_mock=config.student.allow_mock,
                )
            if loaded_provenance is not None:
                results["provenance"] = loaded_provenance

            runtime_images, runtime_language_ids, runtime_input_provenance = _load_export_runtime_inputs(
                student,
                config,
            )
            results["export_runtime_inputs"] = _persist_export_runtime_inputs(
                runtime_images,
                runtime_language_ids,
                output_dir=output_dir,
                labels_provenance=runtime_input_provenance,
            )

            # MLX export (on request)
            if "mlx" in requested_formats:
                mlx_dir = output_dir / "mlx"
                export_mlx(student, mlx_dir, config={"variant": config.student.variant})
                results["export_mlx"] = {"path": str(mlx_dir), "status": "success"}
            else:
                results["export_mlx"] = {"status": "skipped", "reason": "mlx not in export.formats"}

            # ONNX export
            onnx_ok = False
            if "onnx" in requested_formats:
                try:
                    onnx_path = export_onnx(student, onnx_path, optimize=False, opset_version=config.export.onnx_opset)
                    results["export_onnx"] = {
                        "path": str(onnx_path),
                        "artifacts_sha256": _artifact_family_sha256(onnx_path),
                        "status": "success",
                    }
                    results["export_onnx_benchmark"] = _require_runtime_success(
                        benchmark_onnx_runtime(
                            onnx_path,
                            device=device,
                            n_warmup=2,
                            n_runs=10,
                            images=runtime_images,
                            language_ids=runtime_language_ids,
                        ),
                        target="ONNX Runtime",
                    )
                    onnx_ok = True
                except Exception as e:
                    _clear_runtime_export_artifacts(onnx_path, engine_path)
                    results["export_onnx"] = {"status": "failed", "error": str(e)}
                    results["export_onnx_benchmark"] = {
                        "status": "skipped",
                        "reason": "ONNX export failed",
                    }
            else:
                results["export_onnx"] = {"status": "skipped", "reason": "onnx not in export.formats"}
                results["export_onnx_benchmark"] = {
                    "status": "skipped",
                    "reason": "onnx not in export.formats",
                }

            # TensorRT (CUDA only, and only when ONNX is available)
            if device.startswith("cuda") and "tensorrt" in requested_formats:
                if onnx_ok:
                    try:
                        if not check_tensorrt_available():
                            results["export_tensorrt"] = {
                                "status": "failed",
                                "error": "TensorRT was requested on CUDA but is not installed",
                            }
                        else:
                            precision = (getattr(config.export, "tensorrt_precision", "fp16") or "fp16").lower()
                            precision = precision if precision in {"fp16", "int8"} else "fp16"
                            workspace_mb = int(getattr(config.export, "tensorrt_workspace_mb", 2048))
                            calibration_path = None
                            if precision == "int8":
                                calibration_path = Path(str(results["export_runtime_inputs"]["path"]))
                            export_tensorrt(
                                str(onnx_path),
                                engine_path,
                                precision=precision,
                                workspace_mb=workspace_mb,
                                calibration_data=str(calibration_path) if calibration_path is not None else None,
                                device=device,
                            )
                            results["export_tensorrt"] = {
                                "path": str(engine_path),
                                "sha256": _checkpoint_sha256(engine_path),
                                "status": "success",
                                "precision": precision,
                                "calibration_samples": len(runtime_images) if calibration_path is not None else None,
                                "calibration_provenance": (
                                    runtime_input_provenance if calibration_path is not None else None
                                ),
                            }
                            if results["export_tensorrt"].get("status") == "success":
                                results["export_tensorrt_benchmark"] = _require_runtime_success(
                                    benchmark_tensorrt_runtime(
                                        engine_path,
                                        n_warmup=2,
                                        n_runs=10,
                                        images=runtime_images,
                                        language_ids=runtime_language_ids,
                                        precision=precision,
                                        device=device,
                                    ),
                                    target="TensorRT",
                                )
                    except Exception as e:
                        engine_path.unlink(missing_ok=True)
                        results["export_tensorrt"] = {"status": "failed", "error": str(e)}
                else:
                    results["export_tensorrt"] = {
                        "status": "failed",
                        "error": "TensorRT was requested but the required ONNX export failed",
                    }
            elif device.startswith("cuda"):
                results["export_tensorrt"] = {"status": "skipped", "reason": "tensorrt not in export.formats"}
            else:
                results["export_tensorrt"] = {"status": "skipped", "reason": "not a CUDA device"}

        except Exception as e:
            logger.error(f"Stage 4 export failed: {e}")
            results["export"] = {"status": "failed", "error": str(e)}

        export_result = _export_stage_result(results, requested_formats)
        results["export"] = export_result
        finish_stage("export", export_result)
        if _contains_failure(export_result):
            return _finalize(results, t_start, output_dir, config)

    if stage in (None, "validate"):
        begin_stage("validate", "Validation")
        logger.info("\n>>> STAGE 4b: Validation")
        try:
            from forge.student import FORGEStudent
            from forge.validate import run_full_validation

            selected_checkpoint = source_checkpoint or _latest_checkpoint(output_dir)
            if selected_checkpoint is None and export_student is None and not config.student.allow_mock:
                raise FileNotFoundError(
                    "Validation requires a trained checkpoint. Run the distill stage first or use "
                    "--checkpoint to select one, or use --allow-mock only for an explicit synthetic workflow."
                )
            student = export_student or FORGEStudent(config.student, model_dir=config.paths.model_dir)
            loaded_provenance = None
            if export_student is None:
                loaded_provenance = _load_checkpoint_if_available(
                    student,
                    selected_checkpoint,
                    verify_for="eval",
                    allow_mock=config.student.allow_mock,
                )
            if loaded_provenance is not None:
                results["provenance"] = loaded_provenance
            student = student.to(device)

            validation = run_full_validation(
                student,
                device=device,
                stability_duration=5,
                allow_warnings=True,
            )
            validation_status = str(validation.overall_status).lower()
            results["validation"] = {
                "status": ("failed" if validation_status in {"fail", "failed", "error"} else "success"),
                "overall": validation.overall_status,
                "latency_ms": validation.benchmark.mean_latency_ms if validation.benchmark else None,
                "throughput_fps": validation.benchmark.throughput_fps if validation.benchmark else None,
                "model_size_mb": validation.benchmark.model_size_mb if validation.benchmark else None,
            }

            logger.info(f"Validation: {validation.overall_status}")

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            results["validation"] = {"status": "failed", "error": str(e)}

        finish_stage("validate", results["validation"])

    return _finalize(results, t_start, output_dir, config)


def _contains_failure(value: object) -> bool:
    """Return whether a nested pipeline result contains an explicit failure."""
    if isinstance(value, dict):
        if value.get("status") == "failed":
            return True
        return any(_contains_failure(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_failure(item) for item in value)
    return False


def _require_runtime_success(result: dict[str, Any], *, target: str) -> dict[str, Any]:
    """Turn an unavailable selected runtime into an honest pipeline failure."""
    if result.get("status") == "success":
        return result
    reason = str(result.get("error") or result.get("reason") or f"{target} execution did not succeed")
    return {**result, "status": "failed", "error": reason}


def _export_stage_result(results: dict[str, Any], requested_formats: set[str]) -> dict[str, str]:
    """Require every requested export to produce a successful artifact."""
    existing_stage = results.get("export")
    if isinstance(existing_stage, dict) and existing_stage.get("status") == "failed":
        return {"status": "failed", "error": str(existing_stage.get("error", "Export failed"))}
    requested_failures = [
        requested_format
        for requested_format in sorted(requested_formats)
        if not isinstance(results.get(f"export_{requested_format}"), dict)
        or results[f"export_{requested_format}"].get("status") != "success"
    ]
    explicit_failures = [
        value
        for key, value in results.items()
        if key.startswith("export_") and isinstance(value, dict) and value.get("status") == "failed"
    ]
    if not requested_failures and not explicit_failures:
        return {"status": "completed"}
    if requested_failures:
        return {
            "status": "failed",
            "error": "Requested export format(s) did not produce successful artifacts: "
            + ", ".join(requested_failures),
        }
    first_error = explicit_failures[0].get("error", "An export operation failed")
    return {"status": "failed", "error": str(first_error)}


def _clear_runtime_export_artifacts(onnx_path: Path, engine_path: Path) -> None:
    """Remove exact runtime outputs that could be mistaken for this run."""
    if onnx_path.parent.is_dir():
        for artifact in onnx_path.parent.glob(f"{onnx_path.name}*"):
            if artifact.is_file():
                artifact.unlink()
    engine_path.unlink(missing_ok=True)


def _finalize(
    results: dict[str, Any],
    t_start: float,
    output_dir: Path,
    config: ForgeConfig,
) -> dict[str, Any]:
    """Finalize pipeline results."""
    elapsed = time.time() - t_start
    results["total_time_seconds"] = elapsed
    results["status"] = "failed" if _contains_failure(results) else "completed"
    if not isinstance(results.get("provenance"), dict):
        results["provenance"] = build_provenance(
            config=config,
            vision="mock",
            language="mock",
            labels="mock",
        )

    # Failed preflight/training must not create artifacts.
    summary_path = (output_dir / "pipeline_summary.json").resolve()
    results["pipeline_summary_path"] = str(summary_path)
    if results["status"] == "completed":
        write_json_artifact(summary_path, results)

    logger.info("\n" + "=" * 60)
    logger.info("FORGE PIPELINE COMPLETE")
    logger.info(f"Total time: {elapsed:.1f}s")
    logger.info(f"Summary: {summary_path}")
    logger.info("=" * 60)

    return results
