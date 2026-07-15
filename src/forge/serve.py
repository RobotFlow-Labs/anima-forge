"""FORGE Serve — maintained FastAPI inference endpoint.

Serves a provenance-verified trained FORGE checkpoint over HTTP.

Usage:
    forge serve --checkpoint outputs/checkpoints/final.pt --port 8000 --device cuda
    curl -X POST http://localhost:8000/predict -F "image=@photo.jpg" -F "instruction=pick up the block"
"""

from __future__ import annotations

import io
import logging
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from fastapi import UploadFile

from forge.checkpoint_compat import (
    CheckpointLoadReport,
    apply_checkpoint_structure,
    extract_checkpoint_state_dict,
    load_checkpoint_payload,
    load_model_weights_with_compatibility,
    summarize_checkpoint_report,
)

logger = logging.getLogger(__name__)


def _checkpoint_runtime_metadata(checkpoint: Mapping[str, Any]) -> tuple[str, int, int]:
    """Read the public serving contract from mandatory checkpoint architecture metadata."""
    saved = checkpoint.get("student_config")
    if not isinstance(saved, Mapping):
        raise ValueError("Verified serving checkpoint is missing student_config architecture metadata")

    variant = saved.get("variant")
    if not isinstance(variant, str) or not variant.strip():
        raise ValueError("Checkpoint student_config.variant must be a non-empty string")

    dimensions: dict[str, int] = {}
    for field in ("action_horizon", "action_dim"):
        value = saved.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"Checkpoint student_config.{field} must be a positive integer")
        dimensions[field] = value
    return variant.strip(), dimensions["action_horizon"], dimensions["action_dim"]


def _validated_action_batch(
    output: Any,
    *,
    batch_size: int,
    action_horizon: int,
    action_dim: int,
):
    """Return finite actions with the exact public ``(B,H,D)`` contract."""
    import torch

    if not isinstance(output, Mapping) or "actions" not in output:
        raise RuntimeError("FORGE inference returned no actions tensor")
    actions = output["actions"]
    if not isinstance(actions, torch.Tensor):
        raise RuntimeError("FORGE inference actions must be a tensor")
    if actions.ndim == 2 and action_horizon == 1 and tuple(actions.shape) == (batch_size, action_dim):
        actions = actions.unsqueeze(1)
    expected = (batch_size, action_horizon, action_dim)
    if tuple(actions.shape) != expected:
        raise RuntimeError(f"FORGE inference returned action shape {tuple(actions.shape)}; expected {expected}")
    if not torch.isfinite(actions).all():
        raise RuntimeError("FORGE inference returned non-finite actions")
    return actions.detach().float().cpu()


def _tokenize_instructions(
    student: Any,
    instructions: list[str],
    *,
    device: str,
    allow_mock: bool,
) -> Any:
    """Tokenize a request batch without substituting random language inputs."""
    if not instructions or any(not instruction.strip() for instruction in instructions):
        raise ValueError("Every inference request requires a non-empty instruction")

    import torch

    tokenizer = getattr(student, "tokenizer", None)
    if tokenizer is None:
        provenance = getattr(student, "component_provenance", {})
        if allow_mock and provenance.get("language") == "mock":
            return torch.zeros((len(instructions), 1), dtype=torch.long, device=device)
        raise RuntimeError(
            "The loaded language backbone has no tokenizer. Restore the mandatory local tokenizer "
            "with `forge models fetch --all-students`; FORGE will not invent language tokens."
        )

    tokens = tokenizer(
        instructions,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64,
    )
    language_ids = tokens.get("input_ids") if hasattr(tokens, "get") else None
    if not isinstance(language_ids, torch.Tensor) or language_ids.ndim != 2:
        raise RuntimeError("The mandatory tokenizer did not return a two-dimensional input_ids tensor")
    if language_ids.shape[0] != len(instructions):
        raise RuntimeError(
            f"Tokenizer returned batch {language_ids.shape[0]} for {len(instructions)} inference instructions"
        )
    return language_ids.to(device)


def _resolve_runtime_device(requested_device: str) -> str:
    """Use the same strict indexed-device contract as every public CLI command."""
    from forge.cli_commands.shared import resolve_runtime_device

    return resolve_runtime_device(requested_device, command="serve", default="auto", strict=True)


def _load_checkpoint_payload(
    checkpoint: str,
    map_location: str,
    *,
    allow_mock: bool = False,
) -> dict[str, Any] | None:
    """Load a checkpoint and enforce serve-time provenance policy."""
    return load_checkpoint_payload(
        checkpoint,
        map_location=map_location,
        verify_provenance_for="serve",
        allow_mock=allow_mock,
    )


def create_app(
    checkpoint: str,
    model_dir: str | None = None,
    device: str = "cuda",
    allow_mock: bool = False,
):
    """Create the FastAPI application with a loaded FORGE model."""
    try:
        from fastapi import FastAPI, File, Form, HTTPException
        from fastapi.responses import JSONResponse
    except ImportError:
        raise ImportError("FastAPI not installed. Run: uv add fastapi uvicorn python-multipart")

    import torch
    from PIL import Image
    from torchvision import transforms  # type: ignore[import-untyped]

    from forge import __version__
    from forge.config import StudentConfig, apply_checkpoint_student_config, apply_student_variant
    from forge.provenance import require_real_provenance
    from forge.student import FORGEStudent

    if not isinstance(checkpoint, str) or not checkpoint.strip():
        raise ValueError("Serving requires a verified trained checkpoint")
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    device = _resolve_runtime_device(device)
    loaded_checkpoint = _load_checkpoint_payload(
        str(checkpoint_path),
        map_location=device,
        allow_mock=allow_mock,
    )
    if loaded_checkpoint is None:
        raise ValueError(f"Checkpoint payload is unreadable: {checkpoint}")
    provenance = require_real_provenance(
        loaded_checkpoint.get("provenance"),
        action="serve",
        allow_mock=allow_mock,
    )
    if provenance is None:
        raise ValueError("Serving requires checkpoint provenance; legacy or unverified checkpoints are not accepted")

    variant, action_horizon, action_dim = _checkpoint_runtime_metadata(loaded_checkpoint)
    config = apply_student_variant(StudentConfig(), variant)
    apply_checkpoint_student_config(config, loaded_checkpoint)
    config.allow_mock = bool(allow_mock)
    model_name = f"FORGE-{config.variant}"

    app = FastAPI(
        title="FORGE — VLA Inference API",
        description=f"Vision-Language-Action inference with {model_name}",
        version=__version__,
    )

    model_dir = model_dir or os.environ.get("FORGE_MODEL_DIR", "")
    logger.info("Loading %s model...", model_name)
    student = FORGEStudent(config, model_dir=model_dir)

    apply_checkpoint_structure(student, loaded_checkpoint)
    state_dict, extracted_key = extract_checkpoint_state_dict(loaded_checkpoint)
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError(f"Checkpoint missing/invalid state dict payload: {checkpoint}")

    report = CheckpointLoadReport(source=str(checkpoint_path), extracted_key=extracted_key)
    report.raw_key_count = len(state_dict)
    try:
        missing, report = load_model_weights_with_compatibility(
            student,
            state_dict,
            context=f"serve:{checkpoint_path}",
            minimum_coverage=0.8,
        )
    except RuntimeError:
        logger.exception("Failed to load checkpoint: %s", checkpoint_path)
        raise
    for warning in report.warnings:
        logger.warning("%s", warning)
    logger.info(summarize_checkpoint_report("serve", report))
    logger.info("Loaded checkpoint: %s", checkpoint_path)

    if missing.unexpected_keys:
        logger.warning("Unexpected checkpoint keys ignored: %s", ", ".join(missing.unexpected_keys[:8]))
    if missing.missing_keys:
        logger.warning("Model keys missing in checkpoint: %s", ", ".join(missing.missing_keys[:8]))

    student = student.to(device)
    student.eval()

    preprocess: Any = transforms.Compose(
        [
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    app.state.student = student
    app.state.checkpoint = str(checkpoint_path)
    logger.info("%s ready on %s (%.0fM params)", model_name, device, student.total_params / 1e6)

    @app.get("/health")
    async def health():
        mem = {}
        if device.startswith("cuda") and torch.cuda.is_available():
            mem = {
                "gpu_memory_allocated_GB": round(torch.cuda.memory_allocated(device) / 1e9, 2),
                "gpu_memory_reserved_GB": round(torch.cuda.memory_reserved(device) / 1e9, 2),
            }
        return {
            "status": "healthy",
            "version": __version__,
            "model": model_name,
            "variant": config.variant,
            "action_horizon": action_horizon,
            "action_dim": action_dim,
            "checkpoint": checkpoint_path.name,
            "device": device,
            "params_M": round(student.total_params / 1e6, 1),
            "provenance": student.component_provenance,
            **mem,
        }

    @app.post("/predict")
    async def predict(
        image: UploadFile = File(...),
        instruction: str = Form(...),
    ):
        t0 = time.time()
        if not instruction.strip():
            raise HTTPException(status_code=422, detail="A non-empty instruction is required")
        instruction = instruction.strip()

        # Process image
        img_bytes = await image.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_tensor = preprocess(img).unsqueeze(0).to(device)

        lang_ids = _tokenize_instructions(
            student,
            [instruction],
            device=device,
            allow_mock=config.allow_mock,
        )

        # Inference
        with torch.no_grad():
            if device.startswith("cuda"):
                torch.cuda.synchronize(device)
            out = student(img_tensor, language_ids=lang_ids)
            if device.startswith("cuda"):
                torch.cuda.synchronize(device)

        latency = (time.time() - t0) * 1000
        actions = _validated_action_batch(
            out,
            batch_size=1,
            action_horizon=action_horizon,
            action_dim=action_dim,
        ).tolist()[0]

        return JSONResponse(
            {
                "actions": actions,
                "action_horizon": action_horizon,
                "action_dim": action_dim,
                "instruction": instruction,
                "latency_ms": round(latency, 1),
                "model": model_name,
                "version": __version__,
                "provenance": student.component_provenance,
            }
        )

    @app.post("/batch_predict")
    async def batch_predict(
        images: list[UploadFile] = File(...),
        instruction: str = Form(...),
    ):
        t0 = time.time()
        if not images:
            return JSONResponse({"error": "At least one image is required"}, status_code=422)
        if not instruction.strip():
            raise HTTPException(status_code=422, detail="A non-empty instruction is required")
        instruction = instruction.strip()
        batch_tensors = []

        for img_file in images:
            img_bytes = await img_file.read()
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            batch_tensors.append(preprocess(img))

        batch = torch.stack(batch_tensors).to(device)
        lang_ids = _tokenize_instructions(
            student,
            [instruction] * len(images),
            device=device,
            allow_mock=config.allow_mock,
        )

        with torch.no_grad():
            if device.startswith("cuda"):
                torch.cuda.synchronize(device)
            out = student(batch, language_ids=lang_ids)
            if device.startswith("cuda"):
                torch.cuda.synchronize(device)

        latency = (time.time() - t0) * 1000
        actions = _validated_action_batch(
            out,
            batch_size=len(images),
            action_horizon=action_horizon,
            action_dim=action_dim,
        ).tolist()

        return JSONResponse(
            {
                "actions": actions,
                "batch_size": len(images),
                "action_horizon": action_horizon,
                "action_dim": action_dim,
                "instruction": instruction,
                "latency_ms": round(latency, 1),
                "per_sample_ms": round(latency / len(images), 1),
                "model": model_name,
                "version": __version__,
                "provenance": student.component_provenance,
            }
        )

    return app


def start_server(
    checkpoint: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    model_dir: str | None = None,
    device: str = "cuda",
    allow_mock: bool = False,
):
    """Start the FastAPI server."""
    try:
        import uvicorn
    except ImportError:
        raise ImportError("uvicorn not installed. Run: uv add uvicorn")

    app = create_app(
        model_dir=model_dir,
        checkpoint=checkpoint,
        device=device,
        allow_mock=allow_mock,
    )
    uvicorn.run(app, host=host, port=port)
