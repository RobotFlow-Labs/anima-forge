"""OpenVLA teacher adapter — wraps the original teacher.py logic."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from forge.hf_compat import configure_transformers_module_cache
from forge.openvla_loader import load_image_text_model
from forge.processor_compat import install_legacy_tokenization_exports
from forge.teachers.base import ActionChunk, TeacherAdapter, TeacherInfo

logger = logging.getLogger(__name__)

ADAPTER_NAME = "openvla-7b"


def _warn_openvla_timm_warning() -> str | None:
    """Build an actionable OpenVLA + timm compatibility warning."""
    try:
        import timm

        version = getattr(timm, "__version__", "unknown")
    except Exception:
        return "OpenVLA load: timm import failed; verify timm is installed and reachable."

    if str(version).startswith("0.") or str(version).startswith("0.9"):
        return None

    return (
        "OpenVLA compatibility warning: OpenVLA adapters commonly require timm<1.0.0. "
        f"Detected timm={version}. If teacher loading fails with TIMM constraint errors, use a matching "
        "model pack or align timm with the OpenVLA checkpoint/runtime contract."
    )


class OpenVLAAdapter(TeacherAdapter):
    """Adapter for OpenVLA 7B (token-AR architecture).

    OpenVLA uses autoregressive token prediction for actions.
    Actions are decoded from discrete tokens back to continuous values.
    """

    def __init__(self, unnorm_key: str | None = None):
        self._model: Any | None = None
        self._processor: Any | None = None
        self._device = "cpu"
        self._dtype = torch.float32
        self._unnorm_key = unnorm_key or os.environ.get("FORGE_OPENVLA_UNNORM_KEY", "bridge_orig")

    def load(self, model_path: Path, device: str = "cpu", dtype: torch.dtype = torch.float32) -> None:
        self._device = device
        self._dtype = dtype

        if not model_path.exists():
            raise FileNotFoundError(f"OpenVLA weights not found at {model_path}")

        configure_transformers_module_cache(model_path)

        try:
            install_legacy_tokenization_exports()

            from transformers import AutoProcessor

            self._model = load_image_text_model(
                model_path,
                dtype=dtype,
                # Accelerate's string device map probes NVML while dispatching.
                # That makes an otherwise working CUDA runtime fail on hosts
                # where NVML is unavailable or temporarily mismatched. Load the
                # local checkpoint first, then use PyTorch's normal device move.
                device_map=None,
                local_files_only=True,
            )
            if device == "cpu":
                self._model = self._model.to(dtype=torch.float32)
            else:
                self._model = self._model.to(device=device)
            self._model.eval()

            self._processor = AutoProcessor.from_pretrained(
                str(model_path),
                trust_remote_code=True,
                local_files_only=True,
                use_fast=False,
            )

            logger.info(f"OpenVLA loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load OpenVLA: {e}")
            hint = _warn_openvla_timm_warning()
            if hint:
                logger.warning(hint)
            raise

    def predict(
        self,
        image: np.ndarray,
        instruction: str,
        proprioception: np.ndarray | None = None,
    ) -> ActionChunk:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        processor = self._processor
        if processor is None:
            raise RuntimeError("Processor not loaded. Call load() first.")

        from PIL import Image

        pil_image = Image.fromarray(image)

        prompt = f"In: What action should the robot take to {instruction}?\nOut:"
        inputs = processor(prompt, pil_image, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        if not hasattr(self._model, "predict_action"):
            raise RuntimeError("Loaded OpenVLA checkpoint exposes no predict_action decoder")
        with torch.no_grad():
            actions = np.asarray(
                self._model.predict_action(
                    **inputs,
                    unnorm_key=self._unnorm_key,
                    do_sample=False,
                ),
                dtype=np.float32,
            ).reshape(-1)
        if actions.shape != (7,) or not np.isfinite(actions).all():
            raise ValueError(f"OpenVLA returned invalid real action output {actions.shape}")

        return ActionChunk(
            actions=actions.reshape(1, -1),
            action_mean=actions.reshape(1, -1),
            action_std=np.ones_like(actions.reshape(1, -1)) * 0.05,
            confidence=np.ones_like(actions.reshape(1, -1)) * 0.9,
            metadata={
                "teacher": "openvla-7b",
                "architecture": "token-ar",
                "inference": "real",
                "unnorm_key": self._unnorm_key,
                "uncertainty": "deterministic-single-pass",
            },
        )

    def extract_features(self, image: np.ndarray, instruction: str) -> dict[str, np.ndarray]:
        if self._model is None:
            raise RuntimeError("Model not loaded.")
        processor = self._processor
        if processor is None:
            raise RuntimeError("Processor not loaded.")

        from PIL import Image

        pil_image = Image.fromarray(image)
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"
        inputs = processor(prompt, pil_image, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs, output_hidden_states=True)

        features = {}
        if hasattr(outputs, "hidden_states") and outputs.hidden_states:
            features["hidden_states"] = outputs.hidden_states[-1].cpu().numpy()

        return features

    def get_action_space(self) -> dict:
        return {
            "dim": 7,
            "min": np.array([-1.0] * 7),
            "max": np.array([1.0] * 7),
            "names": ["x", "y", "z", "rx", "ry", "rz", "gripper"],
        }

    def info(self) -> TeacherInfo:
        return TeacherInfo(
            name="openvla-7b",
            architecture="token-ar",
            param_count=7.6,
            action_dim=7,
            action_horizon=1,
            vision_encoder="siglip-so400m",
            language_model="llama-2-7b",
            supports_chunking=False,
            supports_features=True,
        )

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def unload(self) -> None:
        del self._model
        del self._processor
        self._model = None
        self._processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


ADAPTER_CLASS = OpenVLAAdapter
