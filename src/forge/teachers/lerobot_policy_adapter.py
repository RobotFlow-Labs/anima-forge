"""Shared local-only adapter for official LeRobot teacher policies.

Model-specific modules select the official policy class and declare the action
contract. Preprocessing, normalization, inference, and postprocessing remain
owned by LeRobot and the checkpoint; FORGE never substitutes generated actions.
"""

from __future__ import annotations

import importlib
import inspect
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch

from forge.teachers.base import ActionChunk, TeacherAdapter, TeacherInfo

logger = logging.getLogger(__name__)


def _feature_is_visual(feature: object) -> bool:
    feature_type = feature.get("type") if isinstance(feature, Mapping) else getattr(feature, "type", None)
    value = getattr(feature_type, "value", feature_type)
    return str(value).upper().endswith("VISUAL")


def _feature_width(feature: object | None) -> int:
    shape = feature.get("shape") if isinstance(feature, Mapping) else getattr(feature, "shape", None)
    if not shape:
        return 0
    return int(shape[0])


class LeRobotPolicyAdapter(TeacherAdapter):
    """Normalize one local LeRobot ``PreTrainedPolicy`` to ``ActionChunk``."""

    policy_module: ClassVar[str]
    policy_class_name: ClassVar[str]
    teacher_info: ClassVar[TeacherInfo]
    action_names: ClassVar[tuple[str, ...]] = (
        "x",
        "y",
        "z",
        "roll",
        "pitch",
        "yaw",
        "gripper",
    )

    def __init__(self) -> None:
        self._policy: Any | None = None
        self._preprocess: Any | None = None
        self._postprocess: Any | None = None
        self._device = "cpu"
        self._dtype = torch.float32
        self._model_path: Path | None = None

    def load(self, model_path: Path, device: str = "cpu", dtype: torch.dtype = torch.float32) -> None:
        model_path = Path(model_path)
        if not model_path.is_dir():
            raise FileNotFoundError(
                f"{self.teacher_info.name} checkpoint directory not found at {model_path}. "
                "Fetch the selected teacher checkpoint into FORGE_MODEL_DIR first."
            )
        if not (model_path / "config.json").is_file():
            raise FileNotFoundError(f"{self.teacher_info.name} checkpoint is missing {model_path / 'config.json'}")
        weight_files = list(model_path.glob("*.safetensors")) + list(model_path.glob("*.safetensors.index.json"))
        if not weight_files:
            raise FileNotFoundError(
                f"{self.teacher_info.name} checkpoint at {model_path} contains no safetensors weights."
            )

        try:
            policy_module = importlib.import_module(self.policy_module)
            policy_class = getattr(policy_module, self.policy_class_name)
            factory_module = importlib.import_module("lerobot.policies.factory")
            processor_factory = getattr(factory_module, "make_pre_post_processors")
        except (ImportError, AttributeError) as exc:
            try:
                self._preflight_companions(model_path)
            except Exception as companion_exc:
                raise RuntimeError(
                    f"Failed to validate {self.teacher_info.name} companion checkpoints: {companion_exc}"
                ) from companion_exc
            raise RuntimeError(
                f"{self.teacher_info.name} requires the LeRobot 2026 teacher runtime included with FORGE. "
                "Reinstall FORGE, then retry."
            ) from exc

        try:
            policy = self._load_policy(policy_class, model_path, device)
            policy = policy.to(device)
            policy.eval()
            preprocess, postprocess = self._make_processors(processor_factory, policy, model_path, device)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load real {self.teacher_info.name} policy and processors from {model_path}: {exc}"
            ) from exc

        self._policy = policy
        self._preprocess = preprocess
        self._postprocess = postprocess
        self._device = device
        self._dtype = dtype
        self._model_path = model_path.resolve()
        logger.info("Loaded real %s teacher from %s", self.teacher_info.name, model_path)

    def _preflight_companions(self, model_path: Path) -> None:
        """Validate and bind required local sidecars before loading the runtime."""

    def _load_policy(self, policy_class: Any, model_path: Path, device: str) -> Any:
        """Load the official policy from local files; subclasses may bind local sidecars."""
        return policy_class.from_pretrained(str(model_path), local_files_only=True)

    def _make_processors(self, factory: Any, policy: Any, model_path: Path, device: str) -> tuple[Any, Any]:
        parameters = inspect.signature(factory).parameters
        accepts_kwargs = any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())
        kwargs: dict[str, Any] = {}
        if "pretrained_path" in parameters:
            kwargs["pretrained_path"] = str(model_path)
        elif "pretrained_name_or_path" in parameters:
            kwargs["pretrained_name_or_path"] = str(model_path)
        if "preprocessor_overrides" in parameters or accepts_kwargs:
            kwargs["preprocessor_overrides"] = self._processor_overrides(device)
        processors = factory(policy.config, **kwargs)
        if not isinstance(processors, tuple) or len(processors) != 2:
            raise TypeError("make_pre_post_processors must return (preprocessor, postprocessor)")
        if not all(callable(processor) for processor in processors):
            raise TypeError("LeRobot checkpoint processors must both be callable")
        return processors

    def _processor_overrides(self, device: str) -> dict[str, dict[str, Any]]:
        return {"device_processor": {"device": device}}

    def predict(
        self,
        image: np.ndarray,
        instruction: str,
        proprioception: np.ndarray | None = None,
    ) -> ActionChunk:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        policy = self._policy
        preprocess = self._preprocess
        postprocess = self._postprocess
        if policy is None or preprocess is None or postprocess is None:
            raise RuntimeError("Model runtime is incomplete. Call load() again.")
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError(f"image must have shape (H, W, 3) and dtype uint8, got {image.shape} {image.dtype}")
        if not instruction.strip():
            raise ValueError("instruction must be non-empty")

        raw_batch = self._build_raw_batch(image, instruction, proprioception)
        processed = preprocess(raw_batch)
        if not isinstance(processed, dict):
            raise TypeError(f"LeRobot preprocessor returned {type(processed).__name__}, expected dict")
        processed = self._move_tensors(processed)

        with torch.inference_mode():
            if hasattr(policy, "predict_action_chunk"):
                prediction = policy.predict_action_chunk(processed)
            elif hasattr(policy, "select_action"):
                prediction = policy.select_action(processed)
            else:
                raise RuntimeError(f"{self.teacher_info.name} policy exposes no action inference method")
            prediction = postprocess(prediction)

        actions = self._canonical_actions(prediction)
        zeros = np.zeros_like(actions, dtype=np.float32)
        ones = np.ones_like(actions, dtype=np.float32)
        return ActionChunk(
            actions=actions,
            action_mean=actions.copy(),
            action_std=zeros,
            confidence=ones,
            metadata={
                "teacher": self.teacher_info.name,
                "architecture": self.teacher_info.architecture,
                "checkpoint": str(self._model_path),
                "inference": "real",
                "uncertainty": "deterministic-single-pass",
            },
        )

    def _build_raw_batch(
        self,
        image: np.ndarray,
        instruction: str,
        proprioception: np.ndarray | None,
    ) -> dict[str, Any]:
        policy = self._policy
        if policy is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        config = policy.config
        input_features = dict(getattr(config, "input_features", {}) or {})
        image_keys: list[str] = []
        image_features = getattr(config, "image_features", None)
        if isinstance(image_features, dict):
            image_keys.extend(str(key) for key in image_features)
        image_keys.extend(str(key) for key, feature in input_features.items() if _feature_is_visual(feature))
        image_keys.extend(str(key) for key in (getattr(config, "image_keys", None) or []))
        image_keys = list(dict.fromkeys(image_keys))
        if not image_keys:
            raise RuntimeError(f"{self.teacher_info.name} checkpoint declares no visual input feature")

        image_tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float().div_(255.0)
        raw: dict[str, Any] = {key: image_tensor.clone() for key in image_keys}

        state_feature = input_features.get("observation.state")
        state_width = _feature_width(state_feature)
        if state_width:
            state = np.zeros(state_width, dtype=np.float32)
            if proprioception is not None:
                source = np.asarray(proprioception, dtype=np.float32).reshape(-1)
                if not np.isfinite(source).all():
                    raise ValueError(f"{self.teacher_info.name} proprioception contains non-finite values")
                state[: min(state_width, source.size)] = source[:state_width]
            raw["observation.state"] = torch.from_numpy(state)
        raw["task"] = instruction.strip()
        return raw

    def _move_tensors(self, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.to(self._device)
        if isinstance(value, dict):
            return {key: self._move_tensors(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._move_tensors(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._move_tensors(item) for item in value)
        return value

    def _canonical_actions(self, prediction: object) -> np.ndarray:
        if isinstance(prediction, torch.Tensor):
            array = prediction.detach().float().cpu().numpy()
        else:
            array = np.asarray(prediction, dtype=np.float32)
        if array.ndim == 3:
            if array.shape[0] != 1:
                raise ValueError(f"{self.teacher_info.name} returned batch size {array.shape[0]}, expected 1")
            array = array[0]
        if array.ndim == 1:
            array = array[None, :]
        if array.ndim != 2 or not array.size:
            raise ValueError(f"{self.teacher_info.name} returned invalid action shape {array.shape}")
        expected_horizon = self.teacher_info.action_horizon
        expected_dim = self.teacher_info.action_dim
        if array.shape != (expected_horizon, expected_dim):
            raise ValueError(
                f"{self.teacher_info.name} returned action shape {array.shape}, "
                f"expected {(expected_horizon, expected_dim)}"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"{self.teacher_info.name} returned non-finite real actions")
        return np.ascontiguousarray(array, dtype=np.float32)

    def extract_features(self, image: np.ndarray, instruction: str) -> dict[str, np.ndarray]:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        # No stable public hidden-state API exists for these policies. An empty
        # mapping is truthful; random stand-ins would corrupt distillation data.
        return {}

    def get_action_space(self) -> dict:
        dim = self.teacher_info.action_dim
        names = list(self.action_names[:dim])
        if len(names) < dim:
            names.extend(f"action_{index}" for index in range(len(names), dim))
        return {
            "dim": dim,
            "min": np.full(dim, -np.inf, dtype=np.float32),
            "max": np.full(dim, np.inf, dtype=np.float32),
            "names": names,
        }

    def info(self) -> TeacherInfo:
        return self.teacher_info

    @property
    def is_loaded(self) -> bool:
        return self._policy is not None and self._preprocess is not None and self._postprocess is not None

    def unload(self) -> None:
        self._policy = None
        self._preprocess = None
        self._postprocess = None
        self._model_path = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


__all__ = ["LeRobotPolicyAdapter"]
