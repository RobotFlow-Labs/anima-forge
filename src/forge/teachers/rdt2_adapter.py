"""Real RDT2-FM teacher adapter using the released VQ backbone and normalizer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from forge.teachers.base import ActionChunk, TeacherAdapter, TeacherInfo

ADAPTER_NAME = "rdt2-fm"
ACTION_DIM = 20
ACTION_HORIZON = 24
NORMALIZER_FILENAME = "rdt2-umi-normalizer.pt"
VQ_DIRECTORY = "robotics-diffusion-transformer--RDT2-VQ"


def _load_rdt2_normalizer(
    normalizer_path: Path,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load the official RDT2 action transform with a narrow pickle allowlist."""
    parameter_dict = torch.nn.ParameterDict
    with torch.serialization.safe_globals([parameter_dict]):
        normalizer = torch.load(normalizer_path, map_location="cpu", weights_only=True)
    if not isinstance(normalizer, parameter_dict):
        raise ValueError("RDT2 normalizer must contain a ParameterDict")

    def entry(name: str) -> tuple[torch.Tensor, torch.Tensor]:
        values = normalizer.get(name)
        if not isinstance(values, parameter_dict):
            raise ValueError(f"RDT2 normalizer is missing its {name} ParameterDict")
        scale = values.get("scale")
        offset = values.get("offset")
        if not isinstance(scale, torch.Tensor) or not isinstance(offset, torch.Tensor):
            raise ValueError(f"RDT2 {name} normalizer must contain tensor scale and offset")
        scale = scale.detach().float().cpu()
        offset = offset.detach().float().cpu()
        if scale.shape != (ACTION_DIM,) or offset.shape != (ACTION_DIM,):
            raise ValueError(f"RDT2 {name} normalizer has invalid scale or offset dimensions")
        if not torch.isfinite(scale).all() or not torch.isfinite(offset).all() or torch.any(scale == 0):
            raise ValueError(f"RDT2 {name} normalizer contains invalid values")
        return scale, offset

    return entry("action")


def _load_action_normalizer(normalizer_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward-compatible action-only view of the official normalizer."""
    return _load_rdt2_normalizer(normalizer_path)


class _RDT2Runtime:
    """Local-only orchestration around the official RDT2 action expert."""

    def __init__(
        self,
        action_expert_path: Path,
        vq_path: Path,
        normalizer_path: Path,
        device: str,
        dtype: torch.dtype,
    ) -> None:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        from forge.vendor.rdt2 import RDTRunner

        config = json.loads((action_expert_path / "config.json").read_text(encoding="utf-8"))
        if (config.get("pred_horizon"), config.get("action_dim"), config.get("state_dim")) != (24, 20, 20):
            raise ValueError("RDT2-FM checkpoint config must declare horizon=24 and action/state dim=20")
        self.selected_layers = list(config["selected_layers"])
        self.device = device
        self.dtype = dtype

        self.processor: Any = AutoProcessor.from_pretrained(
            str(vq_path),
            local_files_only=True,
            padding_side="left",
        )
        vlm_model: Any = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(vq_path),
            local_files_only=True,
            dtype=dtype,
            attn_implementation="sdpa",
        )
        self.vlm: Any = vlm_model.to(device).eval()
        runner_class: Any = RDTRunner
        self.policy: Any = (
            runner_class.from_pretrained(
                str(action_expert_path),
                local_files_only=True,
            )
            .to(device=device, dtype=dtype)
            .eval()
        )

        self.action_scale, self.action_offset = _load_rdt2_normalizer(normalizer_path)

    @torch.inference_mode()
    def predict(self, image: np.ndarray, instruction: str, state: np.ndarray) -> np.ndarray:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": instruction},
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=False)
        prompt += "<|im_start|>assistant\n<|quad_start|>"
        # The public adapter accepts one camera. Duplicate that real observation
        # to satisfy RDT2's released left/right wrist-camera input contract.
        stereo = Image.fromarray(np.concatenate([image, image], axis=1), mode="RGB")
        inputs = self.processor(
            text=[prompt],
            images=[[stereo]],
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        outputs = self.vlm(**inputs, use_cache=True)
        cache = outputs.past_key_values
        if hasattr(cache, "layers"):
            available_layers = [(layer.keys, layer.values) for layer in cache.layers]
        else:
            available_layers = [(layer[0], layer[1]) for layer in cache]
        layer_cache = [available_layers[index] for index in self.selected_layers]
        attention_mask = inputs["attention_mask"].to(self.device, dtype=torch.bool)
        state_tensor = torch.from_numpy(state).reshape(1, 1, ACTION_DIM).float()
        state_tensor = state_tensor.to(self.device, dtype=self.dtype)
        normalized = (
            self.policy.predict_action(
                lang_kv_cache=layer_cache,
                lang_attn_mask=attention_mask,
                img_tokens=None,
                state_tokens=state_tensor,
            )
            .float()
            .cpu()
        )
        actions = (normalized - self.action_offset) / self.action_scale
        actions[..., 9] = actions[..., 9] / 0.088 * 0.1
        actions[..., 19] = actions[..., 19] / 0.088 * 0.1
        return actions.squeeze(0).numpy().astype(np.float32, copy=False)

    def unload(self) -> None:
        self.policy = None
        self.vlm = None


class RDT2Adapter(TeacherAdapter):
    """RDT2-FM's genuine 24-step, 20-D bimanual UMI policy contract."""

    def __init__(self) -> None:
        self._runtime: Any | None = None
        self._checkpoint: Path | None = None

    def _load_runtime(
        self,
        model_path: Path,
        vq_path: Path,
        normalizer_path: Path,
        device: str,
        dtype: torch.dtype,
    ) -> Any:
        return _RDT2Runtime(model_path, vq_path, normalizer_path, device, dtype)

    def load(self, model_path: Path, device: str = "cpu", dtype: torch.dtype = torch.float32) -> None:
        path = Path(model_path).expanduser()
        if not path.is_dir() or not (path / "config.json").is_file():
            raise FileNotFoundError(f"RDT2-FM checkpoint directory not found or incomplete: {path}")
        vq_path = path.parent / VQ_DIRECTORY
        normalizer_path = path.parent / NORMALIZER_FILENAME
        if not vq_path.is_dir():
            raise FileNotFoundError(f"RDT2-VQ companion backbone not found: {vq_path}")
        if not normalizer_path.is_file():
            raise FileNotFoundError(f"RDT2 official UMI normalizer not found: {normalizer_path}")
        self._runtime = self._load_runtime(path, vq_path, normalizer_path, device, dtype)
        self._checkpoint = path.resolve()

    def predict(
        self,
        image: np.ndarray,
        instruction: str,
        proprioception: np.ndarray | None = None,
    ) -> ActionChunk:
        if self._runtime is None or self._checkpoint is None:
            raise RuntimeError("RDT2-FM is not loaded")
        rgb = np.asarray(image)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError("RDT2 image must be HxWx3 uint8 RGB")
        if not instruction.strip():
            raise ValueError("RDT2 instruction must not be empty")
        # The released RDT2-FM checkpoint was trained and published with an
        # all-zero state token. Its official inferencer passes state directly
        # (without the action normalizer), preserving the input only as a
        # future fine-tuning interface. Keep that exact pretrained contract.
        if proprioception is not None:
            source = np.asarray(proprioception, dtype=np.float32).reshape(-1)
            if not np.isfinite(source).all():
                raise ValueError("RDT2 proprioception contains non-finite values")
        state = np.zeros(ACTION_DIM, dtype=np.float32)

        actions = np.asarray(self._runtime.predict(np.ascontiguousarray(rgb), instruction.strip(), state))
        if actions.shape != (ACTION_HORIZON, ACTION_DIM):
            raise ValueError(f"RDT2 returned {actions.shape}; expected {(ACTION_HORIZON, ACTION_DIM)}")
        if not np.isfinite(actions).all():
            raise ValueError("RDT2 returned non-finite actions")
        actions = actions.astype(np.float32, copy=False)
        return ActionChunk(
            actions=actions,
            action_mean=actions.copy(),
            action_std=np.zeros_like(actions),
            confidence=np.ones_like(actions),
            metadata={
                "teacher": ADAPTER_NAME,
                "architecture": "hybrid-vq-flow",
                "checkpoint": str(self._checkpoint),
                "inference": "real",
                "camera_mapping": "single-real-frame-duplicated-to-left-and-right-stereo",
                "state_conditioning": "official-pretrained-zero-state",
            },
        )

    def extract_features(self, image: np.ndarray, instruction: str) -> dict[str, np.ndarray]:
        return {}

    def get_action_space(self) -> dict[str, Any]:
        side = ("x", "y", "z", "rot6d_1", "rot6d_2", "rot6d_3", "rot6d_4", "rot6d_5", "rot6d_6", "gripper")
        return {
            "dim": ACTION_DIM,
            "min": np.full(ACTION_DIM, -np.inf, dtype=np.float32),
            "max": np.full(ACTION_DIM, np.inf, dtype=np.float32),
            "names": [f"{arm}_{name}" for arm in ("right", "left") for name in side],
        }

    def info(self) -> TeacherInfo:
        return TeacherInfo(
            name=ADAPTER_NAME,
            architecture="hybrid-vq-flow",
            param_count=7.5,
            action_dim=ACTION_DIM,
            action_horizon=ACTION_HORIZON,
            vision_encoder="RDT2-VQ (Qwen2.5-VL-7B)",
            language_model="RDT2-VQ (Qwen2.5-VL-7B)",
            supports_chunking=True,
            supports_features=False,
        )

    @property
    def is_loaded(self) -> bool:
        return self._runtime is not None

    def unload(self) -> None:
        if self._runtime is not None and hasattr(self._runtime, "unload"):
            self._runtime.unload()
        self._runtime = None
        self._checkpoint = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


ADAPTER_CLASS = RDT2Adapter

__all__ = ["RDT2Adapter"]
