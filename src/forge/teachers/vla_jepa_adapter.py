"""VLA-JEPA teacher adapter using the official LeRobot policy."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from forge.teachers.base import TeacherInfo
from forge.teachers.lerobot_policy_adapter import LeRobotPolicyAdapter

ADAPTER_NAME = "vla-jepa-3b"


class VLAJEPAAdapter(LeRobotPolicyAdapter):
    """Adapter for the 2026 Qwen3-VL + flow-DiT VLA-JEPA policy."""

    policy_module = "lerobot.policies.vla_jepa.modeling_vla_jepa"
    policy_class_name = "VLAJEPAPolicy"
    teacher_info = TeacherInfo(
        name=ADAPTER_NAME,
        architecture="jepa-flow",
        param_count=3.0,
        action_dim=7,
        action_horizon=7,
        vision_encoder="Qwen3-VL-2B",
        language_model="Qwen3-VL-2B-Instruct",
        supports_chunking=True,
        supports_features=False,
    )

    def _preflight_companions(self, model_path: Path) -> None:
        qwen_path = model_path.parent / "Qwen--Qwen3-VL-2B-Instruct"
        jepa_path = model_path.parent / "facebook--vjepa2-vitl-fpc64-256"
        missing = [path for path in (qwen_path, jepa_path) if not path.is_dir()]
        if missing:
            names = ", ".join(str(path) for path in missing)
            raise FileNotFoundError(f"VLA-JEPA companion checkpoints not found: {names}")
        self._companion_paths = (qwen_path.resolve(), jepa_path.resolve())

    def _load_policy(self, policy_class: Any, model_path: Path, device: str) -> Any:
        from lerobot.configs.policies import PreTrainedConfig  # type: ignore[import-untyped]

        self._preflight_companions(model_path)
        qwen_path, jepa_path = self._companion_paths
        config = PreTrainedConfig.from_pretrained(str(model_path), local_files_only=True)
        config.qwen_model_name = str(qwen_path)
        config.jepa_encoder_name = str(jepa_path)
        config.device = device
        return policy_class.from_pretrained(
            str(model_path),
            config=config,
            local_files_only=True,
        )


ADAPTER_CLASS = VLAJEPAAdapter

__all__ = ["VLAJEPAAdapter"]
