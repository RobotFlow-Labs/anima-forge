"""SmolVLA teacher adapter using the official LeRobot policy runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from forge.teachers.base import TeacherInfo
from forge.teachers.lerobot_policy_adapter import LeRobotPolicyAdapter

ADAPTER_NAME = "smolvla-base"


class SmolVLAAdapter(LeRobotPolicyAdapter):
    """Real adapter for the released six-action SmolVLA base checkpoint."""

    policy_module = "lerobot.policies.smolvla.modeling_smolvla"
    policy_class_name = "SmolVLAPolicy"
    action_names = ("joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6")
    teacher_info = TeacherInfo(
        name=ADAPTER_NAME,
        architecture="flow",
        param_count=0.45,
        action_dim=6,
        action_horizon=50,
        vision_encoder="SmolVLM2 vision encoder",
        language_model="SmolVLM2-500M-Video-Instruct",
        supports_chunking=True,
        supports_features=False,
    )

    def _preflight_companions(self, model_path: Path) -> None:
        vlm_path = model_path.parent / "HuggingFaceTB--SmolVLM2-500M-Video-Instruct"
        if not vlm_path.is_dir():
            raise FileNotFoundError(
                f"SmolVLA companion VLM not found at {vlm_path}. Fetch "
                "HuggingFaceTB/SmolVLM2-500M-Video-Instruct with `forge models fetch`."
            )
        self._companion_vlm_path = vlm_path.resolve()

    def _load_policy(self, policy_class: Any, model_path: Path, device: str) -> Any:
        from lerobot.configs.policies import PreTrainedConfig  # type: ignore[import-untyped]

        self._preflight_companions(model_path)
        config = PreTrainedConfig.from_pretrained(str(model_path), local_files_only=True)
        config.vlm_model_name = str(self._companion_vlm_path)
        config.device = device
        return policy_class.from_pretrained(
            str(model_path),
            config=config,
            local_files_only=True,
        )

    def _processor_overrides(self, device: str) -> dict[str, dict[str, Any]]:
        overrides = super()._processor_overrides(device)
        overrides["tokenizer_processor"] = {"tokenizer_name": str(self._companion_vlm_path)}
        return overrides


ADAPTER_CLASS = SmolVLAAdapter

__all__ = ["SmolVLAAdapter"]
