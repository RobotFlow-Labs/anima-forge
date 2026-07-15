"""MolmoAct2 LIBERO teacher adapter using the official LeRobot policy."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from forge.teachers.base import TeacherInfo
from forge.teachers.lerobot_policy_adapter import LeRobotPolicyAdapter

ADAPTER_NAME = "molmoact2-libero"


class MolmoAct2Adapter(LeRobotPolicyAdapter):
    """Adapter for Ai2's 2026 MolmoAct2 continuous action policy."""

    policy_module = "lerobot.policies.molmoact2.modeling_molmoact2"
    policy_class_name = "MolmoAct2Policy"
    teacher_info = TeacherInfo(
        name=ADAPTER_NAME,
        architecture="hybrid-ar-flow",
        param_count=5.0,
        action_dim=7,
        action_horizon=10,
        vision_encoder="MolmoAct2 vision backbone",
        language_model="MolmoAct2 VLM",
        supports_chunking=True,
        supports_features=False,
    )

    def _preflight_companions(self, model_path: Path) -> None:
        base_path = model_path.parent / "allenai--MolmoAct2-LIBERO"
        tokenizer_path = model_path.parent / "allenai--MolmoAct2-FAST-Tokenizer"
        missing = [path for path in (base_path, tokenizer_path) if not path.is_dir()]
        if missing:
            names = ", ".join(str(path) for path in missing)
            raise FileNotFoundError(f"MolmoAct2 companion checkpoints not found: {names}")
        self._base_path = base_path.resolve()
        self._tokenizer_path = tokenizer_path.resolve()
        self._companion_paths = (self._base_path, self._tokenizer_path)

    def _load_policy(self, policy_class: Any, model_path: Path, device: str) -> Any:
        from lerobot.configs.policies import PreTrainedConfig  # type: ignore[import-untyped]

        self._preflight_companions(model_path)
        payload = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
        # The published LIBERO LeRobot config writes null for this non-optional
        # integer. LeRobot 0.6's draccus decoder rejects it before overrides apply.
        if payload.get("scheduler_decay_steps") is None:
            payload["scheduler_decay_steps"] = 100_000
        with tempfile.TemporaryDirectory(prefix="forge-molmoact2-config-") as directory:
            Path(directory, "config.json").write_text(json.dumps(payload), encoding="utf-8")
            config = PreTrainedConfig.from_pretrained(directory, local_files_only=True)
        config.checkpoint_path = str(self._base_path)
        config.discrete_action_tokenizer = str(self._tokenizer_path)
        config.inference_action_mode = "continuous"
        config.device = device
        return policy_class.from_pretrained(
            str(model_path),
            config=config,
            local_files_only=True,
        )

    def _processor_overrides(self, device: str) -> dict[str, dict[str, Any]]:
        overrides = super()._processor_overrides(device)
        overrides["molmoact2_pack_inputs"] = {
            "checkpoint_path": str(self._base_path),
            "discrete_action_tokenizer": str(self._tokenizer_path),
        }
        return overrides


ADAPTER_CLASS = MolmoAct2Adapter

__all__ = ["MolmoAct2Adapter"]
