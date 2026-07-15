"""Canonical v3 variant and checkpoint architecture contracts."""

from pathlib import Path

import yaml

from forge.config import (
    ForgeConfig,
    StudentConfig,
    apply_checkpoint_student_config,
    apply_student_variant,
)


def test_all_variant_presets_select_the_real_v3_backbones() -> None:
    expected = {
        "micro": ("HuggingFaceTB/SmolLM2-135M", 576, 16),
        "nano": ("Qwen/Qwen3-0.6B", 1024, 32),
        "small": ("Qwen/Qwen3-1.7B", 2048, 64),
        "medium": ("Qwen/Qwen3-4B", 2560, 64),
    }
    for variant, (model_id, d_model, lora_rank) in expected.items():
        config = apply_student_variant(StudentConfig(), variant)
        assert config.language_model == model_id
        assert config.bridge_d_model == d_model
        assert config.lora_rank == lora_rank


def test_checkpoint_student_config_restores_flagship_head() -> None:
    config = apply_student_variant(StudentConfig(), "nano")
    apply_checkpoint_student_config(
        config,
        {
            "student_config": {
                "action_head_type": "flow",
                "lora_rank": 64,
                "lora_alpha": 128,
            }
        },
    )
    assert config.action_head_type == "flow"
    assert config.lora_rank == 64
    assert config.lora_alpha == 128


def test_all_validation_configs_request_onnx_mlx_and_tensorrt() -> None:
    for variant in ("micro", "nano", "small", "medium"):
        config = ForgeConfig.from_yaml(Path("configs") / f"forge_{variant}.yaml")
        assert set(config.export.formats) == {"onnx", "mlx", "tensorrt"}


def test_default_student_dimensions_match_qwen3_nano() -> None:
    config = StudentConfig()
    assert config.language_model == "Qwen/Qwen3-0.6B"
    assert config.bridge_d_model == 1024


def test_variant_only_yaml_applies_full_canonical_preset(tmp_path: Path) -> None:
    path = tmp_path / "small.yaml"
    path.write_text(yaml.safe_dump({"student": {"variant": "small"}}))

    config = ForgeConfig.from_yaml(path)

    assert config.student.language_model == "Qwen/Qwen3-1.7B"
    assert config.student.bridge_d_model == 2048
    assert config.student.lora_rank == 64
