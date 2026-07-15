"""Flagship battle configuration contract."""

from forge.config import ForgeConfig


def test_flagship_config_uses_v3_best_nano_contract() -> None:
    config = ForgeConfig.from_yaml("configs/forge_nano_flagship.yaml")

    assert config.student.variant == "nano"
    assert config.student.language_model == "Qwen/Qwen3-0.6B"
    assert config.student.vision_encoder == "google/siglip2-so400m-patch14-384"
    assert config.student.action_head_type == "flow"
    assert config.student.lora_rank == 64
    assert config.distill.max_steps == 5000
    assert config.distill.batch_size == 80
    assert config.pruning.target_layers == 11
