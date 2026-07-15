"""Tests with the current real local student model fleet.

These tests load actual SigLIP2 and Qwen3 weights.
Skip automatically if models not available (CI-safe).
"""

import os
from pathlib import Path

import pytest
import torch

MODEL_DIR = Path(os.environ.get("FORGE_MODEL_DIR", "models"))
VISION_PATH = MODEL_DIR / "google--siglip2-so400m-patch14-384"
LANGUAGE_PATH = MODEL_DIR / "Qwen--Qwen3-0.6B"
HAS_MODELS = (VISION_PATH / "config.json").is_file() and (LANGUAGE_PATH / "config.json").is_file()

skip_no_models = pytest.mark.skipif(not HAS_MODELS, reason="Current local model fleet is unavailable")


@skip_no_models
def test_load_siglip_real():
    """Load real SigLIP2-SO400M and run inference."""
    from transformers import SiglipVisionModel

    model = SiglipVisionModel.from_pretrained(str(VISION_PATH), local_files_only=True)
    model.eval()

    params = sum(p.numel() for p in model.parameters())
    assert params > 300_000_000, f"SigLIP vision should have >300M params, got {params / 1e6:.0f}M"

    # Run inference
    images = torch.zeros(1, 3, 384, 384)
    with torch.no_grad():
        out = model(images)

    features = out.last_hidden_state
    assert features.shape == (1, 729, 1152), f"Expected (1, 729, 1152), got {features.shape}"
    print(f"SigLIP2 vision: {params / 1e6:.0f}M params, output shape {features.shape}")


@skip_no_models
def test_load_qwen_real():
    """Load real Qwen3-0.6B and run inference."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(str(LANGUAGE_PATH), local_files_only=True, dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(str(LANGUAGE_PATH), local_files_only=True)
    model.eval()
    assert tokenizer.vocab_size > 0

    params = sum(p.numel() for p in model.parameters())
    assert params > 500_000_000, f"Qwen3-0.6B should have >500M params, got {params / 1e6:.0f}M"

    # Test with embeddings input (how FORGE uses it)
    dummy_embeds = torch.zeros(1, 64, 1024)
    with torch.no_grad():
        out = model.model(inputs_embeds=dummy_embeds)

    assert out.last_hidden_state.shape == (1, 64, 1024)
    print(f"Qwen3-0.6B: {params / 1e6:.0f}M params, hidden_dim=1024")


@skip_no_models
def test_forge_student_real_weights():
    """Build FORGE-Nano with real SigLIP + Qwen weights."""
    from forge.config import StudentConfig
    from forge.student import FORGEStudent

    config = StudentConfig(
        variant="nano",
        vision_encoder="google/siglip2-so400m-patch14-384",
        language_model="Qwen/Qwen3-0.6B",
        bridge_d_vision=1152,
        bridge_d_model=1024,
        bridge_n_queries=64,
        bridge_n_heads=8,
        bridge_n_layers=4,
        action_dim=7,
        action_head_layers=4,
        action_diffusion_steps=10,
        lora_rank=32,
        lora_alpha=64,
    )

    student = FORGEStudent(config, model_dir=str(MODEL_DIR))

    total = student.total_params
    trainable = student.trainable_params
    frozen = total - trainable

    print("\nFORGE-Nano with REAL weights:")
    print(f"  Total params:     {total / 1e6:.1f}M")
    print(f"  Trainable params: {trainable / 1e6:.1f}M")
    print(f"  Frozen params:    {frozen / 1e6:.1f}M")
    print(f"  Model size (bf16): {total * 2 / (1024**3):.2f} GB")

    assert total > 400_000_000, "FORGE-Nano should have >400M total params"
    assert trainable < total, "Trainable should be less than total (vision encoder frozen)"

    # Forward pass
    images = torch.zeros(1, 3, 384, 384)
    lang_ids = student.tokenizer("move the block", return_tensors="pt")["input_ids"]

    with torch.no_grad():
        out = student(images, language_ids=lang_ids)

    assert out["actions"].shape == (1, 7)
    assert out["vision_features"].shape[1] == 64  # Bridge compressed to 64 tokens
    print(f"  Forward pass: OK — actions shape {out['actions'].shape}")


@skip_no_models
def test_forge_student_real_training_step():
    """Run one training step with real weights."""
    from forge.config import StudentConfig
    from forge.student import FORGEStudent

    config = StudentConfig(
        variant="nano",
        vision_encoder="google/siglip2-so400m-patch14-384",
        language_model="Qwen/Qwen3-0.6B",
        bridge_d_vision=1152,
        bridge_d_model=1024,
        bridge_n_queries=64,
        bridge_n_heads=8,
        bridge_n_layers=4,
        action_dim=7,
        action_head_layers=4,
        action_diffusion_steps=10,
        lora_rank=32,
        lora_alpha=64,
    )

    student = FORGEStudent(config, model_dir=str(MODEL_DIR))

    # Forward with ground truth (training mode)
    images = torch.zeros(1, 3, 384, 384)
    lang_ids = student.tokenizer("move the block", return_tensors="pt")["input_ids"]
    gt_actions = torch.linspace(-0.5, 0.5, 7).unsqueeze(0)

    out = student(images, language_ids=lang_ids, gt_actions=gt_actions)

    assert "loss" in out
    assert out["loss"].requires_grad

    # Backward
    out["loss"].backward()

    # Check gradients flow to trainable params only
    for name, param in student.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"

    print(f"  Training step: OK — loss={out['loss'].item():.4f}")


@skip_no_models
@pytest.mark.timeout(300)
def test_mlx_export_real():
    """Export real FORGE-Nano to MLX format."""
    import tempfile

    from forge.config import StudentConfig
    from forge.export.mlx_export import export_mlx, validate_mlx_export
    from forge.student import FORGEStudent

    config = StudentConfig(
        variant="nano",
        vision_encoder="google/siglip2-so400m-patch14-384",
        language_model="Qwen/Qwen3-0.6B",
        bridge_d_vision=1152,
        bridge_d_model=1024,
        bridge_n_queries=64,
        bridge_n_heads=8,
        bridge_n_layers=2,  # Fewer layers for speed
        action_dim=7,
        action_head_layers=2,
        action_diffusion_steps=5,
        lora_rank=16,
        lora_alpha=32,
    )

    student = FORGEStudent(config, model_dir=str(MODEL_DIR))

    with tempfile.TemporaryDirectory() as tmpdir:
        export_mlx(student, tmpdir, config={"variant": "nano"})
        result = validate_mlx_export(student, tmpdir)

        assert result["status"] == "passed", f"MLX validation failed: {result['mismatches']}"
        print(f"  MLX export: {result['n_mlx_params']} params exported")
