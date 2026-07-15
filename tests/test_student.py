"""Tests for PRD-02: Student Architecture."""

import torch


def test_bridge_attention_shape():
    """Verify Bridge Attention compresses vision tokens correctly."""
    from forge.modules.bridge_attention import BridgeAttention

    bridge = BridgeAttention(d_vision=1152, d_model=896, n_queries=64, n_heads=8, n_layers=4)

    # Simulate SigLIP output
    vis = torch.randn(2, 729, 1152)
    out = bridge(vis)

    assert out.shape == (2, 64, 896), f"Expected (2, 64, 896), got {out.shape}"
    assert bridge.param_count() > 0


def test_bridge_attention_small():
    """Test with smaller dimensions for speed."""
    from forge.modules.bridge_attention import BridgeAttention

    bridge = BridgeAttention(d_vision=128, d_model=64, n_queries=8, n_heads=4, n_layers=2)
    vis = torch.randn(4, 49, 128)
    out = bridge(vis)
    assert out.shape == (4, 8, 64)


def test_diffusion_head_training():
    """Verify diffusion head training mode produces loss."""
    from forge.modules.diffusion_head import DiffusionActionHead

    head = DiffusionActionHead(d_model=64, d_action=7, n_layers=2, n_diffusion_steps=5, d_hidden=32)

    features = torch.randn(4, 64)
    gt_actions = torch.randn(4, 7)

    result = head(features, gt_actions=gt_actions)

    assert "loss" in result
    assert result["loss"].shape == ()  # scalar
    assert result["loss"].item() > 0


def test_diffusion_head_inference():
    """Verify diffusion head inference mode produces actions."""
    from forge.modules.diffusion_head import DiffusionActionHead

    head = DiffusionActionHead(d_model=64, d_action=7, n_layers=2, n_diffusion_steps=5, d_hidden=32)

    features = torch.randn(4, 64)
    result = head(features, gt_actions=None)

    assert "actions" in result
    assert result["actions"].shape == (4, 7)
    assert "loss" not in result


def test_lora_application():
    """Verify LoRA wraps target modules correctly."""
    from forge.modules.lora import LoRALinear, apply_lora

    # Create a simple model
    model = torch.nn.Sequential(
        torch.nn.Linear(64, 64),  # not targeted
        torch.nn.Linear(64, 64),  # not targeted
    )
    model.q_proj = torch.nn.Linear(64, 64)  # targeted
    model.v_proj = torch.nn.Linear(64, 64)  # targeted

    model = apply_lora(model, rank=8, alpha=16, target_modules=["q_proj", "v_proj"])

    assert isinstance(model.q_proj, LoRALinear)
    assert isinstance(model.v_proj, LoRALinear)
    assert model.q_proj.rank == 8
    assert model.v_proj.scaling == 16 / 8


def test_lora_is_only_trainable_part_of_frozen_backbone():
    """Freezing the base before injection leaves only LoRA matrices trainable."""
    from forge.modules.lora import apply_lora

    model = torch.nn.ModuleDict(
        {
            "q_proj": torch.nn.Linear(16, 16),
            "mlp": torch.nn.Linear(16, 16),
        }
    )
    for parameter in model.parameters():
        parameter.requires_grad = False
    apply_lora(model, rank=4, alpha=8, target_modules=["q_proj"])

    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert trainable == ["q_proj.lora_A.weight", "q_proj.lora_B.weight"]


def test_lora_forward():
    """Verify LoRA forward pass adds low-rank adaptation."""
    from forge.modules.lora import LoRALinear

    original = torch.nn.Linear(64, 64)
    lora = LoRALinear(original, rank=8, alpha=16)

    x = torch.randn(2, 64)

    # LoRA output should be different from original (even with zero init of B)
    # Actually B is zero-initialized, so initially LoRA output == original output
    out_lora = lora(x)
    assert out_lora.shape == (2, 64)

    # After training, they should diverge
    # For now just verify shapes
    assert lora.trainable_params > 0
    assert lora.trainable_params == 8 * 64 + 8 * 64  # A: 64x8, B: 8x64


def test_lora_fp32_adapters_accept_bfloat16_backbone_activations():
    """Mixed precision keeps adapter parameters fp32 and preserves LM output dtype."""
    from forge.modules.lora import LoRALinear

    original = torch.nn.Linear(64, 64, bias=False, dtype=torch.bfloat16)
    lora = LoRALinear(original, rank=8, alpha=16)
    output = lora(torch.randn(2, 64, dtype=torch.bfloat16))

    assert lora.lora_A.weight.dtype == torch.float32
    assert output.dtype == torch.bfloat16


def test_v3_student_defaults():
    """Nano defaults point at the production Qwen3/SigLIP2 pair."""
    from forge.config import ForgeConfig, StudentConfig

    student = StudentConfig()
    config = ForgeConfig.default()
    assert student.vision_encoder == "google/siglip2-so400m-patch14-384"
    assert student.language_model == "Qwen/Qwen3-0.6B"
    assert student.backbone_dtype == "auto"
    assert config.paths.vision_encoder == "google--siglip2-so400m-patch14-384"
    assert config.paths.language_model == "Qwen--Qwen3-0.6B"


def test_student_mock_forward():
    """Test student forward pass with mock encoders (no real weights)."""
    from forge.config import StudentConfig
    from forge.student import FORGEStudent

    config = StudentConfig(
        variant="nano",
        bridge_d_vision=128,
        bridge_d_model=64,
        bridge_n_queries=8,
        bridge_n_heads=4,
        bridge_n_layers=2,
        action_dim=7,
        action_head_layers=2,
        action_diffusion_steps=5,
        lora_rank=8,
        lora_alpha=16,
    )

    student = FORGEStudent(config)

    # Forward pass
    images = torch.randn(2, 3, 384, 384)
    language_ids = torch.randint(0, 1000, (2, 32))

    result = student(images, language_ids=language_ids)
    assert "actions" in result
    assert result["actions"].shape == (2, 7)
    assert "vision_features" in result


def test_student_training_mode():
    """Test student forward pass with ground truth actions (training)."""
    from forge.config import StudentConfig
    from forge.student import FORGEStudent

    config = StudentConfig(
        variant="nano",
        bridge_d_vision=128,
        bridge_d_model=64,
        bridge_n_queries=8,
        bridge_n_heads=4,
        bridge_n_layers=2,
        action_dim=7,
        action_head_layers=2,
        action_diffusion_steps=5,
        lora_rank=8,
        lora_alpha=16,
    )

    student = FORGEStudent(config)

    images = torch.randn(2, 3, 384, 384)
    language_ids = torch.randint(0, 1000, (2, 32))
    gt_actions = torch.randn(2, 7)

    result = student(images, language_ids=language_ids, gt_actions=gt_actions)
    assert "loss" in result
    assert result["loss"].shape == ()
    assert result["loss"].requires_grad


def test_student_param_counts():
    """Verify trainable vs total param split."""
    from forge.config import StudentConfig
    from forge.student import FORGEStudent

    config = StudentConfig(
        variant="nano",
        bridge_d_vision=128,
        bridge_d_model=64,
        bridge_n_queries=8,
        bridge_n_heads=4,
        bridge_n_layers=2,
        action_dim=7,
        action_head_layers=2,
        action_diffusion_steps=5,
        lora_rank=8,
        lora_alpha=16,
    )

    student = FORGEStudent(config)

    # With mock encoders, all params are trainable
    # With real SigLIP, trainable < total (vision encoder frozen)
    assert student.trainable_params > 0
    assert student.total_params > 0
    assert len(student.trainable_parameters()) > 0
    # Verify the student actually has meaningful parameter count
    assert student.total_params > 100_000  # At least 100K params
