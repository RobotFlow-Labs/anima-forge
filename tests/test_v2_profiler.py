"""Tests for FORGE Model Profiler."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from forge.config import StudentConfig
from forge.profiler.dataclasses import (
    ComponentProfile,
    FLOPsEstimate,
    ModelProfileCard,
    RecommendedHyperparams,
    VRAMEstimate,
)
from forge.profiler.flops import VARIANT_SPECS, estimate_flops
from forge.profiler.markdown import generate_ascii_diagram, generate_markdown
from forge.profiler.profiler import FORGEProfiler
from forge.profiler.recommend import recommend_hyperparams
from forge.profiler.vram import estimate_vram

# ── Dataclass tests ───────────────────────────────────────


def test_component_profile_dataclass():
    """Create a ComponentProfile and verify all fields are accessible."""
    cp = ComponentProfile(
        name="vision_encoder",
        param_count=400_000_000,
        trainable_params=0,
        frozen_params=400_000_000,
        input_shape="(B, 3, 384, 384)",
        output_shape="(B, 729, 1152)",
        estimated_flops=583_200_000_000,
        estimated_memory_mb=800.0,
    )
    assert cp.name == "vision_encoder"
    assert cp.param_count == 400_000_000
    assert cp.trainable_params == 0
    assert cp.frozen_params == 400_000_000
    assert cp.input_shape == "(B, 3, 384, 384)"
    assert cp.output_shape == "(B, 729, 1152)"
    assert cp.estimated_flops == 583_200_000_000
    assert cp.estimated_memory_mb == pytest.approx(800.0)


def test_model_profile_card_to_dict():
    """Create a minimal ModelProfileCard, call to_dict(), verify expected keys and types."""
    card = ModelProfileCard(
        model_name="FORGE-Nano",
        variant="nano",
        vision_encoder="google/siglip-so400m-patch14-384",
        language_model="Qwen/Qwen2.5-0.5B",
        action_head_type="flow",
        action_dim=7,
        action_horizon=1,
    )
    result = card.to_dict()

    assert isinstance(result, dict)
    assert result["model_name"] == "FORGE-Nano"
    assert result["variant"] == "nano"
    assert result["action_dim"] == 7
    assert result["action_horizon"] == 1
    assert isinstance(result["components"], list)
    assert result["flops"] is None
    assert result["vram"] is None
    assert result["recommended_hp"] is None
    assert isinstance(result["bridge_config"], dict)


def test_model_profile_card_roundtrip():
    """from_dict(card.to_dict()) produces identical values."""
    profiler = FORGEProfiler("nano")
    original = profiler.generate_card()

    restored = ModelProfileCard.from_dict(original.to_dict())

    assert restored.model_name == original.model_name
    assert restored.variant == original.variant
    assert restored.total_params == original.total_params
    assert restored.trainable_params == original.trainable_params
    assert restored.frozen_params == original.frozen_params
    assert restored.action_dim == original.action_dim
    assert restored.action_horizon == original.action_horizon
    assert isinstance(restored.flops, FLOPsEstimate)
    assert restored.flops.total_gflops == pytest.approx(original.flops.total_gflops)
    assert isinstance(restored.vram, VRAMEstimate)
    assert restored.vram.training_fp16_mb == pytest.approx(original.vram.training_fp16_mb)
    assert isinstance(restored.recommended_hp, RecommendedHyperparams)
    assert restored.recommended_hp.learning_rate == pytest.approx(original.recommended_hp.learning_rate)
    assert len(restored.components) == len(original.components)
    assert restored.components[0].name == original.components[0].name


def test_model_profile_card_json_file(tmp_path: Path):
    """save_json and from_json produce a faithful roundtrip."""
    profiler = FORGEProfiler("nano")
    original = profiler.generate_card()

    json_path = tmp_path / "nano_profile.json"
    original.save_json(str(json_path))

    assert json_path.exists()
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)

    restored = ModelProfileCard.from_json(str(json_path))
    assert restored.model_name == original.model_name
    assert restored.variant == original.variant
    assert restored.total_params == original.total_params
    assert isinstance(restored.flops, FLOPsEstimate)
    assert isinstance(restored.vram, VRAMEstimate)
    assert isinstance(restored.recommended_hp, RecommendedHyperparams)


# ── FORGEProfiler init tests ──────────────────────────────


def test_profiler_init_nano():
    """FORGEProfiler('nano') succeeds and has the correct variant."""
    profiler = FORGEProfiler("nano")
    assert profiler.variant == "nano"
    assert profiler.student_config is not None
    assert profiler.config is not None


def test_profiler_init_all_variants():
    """Each supported variant initialises without error."""
    for variant in ["micro", "nano", "small", "medium"]:
        profiler = FORGEProfiler(variant)
        assert profiler.variant == variant


# ── profile_params tests ──────────────────────────────────


def test_profile_params_5_components():
    """profile_params() returns exactly 5 items with the expected names."""
    profiler = FORGEProfiler("nano")
    components = profiler.profile_params()

    expected_names = ["vision_encoder", "bridge", "language", "lora", "action_head"]
    assert len(components) == 5
    assert [c.name for c in components] == expected_names


def test_profile_params_trainable_lt_total():
    """For the nano profiler, sum of trainable params is less than sum of total params."""
    profiler = FORGEProfiler("nano")
    components = profiler.profile_params()

    total = sum(c.param_count for c in components)
    trainable = sum(c.trainable_params for c in components)
    assert trainable < total


def test_profile_params_vision_frozen():
    """The vision_encoder component has zero trainable parameters."""
    profiler = FORGEProfiler("nano")
    components = profiler.profile_params()

    vision = next(c for c in components if c.name == "vision_encoder")
    assert vision.trainable_params == 0
    assert vision.frozen_params == vision.param_count


# ── FLOPs estimation tests ────────────────────────────────


def test_estimate_flops_positive():
    """All per-component FLOPs fields and total_gflops are strictly positive."""
    cfg = StudentConfig(variant="nano")
    flops = estimate_flops(cfg)

    assert flops.vision_encoder > 0
    assert flops.bridge_attention > 0
    assert flops.language_backbone > 0
    assert flops.lora_adapters > 0
    assert flops.action_head > 0
    assert flops.total_gflops > 0


def test_estimate_flops_vision_dominates():
    """Vision encoder FLOPs exceed 50% of total GFLOPs for the nano variant."""
    cfg = StudentConfig(variant="nano")
    flops = estimate_flops(cfg)

    total_flops = flops.total_gflops * 1e9
    assert flops.vision_encoder > 0.5 * total_flops


# ── VRAM estimation tests ─────────────────────────────────


def test_estimate_vram_training_gt_inference():
    """Training VRAM (FP16 mixed) is greater than FP16 inference VRAM."""
    cfg = StudentConfig(variant="nano")
    vram = estimate_vram(cfg, gpu_vram_gb=24.0)

    assert vram.training_fp16_mb > vram.inference_fp16_mb


def test_estimate_vram_fits_gpu():
    """The L4_24GB GPU reports True in the fits_gpu table for the nano variant."""
    cfg = StudentConfig(variant="nano")
    vram = estimate_vram(cfg, gpu_vram_gb=24.0)

    assert "L4_24GB" in vram.fits_gpu
    assert vram.fits_gpu["L4_24GB"] is True


# ── Hyperparameter recommendation tests ──────────────────


def test_recommend_lr_scales_with_variant():
    """Learning rate decreases monotonically from micro → nano → small → medium."""
    variants = ["micro", "nano", "small", "medium"]
    lrs = []
    for variant in variants:
        cfg = StudentConfig(variant=variant)
        hp = recommend_hyperparams(cfg, dataset_size=50000, gpu_vram_gb=24.0)
        lrs.append(hp.learning_rate)

    # micro lr > nano lr > small lr > medium lr
    assert lrs[0] > lrs[1] > lrs[2] > lrs[3]


def test_recommend_batch_fits_vram():
    """Recommended batch size times per-sample activation fits within the GPU budget."""
    cfg = StudentConfig(variant="nano")
    vram = estimate_vram(cfg, gpu_vram_gb=24.0)
    hp = recommend_hyperparams(cfg, dataset_size=50000, gpu_vram_gb=24.0)

    activation_cost_mb = hp.batch_size * vram.per_sample_activation_mb
    gpu_budget_mb = 24.0 * 1024  # 24 GB expressed in MB
    assert activation_cost_mb < gpu_budget_mb


# ── Full card generation tests ────────────────────────────


def test_generate_card_complete():
    """generate_card() returns a card with all primary fields populated."""
    profiler = FORGEProfiler("nano")
    card = profiler.generate_card()

    assert card.model_name == "FORGE-Nano"
    assert card.variant == "nano"
    assert card.flops is not None
    assert card.vram is not None
    assert card.recommended_hp is not None
    assert card.total_params > 0
    assert card.trainable_params > 0
    assert card.frozen_params > 0
    assert card.fp32_size_mb > 0
    assert card.fp16_size_mb > 0
    assert card.int8_size_mb > 0
    assert card.int4_size_mb > 0
    assert len(card.components) == 5
    assert card.timestamp != ""


def test_generate_card_all_variants():
    """generate_card() succeeds for every supported variant."""
    for variant in ["micro", "nano", "small", "medium"]:
        profiler = FORGEProfiler(variant)
        card = profiler.generate_card()
        assert card.variant == variant
        assert card.model_name == f"FORGE-{variant.capitalize()}"
        assert card.flops is not None
        assert card.vram is not None


# ── Markdown generation tests ─────────────────────────────


def test_generate_markdown_sections():
    """Markdown output contains all required section headings."""
    profiler = FORGEProfiler("nano")
    card = profiler.generate_card()
    md = generate_markdown(card)

    required_sections = [
        "## Model Details",
        "## Architecture",
        "## Parameter Breakdown",
        "## Performance Estimates",
        "## Recommended Training Config",
    ]
    for section in required_sections:
        assert section in md, f"Missing section: {section!r}"


def test_ascii_diagram_has_variant():
    """generate_ascii_diagram(card) contains the model name (e.g. 'FORGE-Nano')."""
    profiler = FORGEProfiler("nano")
    card = profiler.generate_card()
    diagram = generate_ascii_diagram(card)

    assert "FORGE-Nano" in diagram


# ── AutoSense integration test ────────────────────────────


def test_profiler_with_autosense(tmp_path: Path):
    """FORGEProfiler with model_dir uses AutoSense to detect bridge_d_model."""
    model_dir = tmp_path / "models"
    lm_dir = model_dir / "Qwen--Qwen3-0.6B"
    lm_dir.mkdir(parents=True)
    (lm_dir / "config.json").write_text(json.dumps({"hidden_size": 1024, "num_hidden_layers": 28}))

    profiler = FORGEProfiler("nano", model_dir=str(model_dir))
    assert profiler.student_config.bridge_d_model == 1024


def test_v3_variant_specs_match_backbones():
    """Analytical cards use the four GPU-verified v3 language backbones."""
    expected = {
        "micro": (135_000_000, 576),
        "nano": (600_000_000, 1024),
        "small": (1_700_000_000, 2048),
        "medium": (4_000_000_000, 2560),
    }
    for variant, (params, d_model) in expected.items():
        assert VARIANT_SPECS[variant]["lm_params"] == params
        assert FORGEProfiler(variant).student_config.bridge_d_model == d_model
