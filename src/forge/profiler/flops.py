"""Analytical FLOPs estimation for FORGE model components."""

from __future__ import annotations

from forge.config import StudentConfig
from forge.profiler.dataclasses import FLOPsEstimate

# Variant → (lm_params, lm_layers, d_model) lookup.
# lm_params: backbone parameter count; lm_layers: transformer depth;
# d_model: hidden dimension for self-attention and FFN blocks.
VARIANT_SPECS: dict[str, dict[str, int]] = {
    "micro": {"lm_params": 135_000_000, "lm_layers": 30, "d_model": 576},
    "nano": {"lm_params": 600_000_000, "lm_layers": 28, "d_model": 1024},
    "small": {"lm_params": 1_700_000_000, "lm_layers": 28, "d_model": 2048},
    "medium": {"lm_params": 4_000_000_000, "lm_layers": 36, "d_model": 2560},
}


def estimate_flops(config: StudentConfig) -> FLOPsEstimate:
    """Estimate per-component FLOPs for a FORGE student analytically.

    No model weights are loaded; all estimates are formula-based and run on
    CPU in microseconds.  Formulas follow the 2*M*N MAC convention.

    Args:
        config: A fully populated ``StudentConfig`` instance.

    Returns:
        A ``FLOPsEstimate`` with per-component integer FLOPs and the combined
        total expressed in GFLOPs (float).

    Examples:
        >>> from forge.config import StudentConfig
        >>> from forge.profiler.flops import estimate_flops
        >>> est = estimate_flops(StudentConfig())
        >>> est.total_gflops > 0
        True
    """
    specs = VARIANT_SPECS.get(config.variant, VARIANT_SPECS["nano"])
    d = specs["d_model"]
    n_layers = specs["lm_layers"]

    # 1. Vision encoder — SigLIP ViT-SO400M, fixed 400M params / 729 tokens
    vision_flops: int = 2 * 400_000_000 * 729

    # 2. Bridge attention — cross-attention between vision and language tokens
    d_v: int = config.bridge_d_vision  # typically 1152
    d_m: int = config.bridge_d_model  # varies by variant
    nq: int = config.bridge_n_queries  # typically 64
    per_bridge_layer: int = 4 * nq * d_m * (d_v + d_m)
    bridge_flops: int = config.bridge_n_layers * per_bridge_layer

    # 3. Language backbone — standard transformer layer cost
    seq_len: int = config.bridge_n_queries + 1  # vision tokens + 1 lang token
    per_lm_layer: int = 12 * d * d * seq_len
    language_flops: int = n_layers * per_lm_layer

    # 4. LoRA adapters — low-rank projection pairs on target modules
    rank: int = config.lora_rank
    n_target: int = len(config.lora_target_modules)  # typically 4
    per_adapter: int = 2 * d * rank + 2 * rank * d
    lora_flops: int = n_target * n_layers * per_adapter

    # 5. Action head — cost depends on head type and number of inference steps
    d_h: int = 256
    n_blocks: int = config.action_head_layers

    if config.action_head_type == "diffusion":
        k_steps: int = config.action_diffusion_steps
    elif config.action_head_type == "flow":
        k_steps = config.flow_inference_steps
    else:
        k_steps = 1  # "chunk" or "consistency" — single forward pass

    per_step: int = 4 * d_h * d_h * n_blocks + config.bridge_d_model * d_h
    action_flops: int = k_steps * per_step

    # 6. Total GFLOPs
    total_gflops: float = (vision_flops + bridge_flops + language_flops + lora_flops + action_flops) / 1e9

    return FLOPsEstimate(
        vision_encoder=vision_flops,
        bridge_attention=bridge_flops,
        language_backbone=language_flops,
        lora_adapters=lora_flops,
        action_head=action_flops,
        total_gflops=total_gflops,
    )
