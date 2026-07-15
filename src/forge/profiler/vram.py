"""VRAM estimation for FORGE model variants."""

from __future__ import annotations

from forge.config import StudentConfig
from forge.profiler.dataclasses import VRAMEstimate
from forge.profiler.flops import VARIANT_SPECS

GPU_PROFILES = {
    "L4_24GB": 24.0,
    "A100_40GB": 40.0,
    "A100_80GB": 80.0,
    "RTX4090_24GB": 24.0,
    "Jetson_Orin_8GB": 8.0,
    "Jetson_Orin_16GB": 16.0,
    "T4_16GB": 16.0,
}


def estimate_vram(config: StudentConfig, gpu_vram_gb: float = 24.0) -> VRAMEstimate:
    """Estimate VRAM usage for a FORGE student across precisions and use cases.

    Formula-based; no weights are loaded.  Covers FP32/FP16 inference,
    mixed-precision and FP32 training, per-sample activation cost, a
    recommended batch size for the target GPU, and a boolean fit-table.

    Args:
        config: A fully populated ``StudentConfig`` instance.
        gpu_vram_gb: Target GPU VRAM in GB for batch-size recommendation.

    Returns:
        ``VRAMEstimate`` with all figures in MB and a per-GPU fit table.
    """
    specs = VARIANT_SPECS.get(config.variant, VARIANT_SPECS["nano"])

    # 1. Parameter counts ------------------------------------------------
    vision_params: int = 400_000_000  # SigLIP SO400M — always frozen
    lm_params: int = specs["lm_params"]

    bridge_params: int = (
        config.bridge_n_layers * 12 * config.bridge_d_model**2
        + config.bridge_d_vision * config.bridge_d_model
        + config.bridge_n_queries * config.bridge_d_model
    )
    lora_params: int = len(config.lora_target_modules) * specs["lm_layers"] * 2 * specs["d_model"] * config.lora_rank
    action_params: int = (
        config.action_head_layers * (256 * 256 * 2 + 256 * config.bridge_d_model)
        + config.bridge_d_model * 256
        + 256 * config.action_dim
        + 256 * 4
    )

    frozen_params: int = vision_params + lm_params
    trainable_params: int = bridge_params + lora_params + action_params
    total_params: int = frozen_params + trainable_params

    # 2. Memory in MB (bytes / 1e6) --------------------------------------
    inference_mb: float = total_params * 4 / 1e6
    inference_fp16_mb: float = total_params * 2 / 1e6

    # Mixed-precision: frozen FP16, trainable FP16 + FP32 master copy
    # + FP16 gradients + Adam FP32 m/v states.
    training_fp16_mb: float = (
        frozen_params * 2  # frozen weights in FP16
        + trainable_params * 2  # FP16 weights
        + trainable_params * 4  # FP32 master copy
        + trainable_params * 2  # FP16 gradients
        + trainable_params * 8  # Adam m + v in FP32 (4 bytes each)
    ) / 1e6

    # Full FP32 training: weights + gradients + optimizer states.
    training_mb: float = (
        total_params * 4  # FP32 weights
        + trainable_params * 4  # FP32 gradients
        + trainable_params * 8  # Adam FP32 states
    ) / 1e6

    # 3. Per-sample activation memory (FP16) -----------------------------
    seq_len: int = config.bridge_n_queries + 1
    per_sample_activation_mb: float = 2 * specs["lm_layers"] * seq_len * specs["d_model"] * 2 / 1e6

    # 4. Recommended batch size for the target GPU -----------------------
    available_mb: float = gpu_vram_gb * 1024 - training_fp16_mb
    if available_mb > 0 and per_sample_activation_mb > 0:
        recommended_batch: int = max(1, int(available_mb / per_sample_activation_mb))
        recommended_batch = min(recommended_batch, 64)
    else:
        recommended_batch = 1

    # 5. GPU fit table (batch=1, mixed-precision) ------------------------
    fits_gpu: dict[str, bool] = {
        gpu_name: (gpu_gb * 1024) >= (training_fp16_mb + per_sample_activation_mb)
        for gpu_name, gpu_gb in GPU_PROFILES.items()
    }

    return VRAMEstimate(
        inference_mb=inference_mb,
        inference_fp16_mb=inference_fp16_mb,
        training_mb=training_mb,
        training_fp16_mb=training_fp16_mb,
        per_sample_activation_mb=per_sample_activation_mb,
        recommended_batch_size=recommended_batch,
        fits_gpu=fits_gpu,
    )
