"""Hyperparameter recommendation heuristics for FORGE variants."""

from __future__ import annotations

from forge.config import StudentConfig
from forge.profiler.dataclasses import RecommendedHyperparams

# Per-variant baseline hyperparameters calibrated for a 24 GB GPU.
VARIANT_DEFAULTS: dict[str, dict] = {
    "micro": {"lr": 5e-4, "lora_rank": 16, "bridge_layers": 2, "bridge_queries": 32, "batch_24gb": 32},
    "nano": {"lr": 2e-4, "lora_rank": 64, "bridge_layers": 4, "bridge_queries": 64, "batch_24gb": 16},
    "small": {"lr": 1e-4, "lora_rank": 32, "bridge_layers": 4, "bridge_queries": 64, "batch_24gb": 8},
    "medium": {"lr": 5e-5, "lora_rank": 32, "bridge_layers": 4, "bridge_queries": 64, "batch_24gb": 4},
}


def recommend_hyperparams(
    config: StudentConfig,
    dataset_size: int = 50000,
    gpu_vram_gb: float = 24.0,
    objective: str = "balanced",
) -> RecommendedHyperparams:
    """Return training hyperparameter recommendations for a FORGE student variant.

    Heuristics are derived from the variant baseline table and scaled by
    available GPU VRAM and the requested training objective.

    Args:
        config: Student architecture config.  Only ``config.variant`` is used
            for the lookup; all other fields are ignored.
        dataset_size: Total number of training samples.  Used to estimate the
            number of gradient steps needed for roughly 3 epochs.
        gpu_vram_gb: Available GPU VRAM in gigabytes.  Batch size is scaled
            linearly from the 24 GB baseline.
        objective: Training priority — one of ``"balanced"`` (default),
            ``"quality"`` (more conservative LR, more ODE steps), or
            ``"speed"`` (aggressive LR, single ODE step).

    Returns:
        A :class:`~forge.profiler.dataclasses.RecommendedHyperparams` instance
        with all fields filled and a plain-English ``rationale`` dict.

    Examples:
        >>> from forge.config import StudentConfig
        >>> cfg = StudentConfig(variant="nano")
        >>> hp = recommend_hyperparams(cfg, dataset_size=20000, gpu_vram_gb=16.0)
        >>> hp.action_head_type
        'flow'
    """
    defaults = VARIANT_DEFAULTS.get(config.variant, VARIANT_DEFAULTS["nano"])

    # --- Learning rate ---------------------------------------------------
    lr: float = defaults["lr"]
    if objective == "quality":
        lr *= 0.5  # more careful, slower convergence
    elif objective == "speed":
        lr *= 2.0  # faster convergence, higher risk

    # --- Batch size (VRAM-scaled) ----------------------------------------
    batch_24gb: int = defaults["batch_24gb"]
    scale: float = gpu_vram_gb / 24.0
    batch_size: int = max(1, int(batch_24gb * scale))
    batch_size = min(batch_size, 64)

    # --- Gradient accumulation ------------------------------------------
    # Target an effective batch that is at least 4x the 24 GB baseline.
    target_effective: int = max(batch_24gb * 4, 32)
    grad_accum: int = max(1, target_effective // batch_size)
    effective_batch: int = batch_size * grad_accum

    # --- Steps and warmup ------------------------------------------------
    max_steps: int = max(5000, int(dataset_size / effective_batch * 3))  # ~3 epochs
    warmup_steps: int = max(100, int(max_steps * 0.05))

    # --- Action head -----------------------------------------------------
    action_head_type: str = "flow"  # modern default for all variants
    flow_inference_steps: int = 1 if objective == "speed" else 4

    # --- Rationale -------------------------------------------------------
    rationale: dict[str, str] = {
        "learning_rate": (f"Base {defaults['lr']} for {config.variant}, scaled by objective={objective}"),
        "batch_size": (f"Scaled from {batch_24gb} (24GB baseline) to {gpu_vram_gb}GB GPU"),
        "max_steps": (f"~3 epochs over {dataset_size} samples at effective_batch={effective_batch}"),
        "lora_rank": f"Default for {config.variant} variant",
        "warmup_steps": "5% of max_steps",
        "action_head_type": "Flow matching (modern default, fewer inference steps)",
        "flow_inference_steps": (f"{'Fast 1-step' if objective == 'speed' else 'Balanced 4-step'} inference"),
    }

    return RecommendedHyperparams(
        learning_rate=lr,
        batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        effective_batch_size=effective_batch,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        weight_decay=0.01,
        lora_rank=defaults["lora_rank"],
        action_head_type=action_head_type,
        bridge_n_queries=defaults["bridge_queries"],
        bridge_n_layers=defaults["bridge_layers"],
        flow_inference_steps=flow_inference_steps,
        rationale=rationale,
    )
