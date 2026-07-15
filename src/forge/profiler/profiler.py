"""FORGEProfiler — deep introspection for FORGE student models."""

from __future__ import annotations

from datetime import UTC, datetime

from forge.config import ForgeConfig, StudentConfig, apply_student_variant
from forge.profiler.dataclasses import (
    ComponentProfile,
    FLOPsEstimate,
    ModelProfileCard,
    RecommendedHyperparams,
    VRAMEstimate,
)
from forge.profiler.flops import VARIANT_SPECS, estimate_flops
from forge.profiler.markdown import generate_ascii_diagram, generate_markdown
from forge.profiler.recommend import recommend_hyperparams
from forge.profiler.vram import estimate_vram

VARIANT_LM: dict[str, str] = {
    "micro": "HuggingFaceTB/SmolLM2-135M",
    "nano": "Qwen/Qwen3-0.6B",
    "small": "Qwen/Qwen3-1.7B",
    "medium": "Qwen/Qwen3-4B",
}


class FORGEProfiler:
    """Analytical profiler for FORGE student models.

    All core methods are formula-based — no model loading or GPU needed.
    """

    def __init__(
        self,
        variant: str = "nano",
        model_dir: str | None = None,
        device: str = "cpu",
    ) -> None:
        config = ForgeConfig.default()
        apply_student_variant(config.student, variant)
        lm_id = VARIANT_LM.get(variant, VARIANT_LM["nano"])
        config.student.language_model = lm_id
        config.paths.language_model = lm_id.replace("/", "--")
        if model_dir:
            from forge.autosense import apply_autosense

            config.paths.model_dir = model_dir
            apply_autosense(config, model_dir)
        # Fall back to VARIANT_SPECS when autosense has not updated bridge_d_model.
        specs = VARIANT_SPECS.get(variant, VARIANT_SPECS["nano"])
        if config.student.bridge_d_model == ForgeConfig.default().student.bridge_d_model:
            config.student.bridge_d_model = specs["d_model"]
        self.config: ForgeConfig = config
        self.student_config: StudentConfig = config.student
        self.variant: str = variant

    def profile_params(self) -> list[ComponentProfile]:
        """Return per-component parameter and memory profiles (5 entries)."""
        cfg = self.student_config
        specs = VARIANT_SPECS.get(self.variant, VARIANT_SPECS["nano"])
        flops_est = estimate_flops(cfg)
        d_vision: int = cfg.bridge_d_vision
        d_model: int = cfg.bridge_d_model
        n_queries: int = cfg.bridge_n_queries
        n_layers: int = cfg.bridge_n_layers
        lm_layers: int = specs["lm_layers"]
        n_target: int = len(cfg.lora_target_modules)
        n_blocks: int = cfg.action_head_layers

        def _cp(name: str, params: int, trainable: int, in_s: str, out_s: str, flops: int) -> ComponentProfile:
            return ComponentProfile(
                name=name,
                param_count=params,
                trainable_params=trainable,
                frozen_params=params - trainable,
                input_shape=in_s,
                output_shape=out_s,
                estimated_flops=flops,
                estimated_memory_mb=params * 2 / 1e6,
            )

        vision_params = 400_000_000
        bridge_params = n_layers * 12 * d_model**2 + d_vision * d_model + n_queries * d_model
        lm_params = specs["lm_params"]
        lora_params = n_target * lm_layers * 2 * d_model * cfg.lora_rank
        action_params = n_blocks * (256 * 256 * 2 + 256 * d_model) + d_model * 256 + 256 * cfg.action_dim + 256 * 4

        lm_io = f"(B, {n_queries}, {d_model})"
        return [
            _cp(
                "vision_encoder",
                vision_params,
                0,
                "(B, 3, 384, 384)",
                f"(B, 729, {d_vision})",
                flops_est.vision_encoder,
            ),
            _cp(
                "bridge",
                bridge_params,
                bridge_params,
                f"(B, 729, {d_vision})",
                f"(B, {n_queries}, {d_model})",
                flops_est.bridge_attention,
            ),
            _cp("language", lm_params, 0, lm_io, lm_io, flops_est.language_backbone),
            _cp("lora", lora_params, lora_params, lm_io, lm_io, flops_est.lora_adapters),
            _cp("action_head", action_params, action_params, lm_io, f"(B, {cfg.action_dim})", flops_est.action_head),
        ]

    def estimate_flops(self) -> FLOPsEstimate:
        """Delegate to :func:`~forge.profiler.flops.estimate_flops`."""
        return estimate_flops(self.student_config)

    def estimate_vram(self, gpu_vram_gb: float = 24.0) -> VRAMEstimate:
        """Delegate to :func:`~forge.profiler.vram.estimate_vram`."""
        return estimate_vram(self.student_config, gpu_vram_gb)

    def recommend_hyperparams(
        self,
        dataset_size: int = 50000,
        gpu_vram_gb: float = 24.0,
        objective: str = "balanced",
    ) -> RecommendedHyperparams:
        """Delegate to :func:`~forge.profiler.recommend.recommend_hyperparams`."""
        return recommend_hyperparams(self.student_config, dataset_size, gpu_vram_gb, objective)

    def generate_card(
        self,
        dataset_size: int = 50000,
        gpu_vram_gb: float = 24.0,
    ) -> ModelProfileCard:
        """Build and return a complete :class:`~forge.profiler.dataclasses.ModelProfileCard`."""
        cfg = self.student_config
        components = self.profile_params()
        flops = self.estimate_flops()
        vram = self.estimate_vram(gpu_vram_gb)
        hp = self.recommend_hyperparams(dataset_size, gpu_vram_gb)
        total_params: int = sum(c.param_count for c in components)
        trainable_params: int = sum(c.trainable_params for c in components)
        frozen_params: int = sum(c.frozen_params for c in components)
        card = ModelProfileCard(
            model_name=f"FORGE-{self.variant.capitalize()}",
            variant=self.variant,
            vision_encoder=cfg.vision_encoder,
            language_model=cfg.language_model,
            action_head_type=cfg.action_head_type,
            action_dim=cfg.action_dim,
            action_horizon=cfg.action_horizon,
            components=components,
            total_params=total_params,
            trainable_params=trainable_params,
            frozen_params=frozen_params,
            flops=flops,
            vram=vram,
            recommended_hp=hp,
            fp32_size_mb=total_params * 4 / 1e6,
            fp16_size_mb=total_params * 2 / 1e6,
            int8_size_mb=total_params * 1 / 1e6,
            int4_size_mb=total_params * 0.5 / 1e6,
            bridge_config={
                "d_vision": cfg.bridge_d_vision,
                "d_model": cfg.bridge_d_model,
                "n_queries": cfg.bridge_n_queries,
                "n_heads": cfg.bridge_n_heads,
                "n_layers": cfg.bridge_n_layers,
            },
            architecture_diagram="",
            timestamp=datetime.now(UTC).isoformat(),
        )
        card.architecture_diagram = generate_ascii_diagram(card)
        return card

    def generate_markdown(self, card: ModelProfileCard) -> str:
        """Delegate to :func:`~forge.profiler.markdown.generate_markdown`."""
        return generate_markdown(card)
