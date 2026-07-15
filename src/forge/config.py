"""FORGE configuration management.

Loads YAML configs and provides typed access to all pipeline settings.
Supports environment variable overrides via FORGE_ prefix.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

STUDENT_VARIANT_PRESETS: dict[str, dict[str, Any]] = {
    "micro": {
        "vision_encoder": "google/siglip2-so400m-patch14-384",
        "language_model": "HuggingFaceTB/SmolLM2-135M",
        "bridge_d_model": 576,
        "bridge_n_heads": 8,
        "bridge_n_layers": 3,
        "action_head_layers": 3,
        "lora_rank": 16,
        "lora_alpha": 32,
    },
    "nano": {
        "vision_encoder": "google/siglip2-so400m-patch14-384",
        "language_model": "Qwen/Qwen3-0.6B",
        "bridge_d_model": 1024,
        "bridge_n_heads": 8,
        "bridge_n_layers": 4,
        "action_head_layers": 4,
        "lora_rank": 32,
        "lora_alpha": 64,
    },
    "small": {
        "vision_encoder": "google/siglip2-so400m-patch14-384",
        "language_model": "Qwen/Qwen3-1.7B",
        "bridge_d_model": 2048,
        "bridge_n_heads": 16,
        "bridge_n_layers": 6,
        "action_head_layers": 6,
        "lora_rank": 64,
        "lora_alpha": 128,
    },
    "medium": {
        "vision_encoder": "google/siglip2-so400m-patch14-384",
        "language_model": "Qwen/Qwen3-4B",
        "backbone_dtype": "bfloat16",
        "bridge_d_model": 2560,
        "bridge_n_heads": 16,
        "bridge_n_layers": 4,
        "action_head_layers": 4,
        "lora_rank": 64,
        "lora_alpha": 128,
    },
}


def _env_enabled(name: str) -> bool:
    """Return whether a FORGE boolean environment flag is enabled."""
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _coerce_bool(value: Any, *, name: str) -> bool:
    """Coerce common config boolean spellings without truthy-string bugs."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


@dataclass
class ModelPaths:
    """Paths to model weights and datasets."""

    model_dir: str = os.environ.get("FORGE_MODEL_DIR", "./models")
    teacher: str = "openvla--openvla-7b"
    vision_encoder: str = "google--siglip2-so400m-patch14-384"
    language_model: str = "Qwen--Qwen3-0.6B"
    output_dir: str = "./outputs"
    data_dir: str = "./data"

    @property
    def teacher_path(self) -> Path:
        return Path(self.model_dir) / self.teacher

    @property
    def vision_encoder_path(self) -> Path:
        return Path(self.model_dir) / self.vision_encoder

    @property
    def language_model_path(self) -> Path:
        return Path(self.model_dir) / self.language_model


@dataclass
class TeacherConfig:
    """PRD-01: Teacher label generation config."""

    adapter: str = "openvla-7b"
    dataset: str = "HuggingFaceVLA--smol-libero"
    benchmark: str = "libero_spatial"
    batch_size: int = 4
    save_attention: bool = False
    save_vision_features: bool = True
    episodes_per_task: int = 50
    max_steps_per_episode: int = 200


@dataclass
class StudentConfig:
    """PRD-02: Student architecture config."""

    variant: str = "nano"  # micro, nano, small, medium
    vision_encoder: str = "google/siglip2-so400m-patch14-384"
    language_model: str = "Qwen/Qwen3-0.6B"
    backbone_dtype: str = "auto"  # auto follows the local HF config (bf16 for v3 backbones)
    bridge_d_vision: int = 1152
    bridge_d_model: int = 1024
    bridge_n_queries: int = 64
    bridge_n_heads: int = 8
    bridge_n_layers: int = 4
    action_dim: int = 7
    action_head_layers: int = 4
    action_diffusion_steps: int = 10
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])

    # v2: Action Chunking (PRD-09)
    action_horizon: int = 1  # H=1 is v1 compat, H=8 for chunking
    chunk_overlap: int = 2  # Overlap for blending at inference

    # v2: Action Head Selection (PRD-11)
    action_head_type: str = "diffusion"  # "diffusion" | "flow" | "chunk"
    flow_inference_steps: int = 4  # ODE steps for flow head (1, 2, 4)

    # v2: AutoSense (PRD-25)
    autosense: bool = True  # Auto-detect model dimensions from config.json

    # v3: production never substitutes synthetic backbones unless explicitly allowed.
    allow_mock: bool = field(default_factory=lambda: _env_enabled("FORGE_ALLOW_MOCK"))

    def __post_init__(self) -> None:
        self.allow_mock = _coerce_bool(self.allow_mock, name="student.allow_mock")
        if self.backbone_dtype not in {"auto", "float32", "float16", "bfloat16"}:
            raise ValueError(
                f"student.backbone_dtype must be one of auto, float32, float16, bfloat16, got {self.backbone_dtype!r}"
            )


@dataclass
class VisionConfig:
    """PRD-10: Multi-encoder vision config."""

    encoders: list[str] = field(default_factory=lambda: ["siglip2-so400m"])
    fusion_method: str = "attention_pool"  # "attention_pool" | "concat" | "average"
    d_output: int = 1152


@dataclass
class DistillConfig:
    """PRD-03: Knowledge distillation config."""

    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_steps: int = 50000
    batch_size: int = 16
    gradient_accumulation_steps: int = 4
    temperature: float = 4.0
    alpha_kd: float = 0.4
    alpha_task: float = 0.3
    alpha_feat: float = 0.2
    alpha_action: float = 0.1
    eval_every: int = 1000
    save_every: int = 2000


@dataclass
class PruningConfig:
    """PRD-04: Layer pruning config."""

    target_layers: int = 8
    calibration_samples: int = 500
    recovery_lr: float = 5e-5
    recovery_steps: int = 5000
    keep_first_n: int = 2
    keep_last_n: int = 2


@dataclass
class QuantConfig:
    """PRD-05: Quantization config."""

    method: str = "qvla"
    bits: int = 4
    target_avg_bits: float = 4.0
    min_bits: int = 2
    max_bits: int = 8
    group_size: int = 128
    seed: int = 42
    calibration_samples: int = 200
    benchmark_samples: int = 256
    post_quant_finetune_steps: int = 1000


@dataclass
class ExportConfig:
    """PRD-06: Runtime export config."""

    formats: list[str] = field(default_factory=lambda: ["onnx", "mlx"])
    tensorrt_precision: str = "int8"
    tensorrt_workspace_mb: int = 2048
    coreml_min_target: str = "macOS14"
    onnx_opset: int = 19


@dataclass
class WebConfig:
    """PRD-20: Web dashboard config."""

    host: str = "127.0.0.1"
    port: int = 3000
    cors_origins: list[str] = field(default_factory=lambda: ["http://localhost:3000"])
    auto_open_browser: bool = True


@dataclass
class UniversalDistillConfig:
    """PRD-21: Universal teacher ensemble distillation config."""

    teacher_names: list[str] = field(
        default_factory=lambda: [
            "openvla-7b",
            "rdt2-fm",
            "smolvla-base",
        ]
    )
    gpu_assignment: dict[str, int] = field(default_factory=dict)  # teacher→GPU id
    router_temperature: float = 1.0
    use_gumbel: bool = True
    max_steps: int = 100000
    batch_size: int = 8
    staged: bool = False
    teachers_per_stage: int = 3
    steps_per_stage: int = 25000
    alpha_task: float = 0.3
    alpha_diversity: float = 0.05
    alpha_consistency: float = 0.1


@dataclass
class CurriculumConfig:
    """PRD-22: Curriculum learning & adaptive training config."""

    enabled: bool = True
    # Difficulty scoring
    difficulty_metric: str = "loss"  # "loss" | "confidence" | "teacher_disagreement"
    initial_difficulty: float = 0.3  # Start with easiest 30% of data
    final_difficulty: float = 1.0  # Ramp to all data
    ramp_steps: int = 50000  # Steps to go from initial → final
    ramp_schedule: str = "linear"  # "linear" | "cosine" | "step"

    # Loss plateau detection
    plateau_window: int = 2000  # Window for detecting plateaus
    plateau_threshold: float = 0.01  # Min improvement to NOT be a plateau
    plateau_lr_factor: float = 0.5  # LR multiplied by this on plateau
    max_plateaus: int = 3  # Max plateau-triggered reductions

    # Teacher dropout scheduling
    teacher_dropout: bool = False
    teacher_dropout_start: float = 0.0  # Start dropout rate
    teacher_dropout_end: float = 0.3  # End dropout rate
    teacher_dropout_ramp_steps: int = 30000

    # Loss-aware sampling
    hard_example_mining: bool = True
    hard_example_ratio: float = 0.3  # Fraction of batch from hard examples
    loss_history_size: int = 10000  # Track this many per-sample losses


@dataclass
class ForgeConfig:
    """Master config combining all stages."""

    paths: ModelPaths = field(default_factory=ModelPaths)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    student: StudentConfig = field(default_factory=StudentConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    distill: DistillConfig = field(default_factory=DistillConfig)
    pruning: PruningConfig = field(default_factory=PruningConfig)
    quant: QuantConfig = field(default_factory=QuantConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    web: WebConfig = field(default_factory=WebConfig)
    universal: UniversalDistillConfig = field(default_factory=UniversalDistillConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ForgeConfig:
        """Load config from YAML file with env var overrides."""
        source_path = Path(path).expanduser().resolve()
        source_bytes = source_path.read_bytes()
        data = yaml.safe_load(source_bytes.decode("utf-8")) or {}

        config = cls()
        setattr(config, "_forge_config_path", str(source_path))
        setattr(config, "_forge_config_sha256", hashlib.sha256(source_bytes).hexdigest())

        # A variant-only YAML must select the complete canonical architecture;
        # explicit fields in the same YAML are then applied as intentional
        # overrides for backward compatibility and experimentation.
        student_data = data.get("student") if isinstance(data, dict) else None
        if isinstance(student_data, dict) and "variant" in student_data:
            apply_student_variant(config.student, str(student_data["variant"]))

        # Apply YAML overrides
        _apply_overrides(config, data)
        # Override model_dir from env (takes precedence for portability)
        env_model_dir = os.environ.get("FORGE_MODEL_DIR")
        if env_model_dir:
            config.paths.model_dir = env_model_dir
        if _env_enabled("FORGE_ALLOW_MOCK"):
            config.student.allow_mock = True
        if env_teacher_dataset := os.environ.get("FORGE_TEACHER_DATASET"):
            config.teacher.dataset = env_teacher_dataset

        return config

    @classmethod
    def default(cls) -> ForgeConfig:
        """Return default config."""
        return cls()


def apply_student_variant(config: StudentConfig, variant: str) -> StudentConfig:
    """Apply the canonical v3 architecture for a named student variant."""
    if variant not in STUDENT_VARIANT_PRESETS:
        raise ValueError(f"Unknown student variant {variant!r}; choose {sorted(STUDENT_VARIANT_PRESETS)}")
    config.variant = variant
    for key, value in STUDENT_VARIANT_PRESETS[variant].items():
        setattr(config, key, value)
    return config


def apply_checkpoint_student_config(
    config: StudentConfig,
    checkpoint: dict[str, Any],
) -> StudentConfig:
    """Apply architecture metadata saved in a training checkpoint."""
    saved = checkpoint.get("student_config")
    if not isinstance(saved, dict):
        saved = checkpoint.get("hp")
    if not isinstance(saved, dict):
        return config
    for key, value in saved.items():
        if hasattr(config, key):
            setattr(config, key, value)
    config.__post_init__()
    return config


def _apply_overrides(config: Any, data: dict) -> None:
    """Recursively apply dict overrides to dataclass."""
    for key, value in data.items():
        if hasattr(config, key):
            attr = getattr(config, key)
            if isinstance(value, dict) and hasattr(attr, "__dataclass_fields__"):
                _apply_overrides(attr, value)
            else:
                if isinstance(attr, bool):
                    value = _coerce_bool(value, name=key)
                setattr(config, key, value)
