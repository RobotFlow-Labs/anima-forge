# FORGE Configuration Reference

All configuration is managed through dataclasses in `src/forge/config.py`. Configuration can be loaded from YAML files, constructed programmatically, or overridden via environment variables.

---

## Config Hierarchy

```
ForgeConfig (master)
├── paths: ModelPaths              # File system paths
├── teacher: TeacherConfig         # PRD-01 label generation
├── student: StudentConfig         # PRD-02 student architecture
├── vision: VisionConfig           # PRD-10 multi-encoder
├── distill: DistillConfig         # PRD-03 knowledge distillation
├── pruning: PruningConfig         # PRD-04 layer pruning
├── quant: QuantConfig             # PRD-05 quantization
├── export: ExportConfig           # PRD-06 runtime export
├── web: WebConfig                 # PRD-20 web dashboard
├── universal: UniversalDistillConfig  # PRD-21 multi-teacher
└── curriculum: CurriculumConfig   # PRD-22 curriculum learning
```

---

## Loading Configuration

### From YAML

```python
from forge.config import ForgeConfig, apply_student_variant

config = ForgeConfig.from_yaml("configs/forge_nano.yaml")
```

### Default Values

```python
config = ForgeConfig.default()
```

### Programmatic Override

```python
config = ForgeConfig.default()
apply_student_variant(config.student, "small")
config.distill.max_steps = 100000
config.distill.learning_rate = 1e-4
```

### Environment variables

`FORGE_MODEL_DIR` overrides `paths.model_dir`:

```bash
export FORGE_MODEL_DIR=./models
```

This is applied automatically during `ForgeConfig.from_yaml()`.

| Variable | Purpose |
|---|---|
| `FORGE_MODEL_DIR` | Local Hugging Face-style model asset root |
| `FORGE_TEACHER_DATASET` | Real robot episode/teacher-label input selected by config loading |
| `FORGE_DATASET_DIR` / `FORGE_DATA_DIR` | Dataset discovery fallback used by doctor and label generation |
| `FORGE_OUTPUT_DIR` | Output-space check used by doctor |
| `FORGE_DEVICE` | Default runtime device (`cuda`, `cpu`, or `auto`) |
| `FORGE_REQUIRE_GPU` | Reject automatic CPU selection when enabled |
| `FORGE_ALLOW_CPU_FALLBACK` | Explicitly permit CUDA→CPU fallback |
| `FORGE_ALLOW_MOCK` | Explicit test-only permission for mock weights/labels; outputs are stamped mock |
| `FORGE_CONFIG_HOME` | First-run/completion state directory (default `~/.config/forge`) |
| `FORGE_CLI_LOG_FILE` | Rotating CLI log destination |
| `FORGE_CLI_LOG_JSON` | JSON-formatted CLI logs when enabled |
| `FORGE_OPENVLA_UNNORM_KEY` | OpenVLA action denormalization key |
| `FORGE_GIT_SHA` | Explicit source revision stamp when Git metadata is unavailable |

Boolean flags accept `1`, `true`, `yes`, or `on`. Mock and CPU-fallback flags are never
enabled implicitly by production commands.

---

## YAML Format

### Full Example: `configs/forge_nano.yaml`

```yaml
paths:
  model_dir: /path/to/models
  teacher: openvla--openvla-7b
  vision_encoder: google--siglip2-so400m-patch14-384
  language_model: Qwen--Qwen3-0.6B
  output_dir: ./outputs
  data_dir: ./data

teacher:
  benchmark: libero_spatial
  batch_size: 4
  save_attention: false
  save_vision_features: true
  episodes_per_task: 50
  max_steps_per_episode: 200

student:
  variant: nano
  vision_encoder: google/siglip2-so400m-patch14-384
  language_model: Qwen/Qwen3-0.6B
  bridge_d_vision: 1152
  bridge_d_model: 1024
  bridge_n_queries: 64
  bridge_n_heads: 8
  bridge_n_layers: 4
  action_dim: 7
  action_head_layers: 4
  action_diffusion_steps: 10
  lora_rank: 32
  lora_alpha: 64
  lora_target_modules:
    - q_proj
    - v_proj
    - k_proj
    - o_proj
  action_horizon: 1
  chunk_overlap: 2
  action_head_type: diffusion
  flow_inference_steps: 4
  autosense: true

vision:
  encoders:
    - siglip2-so400m
  fusion_method: attention_pool
  d_output: 1152

distill:
  learning_rate: 2e-4
  weight_decay: 0.01
  warmup_steps: 500
  max_steps: 50000
  batch_size: 16
  gradient_accumulation_steps: 4
  temperature: 4.0
  alpha_kd: 0.4
  alpha_task: 0.3
  alpha_feat: 0.2
  alpha_action: 0.1
  eval_every: 1000
  save_every: 2000

pruning:
  target_layers: 8
  calibration_samples: 500
  recovery_lr: 5e-5
  recovery_steps: 5000
  keep_first_n: 2
  keep_last_n: 2

quant:
  target_avg_bits: 4.0
  min_bits: 2
  max_bits: 8
  calibration_samples: 200
  post_quant_finetune_steps: 1000

export:
  formats:
    - onnx
    - mlx
  tensorrt_precision: int8
  tensorrt_workspace_mb: 2048
  coreml_min_target: macOS14
  onnx_opset: 17

web:
  host: 127.0.0.1
  port: 3000
  cors_origins:
    - http://localhost:3000
  auto_open_browser: true

universal:
  teacher_names:
    - openvla-7b
    - rdt2-fm
    - smolvla-base
  gpu_assignment: {}
  router_temperature: 1.0
  use_gumbel: true
  max_steps: 100000
  batch_size: 8
  staged: false
  teachers_per_stage: 3
  steps_per_stage: 25000
  alpha_task: 0.3
  alpha_diversity: 0.05
  alpha_consistency: 0.1

curriculum:
  enabled: true
  difficulty_metric: loss
  initial_difficulty: 0.3
  final_difficulty: 1.0
  ramp_steps: 50000
  ramp_schedule: linear
  plateau_window: 2000
  plateau_threshold: 0.01
  plateau_lr_factor: 0.5
  max_plateaus: 3
  teacher_dropout: false
  teacher_dropout_start: 0.0
  teacher_dropout_end: 0.3
  teacher_dropout_ramp_steps: 30000
  hard_example_mining: true
  hard_example_ratio: 0.3
  loss_history_size: 10000
```

---

## Config Dataclass Reference

### ModelPaths

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_dir` | `str` | `./models` | Base directory for model weights |
| `teacher` | `str` | `openvla--openvla-7b` | Teacher model subdirectory |
| `vision_encoder` | `str` | `google--siglip2-so400m-patch14-384` | Vision encoder subdirectory |
| `language_model` | `str` | `Qwen--Qwen3-0.6B` | Language model subdirectory |
| `output_dir` | `str` | `./outputs` | Pipeline output directory |
| `data_dir` | `str` | `./data` | Teacher label data directory |

**Computed properties**: `teacher_path`, `vision_encoder_path`, `language_model_path` return `Path(model_dir) / name`.

### TeacherConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `benchmark` | `str` | `libero_spatial` | Benchmark suite for label generation |
| `batch_size` | `int` | `4` | Teacher inference batch size |
| `save_attention` | `bool` | `False` | Save attention maps |
| `save_vision_features` | `bool` | `True` | Save vision encoder features |
| `episodes_per_task` | `int` | `50` | Episodes per benchmark task |
| `max_steps_per_episode` | `int` | `200` | Max timesteps per episode |

### StudentConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `variant` | `str` | `nano` | Student variant: `micro`, `nano`, `small`, `medium` |
| `vision_encoder` | `str` | `google/siglip2-so400m-patch14-384` | HF model ID |
| `language_model` | `str` | `Qwen/Qwen3-0.6B` | HF model ID |
| `bridge_d_vision` | `int` | `1152` | Vision encoder output dimension |
| `bridge_d_model` | `int` | `1024` | Language model hidden dimension |
| `bridge_n_queries` | `int` | `64` | Number of compressed vision tokens |
| `bridge_n_heads` | `int` | `8` | Attention heads in bridge |
| `bridge_n_layers` | `int` | `4` | Cross-attention layers in bridge |
| `action_dim` | `int` | `7` | Robot action dimensions |
| `action_head_layers` | `int` | `4` | MLP layers in action head |
| `action_diffusion_steps` | `int` | `10` | DDPM denoising steps |
| `lora_rank` | `int` | `32` | LoRA adapter rank |
| `lora_alpha` | `int` | `64` | LoRA alpha scaling |
| `lora_target_modules` | `list[str]` | `[q_proj, v_proj, k_proj, o_proj]` | LoRA target layers |
| `action_horizon` | `int` | `1` | Action chunk size (H=1 for v1 compat) |
| `chunk_overlap` | `int` | `2` | Overlap for chunk blending |
| `action_head_type` | `str` | `diffusion` | `diffusion`, `flow`, `chunk`, `consistency` |
| `flow_inference_steps` | `int` | `4` | ODE steps for flow head |
| `autosense` | `bool` | `True` | Auto-detect dimensions from config.json |

### DistillConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `learning_rate` | `float` | `2e-4` | Base learning rate |
| `weight_decay` | `float` | `0.01` | AdamW weight decay |
| `warmup_steps` | `int` | `500` | Linear warmup steps |
| `max_steps` | `int` | `50000` | Maximum training steps |
| `batch_size` | `int` | `16` | Training batch size |
| `gradient_accumulation_steps` | `int` | `4` | Gradient accumulation |
| `temperature` | `float` | `4.0` | KD temperature (higher = softer) |
| `alpha_kd` | `float` | `0.4` | KD loss weight |
| `alpha_task` | `float` | `0.3` | Task (GT) loss weight |
| `alpha_feat` | `float` | `0.2` | Feature alignment loss weight |
| `alpha_action` | `float` | `0.1` | Action distribution loss weight |
| `eval_every` | `int` | `1000` | Evaluation frequency |
| `save_every` | `int` | `2000` | Checkpoint frequency |

### PruningConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `target_layers` | `int` | `8` | Target remaining layers |
| `calibration_samples` | `int` | `500` | Samples for importance scoring |
| `recovery_lr` | `float` | `5e-5` | Recovery fine-tuning LR |
| `recovery_steps` | `int` | `5000` | Recovery fine-tuning steps |
| `keep_first_n` | `int` | `2` | Protect first N layers |
| `keep_last_n` | `int` | `2` | Protect last N layers |

### QuantConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `target_avg_bits` | `float` | `4.0` | Target average precision |
| `min_bits` | `int` | `2` | Minimum per-layer bits |
| `max_bits` | `int` | `8` | Maximum per-layer bits |
| `calibration_samples` | `int` | `200` | Quantization calibration samples |
| `post_quant_finetune_steps` | `int` | `1000` | Recovery after quantization |

### ExportConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `formats` | `list[str]` | `[onnx, mlx]` | Export formats |
| `tensorrt_precision` | `str` | `int8` | TensorRT precision |
| `tensorrt_workspace_mb` | `int` | `2048` | TensorRT workspace memory |
| `coreml_min_target` | `str` | `macOS14` | CoreML minimum target |
| `onnx_opset` | `int` | `17` | ONNX opset version |

### WebConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `host` | `str` | `127.0.0.1` | Web server host |
| `port` | `int` | `3000` | Web server port |
| `cors_origins` | `list[str]` | `[http://localhost:3000]` | CORS allowed origins |
| `auto_open_browser` | `bool` | `True` | Auto-open browser on launch |

### UniversalDistillConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `teacher_names` | `list[str]` | `[openvla-7b, rdt2-fm, smolvla-base]` | Teacher models |
| `gpu_assignment` | `dict` | `{}` | Teacher-to-GPU mapping |
| `router_temperature` | `float` | `1.0` | Router softmax temperature |
| `use_gumbel` | `bool` | `True` | Gumbel softmax in training |
| `max_steps` | `int` | `100000` | Maximum training steps |
| `batch_size` | `int` | `8` | Batch size |
| `staged` | `bool` | `False` | Staged teacher rotation |
| `teachers_per_stage` | `int` | `3` | Teachers per stage |
| `steps_per_stage` | `int` | `25000` | Steps per stage |
| `alpha_task` | `float` | `0.3` | Task loss weight |
| `alpha_diversity` | `float` | `0.05` | Diversity loss weight |
| `alpha_consistency` | `float` | `0.1` | Consistency loss weight |

### CurriculumConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` | `True` | Enable curriculum learning |
| `difficulty_metric` | `str` | `loss` | `loss`, `confidence`, `teacher_disagreement` |
| `initial_difficulty` | `float` | `0.3` | Start with easiest 30% of data |
| `final_difficulty` | `float` | `1.0` | Ramp to 100% of data |
| `ramp_steps` | `int` | `50000` | Steps for difficulty ramp |
| `ramp_schedule` | `str` | `linear` | `linear`, `cosine`, `step` |
| `plateau_window` | `int` | `2000` | Window for plateau detection |
| `plateau_threshold` | `float` | `0.01` | Min improvement threshold |
| `plateau_lr_factor` | `float` | `0.5` | LR multiplier on plateau |
| `max_plateaus` | `int` | `3` | Max plateau reductions |
| `teacher_dropout` | `bool` | `False` | Enable teacher dropout |
| `teacher_dropout_start` | `float` | `0.0` | Initial dropout rate |
| `teacher_dropout_end` | `float` | `0.3` | Final dropout rate |
| `teacher_dropout_ramp_steps` | `int` | `30000` | Dropout ramp steps |
| `hard_example_mining` | `bool` | `True` | Enable hard example mining |
| `hard_example_ratio` | `float` | `0.3` | Fraction of batch from hard examples |
| `loss_history_size` | `int` | `10000` | Track this many per-sample losses |

---

## Variant Presets

### Micro

```yaml
student:
  variant: micro
  language_model: HuggingFaceTB/SmolLM2-135M
  # AutoSense -> bridge_d_model: 576
```

### Nano (default)

```yaml
student:
  variant: nano
  language_model: Qwen/Qwen3-0.6B
  # canonical bridge_d_model: 1024
```

### Small

```yaml
student:
  variant: small
  language_model: Qwen/Qwen3-1.7B
  # canonical bridge_d_model: 2048
```

### Medium

```yaml
student:
  variant: medium
  language_model: Qwen/Qwen3-4B
  backbone_dtype: bfloat16
  # canonical bridge_d_model: 2560
```

---

## AutoSense

When `autosense: true` (default), FORGE reads `config.json` from model directories at load time and auto-populates dimension fields. This eliminates config mismatches when switching between models.

**What gets auto-detected**:
- `bridge_d_vision`: From vision encoder's `hidden_size`
- `bridge_d_model`: From language model's `hidden_size`
- `n_tokens`: Computed from `image_size / patch_size`

**Disable**: Set `student.autosense: false` in YAML or `config.student.autosense = False` in Python.

**CLI check**:
```bash
forge autosense --model-dir /path/to/models
```

---

## Environment Variables

| Variable | Effect |
|----------|--------|
| `FORGE_MODEL_DIR` | Overrides `paths.model_dir` |
| `FORGE_DEVICE` | Forces backend: `cuda`, `mlx`, `cpu` |
| `HF_TOKEN` | HuggingFace token for gated model access |

---

## Available Config Files

| File | Variant | Notes |
|------|---------|-------|
| `configs/forge_nano.yaml` | Nano (0.5B) | Default configuration |
| `configs/forge_small.yaml` | Small (1.5B) | Larger backbone |
| `configs/forge_cuda.yaml` | Nano | CUDA-specific settings |
| `configs/eval/forge_student.yaml` | -- | Eval model server config |
| `configs/eval/libero_forge.yaml` | -- | LIBERO benchmark config |
| `configs/eval/simpler_forge.yaml` | -- | SimplerEnv benchmark config |
| `configs/eval/vlabench_forge.yaml` | -- | VLABench benchmark config |

---

## Config Override Mechanism

YAML loading uses recursive override via `_apply_overrides()`:

```python
def _apply_overrides(config: Any, data: dict) -> None:
    """Recursively apply dict overrides to dataclass."""
    for key, value in data.items():
        if hasattr(config, key):
            attr = getattr(config, key)
            if isinstance(value, dict) and hasattr(attr, "__dataclass_fields__"):
                _apply_overrides(attr, value)
            else:
                setattr(config, key, value)
```

This means you only need to specify fields you want to override in YAML. All other fields retain their default values.

### Partial YAML

```yaml
# Only override what you need
student:
  variant: small
  lora_rank: 64

distill:
  max_steps: 100000
  learning_rate: 1e-4
```

All unspecified fields use their defaults from the dataclass definitions.
