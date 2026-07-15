# FORGE Pipeline Reference

The FORGE pipeline turns registered VLA teacher outputs into trained, compressed, and
runtime-validated student artifacts in four stages. Source: `src/forge/pipeline.py`.

```
Stage 1            Stage 2            Stage 3            Stage 4
Teacher Labels --> Knowledge       --> Compression    --> Export +
(PRD-01)           Distillation        (PRD-04/05)       Validation
                   (PRD-02/03)                           (PRD-06/07)
```

---

## Running the Pipeline

```bash
# Full pipeline
forge pipeline --device cuda

# With custom config
forge pipeline --config configs/forge_nano.yaml --device cuda

# Single stage
forge pipeline --stage distill --device cuda --max-steps 5000

# Skip labels (reuse existing)
forge pipeline --skip-labels --device cuda
```

### Pipeline API

```python
from forge.config import ForgeConfig
from forge.pipeline import run_pipeline

config = ForgeConfig.from_yaml("configs/forge_nano.yaml")
results = run_pipeline(
    config,
    device="cuda",
    skip_labels=False,
    stage=None,           # None = all stages; "labels", "distill", "compress", "export", "validate"
    max_distill_steps=50000,
    max_recovery_steps=5000,
)
# results: dict with per-stage results + total_time_seconds
```

---

## Stage 1: Teacher Label Generation (PRD-01)

**Source**: `src/forge/teacher.py` -- `generate_teacher_labels()`

Generates soft labels from a teacher VLA model on benchmark tasks. Captures action logits, vision features, and confidence scores.

### What Gets Generated

For each episode (T timesteps):
- `images`: (T, H, W, 3) uint8 camera observations
- `proprioception`: (T, D_proprio) robot joint states
- `teacher_action_logits`: (T, D_action) soft action labels
- `teacher_action_mean`: (T, D_action) action distribution mean
- `teacher_action_std`: (T, D_action) action distribution std
- `teacher_vision_features`: (T, N_tokens, D_vision) optional vision features
- `confidence`: (T, D_action) per-dimension confidence (1 / (1 + std))
- `ground_truth_actions`: (T, D_action) demonstration actions

### Config

```yaml
teacher:
  benchmark: libero_spatial       # Benchmark suite
  batch_size: 4                   # Inference batch size
  save_attention: false           # Save attention maps
  save_vision_features: true      # Save vision encoder features
  episodes_per_task: 50           # Episodes per benchmark task
  max_steps_per_episode: 200      # Max timesteps per episode
```

### Storage

Labels are written to HDF5 files via `LabelWriter` and read back by `TeacherLabelDataset`:

```
data/teacher_labels/
├── labels_0000.h5               # 50 episodes per file
├── labels_0001.h5
├── ...
└── metadata.json                # Schema version, stats
```

### Supported Teachers

| Teacher | Loader | Notes |
|---------|--------|-------|
| OpenVLA-7B | `AutoModelForVision2Seq` | Primary teacher, token-AR |
| RDT2 | Official RDT runner + Qwen-VL | Native 24×20 contract and required normalizer |
| SmolVLA | LeRobot policy | Flow policy with native action chunks |
| MolmoAct2 | LeRobot policy | Hybrid autoregressive/flow policy |
| VLA-JEPA | LeRobot policy | Qwen3-VL + flow-DiT policy |

All adapters load local assets only. Missing companions raise an actionable model error;
there is no random teacher fallback.

### Forward Hooks

`ExtractionHooks` class registers PyTorch forward hooks on the teacher's vision encoder to capture intermediate representations without modifying the teacher model.

```python
hooks = ExtractionHooks()
hooks.register(teacher, extract_vision=True)
# After forward pass:
vision_features = hooks.vision_features  # (B, N, D)
hooks.reset()
hooks.remove()
```

---

## Stage 2: Knowledge Distillation (PRD-02/03)

**Source**: `src/forge/distill.py` -- `train_forge()`

Trains the FORGE student using teacher soft labels with a composite loss function.

### Loss Function

**Source**: `src/forge/losses.py` -- `ForgeDistillationLoss`

```
L_total = alpha_kd * L_KD + alpha_task * L_task + alpha_feat * L_feat + alpha_action * L_action
```

| Component | Weight | Formula | Purpose |
|-----------|--------|---------|---------|
| `L_KD` | 0.4 | Temperature-scaled MSE on soft labels | Match teacher action predictions |
| `L_task` | 0.3 | MSE on ground truth actions | Direct supervision |
| `L_feat` | 0.2 | Cosine embedding loss on vision features | Align visual representations |
| `L_action` | 0.1 | Confidence-weighted action distribution loss | Focus on reliable dimensions |

For action-chunking models (H > 1), use `chunk_aware_kd_loss()` which applies exponential decay weighting across the horizon.

### Training Phases

| Phase | Steps | Trainable Parameters | Purpose |
|-------|-------|---------------------|---------|
| Phase 1 | 0 - 10% | Bridge + Action head | Warm up new modules |
| Phase 2 | 10% - 83% | Bridge + LoRA + Action head | Full distillation |
| Phase 3 | 83% - 100% | Action head only | Refine action predictions |

### Config

```yaml
distill:
  learning_rate: 2e-4
  weight_decay: 0.01
  warmup_steps: 500
  max_steps: 50000
  batch_size: 16
  gradient_accumulation_steps: 4
  temperature: 4.0                # Higher = softer matching
  alpha_kd: 0.4
  alpha_task: 0.3
  alpha_feat: 0.2
  alpha_action: 0.1
  eval_every: 1000
  save_every: 2000
```

### Checkpointing

```
outputs/checkpoints/
├── step_2000.pt                 # Periodic checkpoints
├── step_4000.pt
├── best.pt                      # Lowest loss checkpoint
└── final.pt                     # End-of-training checkpoint
```

Each checkpoint contains: `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `step`.

---

## Stage 3: Compression (PRD-04/05)

Two sequential compression steps: layer pruning then quantization.

### 3a: Layer Pruning (PRD-04)

**Source**: `src/forge/prune.py`, `src/forge/prune_v2.py`

Shallow-Pi pruning removes entire layers from the language backbone based on importance scores.

```python
from forge.prune import prune_layers
from forge.prune_v2 import compute_chunk_layer_importance

scores = compute_chunk_layer_importance(student, device="cuda")
pruned_student, removed = prune_layers(student, scores, config.pruning)
```

**Config**:
```yaml
pruning:
  target_layers: 8                # Target number of remaining layers
  calibration_samples: 500        # Samples for importance scoring
  recovery_lr: 5e-5               # LR for recovery fine-tuning
  recovery_steps: 5000            # Steps of recovery fine-tuning
  keep_first_n: 2                 # Always keep first N layers
  keep_last_n: 2                  # Always keep last N layers
```

For chunk-producing models (H > 1), use `compute_chunk_layer_importance()` from `prune_v2.py` which considers temporal coherence in addition to action accuracy.

### 3b: Quantization (PRD-05)

**Source**: `src/forge/quantize.py`, `src/forge/quantize_v2.py`

QVLA action-centric mixed-precision quantization.

```python
from forge.quantize import quantize_model, create_quant_profile

quantized = quantize_model(pruned_student, uniform_bits=4)
profile = create_quant_profile(quantized, {}, name="q4_nano")
# profile.avg_bits, profile.compressed_size_mb
```

**Config**:
```yaml
quant:
  target_avg_bits: 4.0            # Average target precision
  min_bits: 2                     # Minimum per-layer bits
  max_bits: 8                     # Maximum per-layer bits
  calibration_samples: 200        # Samples for quantization calibration
  post_quant_finetune_steps: 1000 # Recovery fine-tuning after quantization
```

For chunk models, `quantize_v2.quantize_chunk_aware()` ensures the action head always gets higher precision (8-bit) to prevent temporal jitter in multi-step predictions.

For a release comparison matrix, prune once and quantize every candidate from that exact
`pruned.pt` artifact. Mixing an unpruned checkpoint into the matrix changes both the model
and compression denominator, so those results are not comparable:

```bash
forge quantize run --checkpoint outputs/compressed/pruned.pt \
  --output outputs/quantized/qvla_8bit.pt --method qvla --bits 8 --device cuda
forge quantize run --checkpoint outputs/compressed/pruned.pt \
  --output outputs/quantized/turboquant_mse_4bit.pt \
  --method turboquant-mse --bits 4 --device cuda
forge quantize run --checkpoint outputs/compressed/pruned.pt \
  --output outputs/quantized/turboquant_mse_8bit.pt \
  --method turboquant-mse --bits 8 --device cuda
```

Release evidence must report `requested_device: cuda`, `device: cuda`, an empty
`fallbacks` list, and real vision, language, and label provenance. Do not enable a CPU
fallback for this matrix.

---

## Stage 4: Export + Validation (PRD-06/07)

### Export Formats

**Source**: `src/forge/export/`

| Format | File | Platform | Notes |
|--------|------|----------|-------|
| MLX | `weights.npz` + `config.json` | Apple Silicon | Always exported |
| ONNX | `forge.onnx` | Universal | ORT optimization optional |
| TensorRT | `forge.engine` | NVIDIA GPU | CUDA-only, INT8/FP16 |

#### MLX Export

```python
from forge.export.mlx_export import export_mlx
export_mlx(student, "./outputs/mlx", config={"variant": "nano"})
```

Output directory:
```
outputs/mlx/
├── weights.npz        # FP16 weights (compressed)
├── config.json        # Model architecture config
└── metadata.json      # Export metadata (param count, size)
```

#### ONNX Export

```python
from forge.export.onnx_export import export_onnx
export_onnx(student, "./outputs/forge.onnx", opset_version=17, optimize=True)
```

Dynamic axes for batch size and sequence length. Validation against PyTorch output available via `validate_onnx()`.

#### TensorRT Export

```python
from forge.export.tensorrt_export import export_tensorrt
export_tensorrt("forge.onnx", "forge.engine", precision="fp16", workspace_mb=2048)
```

Requires NVIDIA GPU + TensorRT SDK. Supports FP16 and INT8 (with calibration data).

### Validation

The pipeline runs `run_full_validation()` which checks:
- Model correctness (forward pass succeeds)
- Latency benchmarking (p50/p95/p99)
- Throughput measurement (FPS)
- Model size verification

---

## Production Training (PRD-23)

**Source**: `src/forge/trainer.py` -- `ProductionTrainer`

The production trainer extends basic distillation with:

### Curriculum Learning (PRD-22)

```python
from forge.curriculum import CurriculumSampler, PlateauDetector

# Difficulty ramps from 0.3 to 1.0 over ramp_steps
# Schedule options: linear, cosine, step
sampler = CurriculumSampler(dataset_size=10000, config=config.curriculum)
```

### Plateau Detection

Automatically reduces learning rate when training loss stalls:

```python
detector = PlateauDetector(window=2000, threshold=0.01, lr_factor=0.5, max_plateaus=3)
```

### Hard Example Mining

Re-samples high-loss examples more frequently:

```python
miner = HardExampleMiner(dataset_size=10000, hard_ratio=0.3, history_size=10000)
```

### Teacher Dropout (Multi-teacher only)

Progressive teacher dropping for robustness in multi-teacher setups:

```python
dropout = TeacherDropout(n_teachers=3, dropout_start=0.0, dropout_end=0.3, ramp_steps=30000)
```

### Full Production Training Example

```python
from forge.trainer import ProductionTrainer
from forge.config import ForgeConfig

config = ForgeConfig.default()
config.curriculum.enabled = True
config.curriculum.hard_example_mining = True

trainer = ProductionTrainer(
    student=student,
    dataset=dataset,
    loss_fn=loss_fn,
    config=config,
    device="cuda",
    n_teachers=3,
)
report = trainer.train(max_steps=50000, log_every=100, checkpoint_every=2000)
print(report.to_dict())
```

---

## Multi-Teacher Distillation (PRD-21)

**Source**: `src/forge/universal_distill.py`

Train a student from multiple teacher models simultaneously with learned routing.

```python
from forge.universal_distill import UniversalDistillationLoss, plan_gpu_placement

# Plan GPU placement
placement = plan_gpu_placement(
    teacher_names=["openvla-7b", "rdt2-fm", "smolvla-base"],
    n_gpus=4,
)

# Loss with confidence routing
loss_fn = UniversalDistillationLoss(n_teachers=3, d_student=896, confidence_dim=7)
result = loss_fn(student_actions, teacher_list, gt_actions, features, confidences)
```

**Loss formula**:
```
L_total = alpha_kd * sum(w_i * KD_i)
        + alpha_task * L_task
        + alpha_div * L_diversity
        + alpha_con * L_consistency
```

Staged mode trains on rotating subsets instead of keeping every teacher resident concurrently.

---

## Benchmark Results

FORGE does not embed stale hardware claims in this guide. Run the packaged suites on
the target machine with the required real LeRobot data and local weights:

```bash
forge benchmark all --device cuda --data-dir models/datasets/lerobot--pusht
```

The command writes fresh, portable JSON under `benchmarks/` and exits nonzero if a
required suite, runtime, export, teacher, or four-GPU acceptance check is unavailable.
