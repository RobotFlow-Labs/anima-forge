# FORGE Architecture

## System Architecture

```
                            FORGE v3.0.1
    ┌─────────────────────────────────────────────────────┐
    │                                                     │
    │  ┌──────────┐  ┌──────────┐  ┌───────┐  ┌───────┐ │
    │  │ Teacher  │  │ Student  │  │Compress│  │ Export │ │
    │  │ Labels   │──│  KD      │──│ Prune  │──│ ONNX  │ │
    │  │ PRD-01   │  │ PRD-02/3 │  │ Quant  │  │ TRT   │ │
    │  └──────────┘  └──────────┘  │ PRD-4/5│  │ MLX   │ │
    │                              └───────┘  │ PRD-6/7│ │
    │                                          └───────┘ │
    │  ┌────────────────────────────────────────────────┐ │
    │  │              v2 Enhancements                   │ │
    │  │  Multi-Teacher  Curriculum   Auto-HP  Eval    │ │
    │  │  PRD-21         PRD-22       PRD-31   PRD-32  │ │
    │  └────────────────────────────────────────────────┘ │
    │  ┌────────────────────────────────────────────────┐ │
    │  │              Infrastructure                    │ │
    │  │  CLI  Web  Backend  Telemetry  Registry        │ │
    │  └────────────────────────────────────────────────┘ │
    └─────────────────────────────────────────────────────┘
```

---

## Module Map

```
src/forge/
├── __init__.py                     # Version: 3.0.1
├── cli_v2.py                       # Public Typer CLI application
├── cli_commands/                   # Command groups + strict JSON/error contracts
├── config.py                       # All config dataclasses + YAML loader
├── backend.py                      # CUDA/MLX/CPU abstraction layer
├── types.py                        # Core data types (EpisodeData, TeacherOutput)
│
├── teacher.py                      # PRD-01: Teacher label generation
├── student.py                      # PRD-02: Student architecture
├── distill.py                      # PRD-03: KD training loop
├── prune.py                        # PRD-04: Shallow-Pi layer pruning
├── prune_v2.py                     # PRD-04v2: Chunk-aware pruning
├── quantize.py                     # PRD-05: QVLA quantization
├── quantize_v2.py                  # PRD-05v2: Chunk-aware quantization
├── validate.py                     # PRD-07: Edge validation
├── pipeline.py                     # Full 4-stage pipeline runner
├── losses.py                       # KD loss functions (4 components)
├── trainer.py                      # PRD-23: Production trainer
├── curriculum.py                   # PRD-22: Curriculum learning
├── universal_distill.py            # PRD-21: Multi-teacher distillation
├── multi_teacher.py                # Multi-teacher loss + router
├── metrics.py                      # PRD-24: Training metrics
├── autosense.py                    # PRD-25: Auto-detect model dimensions
├── model_registry.py               # PRD-26: Trained model registry
├── hyperparam.py                   # PRD-27: Hyperparameter search
├── auto_hyperparam.py              # PRD-31: Optuna auto-HP
├── finetune.py                     # PRD-28: Domain adaptation
├── telemetry.py                    # PRD-29: Inference telemetry
├── cross_embodiment.py             # PRD-30: Cross-embodiment transfer
├── serve.py                        # FastAPI inference endpoint
├── demo/                           # VC demo and report runners
├── report.py                       # Markdown report generator
│
├── modules/                        # Neural network building blocks
│   ├── bridge_attention.py         # Cross-attention compression (729 -> 64 tokens)
│   ├── diffusion_head.py           # DDPM action head (10 steps)
│   ├── flow_head.py                # Flow matching head (1-4 steps)
│   ├── action_chunking.py          # Multi-step action prediction
│   ├── consistency_head.py         # Consistency distillation head (1 step)
│   ├── action_head_factory.py      # Factory: config -> head instance
│   └── lora.py                     # LoRA adapter injection
│
├── data/                           # Data loading and storage
│   ├── teacher_dataset.py          # HDF5 teacher label dataset
│   └── label_writer.py             # HDF5 label file writer
│
├── export/                         # Model export formats
│   ├── onnx_export.py              # ONNX export + ORT optimization
│   ├── tensorrt_export.py          # TensorRT INT8/FP16 engines
│   └── mlx_export.py              # Apple Silicon MLX format
│
├── runtime/                        # Asynchronous inference
│   └── async_engine.py             # Async vision/action decoupling
│
├── eval/                           # PRD-32: VLA evaluation harness
│   ├── model_server.py             # WebSocket/msgpack model server
│   ├── runner.py                   # Docker benchmark orchestrator
│   └── results.py                  # Result parsing + generated artifact report
│
├── teachers/                       # Teacher adapter registry
│   ├── registry.py                 # Auto-discovery registry
│   ├── openvla_adapter.py          # OpenVLA-7B adapter
│   ├── rdt2_adapter.py             # RDT2 + Qwen-VL + normalizer contract
│   ├── smolvla_adapter.py          # SmolVLA adapter
│   ├── molmoact2_adapter.py        # MolmoAct2 LeRobot adapter
│   └── vla_jepa_adapter.py         # VLA-JEPA LeRobot adapter
│
├── embodiments/                    # Robot profiles
│   └── registry.py                 # Franka, Aloha, XArm, UR5e
│
├── vision/                         # Vision encoder registry
│   └── ...                         # SigLIP, DINOv2, Theia
│
├── benchmark/                      # Performance benchmarking
│   ├── runner.py                   # Benchmark suite runner
│   └── comparison.py               # Report comparison
│
├── demo/                           # Demo report generation
│   ├── runner.py                   # DemoRunner orchestrator
│   └── report.py                   # HTML report with inline SVG
│
└── web/                            # Command Center dashboard
    ├── api.py                      # FastAPI endpoints (26 routes)
    ├── state.py                    # ServerState singleton
    ├── websockets.py               # Real-time streaming managers
    └── dashboard.html              # Single-file SPA (SACRED design)
```

---

## Student Model Architecture

The FORGE student is a 4-stage pipeline within a single `nn.Module`:

```
Input Image (B, 3, 384, 384)
         │
         ▼
┌─────────────────────┐
│  Stage A: Vision    │  SigLIP2-SO400M (about 400M params, FROZEN)
│  Encoder            │  Output: (B, 729, 1152)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Stage B: Bridge    │  Cross-attention compression (~2M params, TRAINED)
│  Attention          │  64 learned queries attend over 729 vision tokens
│                     │  Output: (B, 64, d_model)
└─────────┬───────────┘
          │ + Language embeddings
          ▼
┌─────────────────────┐
│  Stage C: Language  │  Qwen3/SmolLM2 backbone with LoRA adapters
│  Backbone (LoRA)    │  Full model frozen, only q/k/v/o projections adapted
│                     │  Output: (B, 64+seq_len, d_model)
└─────────┬───────────┘
          │ Pool vision tokens → mean
          ▼
┌─────────────────────┐
│  Stage D: Action    │  One of: Diffusion (10 steps), Flow (1-4 steps),
│  Head               │  Chunk (multi-step), Consistency (1 step)
│                     │  Output: (B, [H,] D_action)
└─────────────────────┘
```

**Source**: `src/forge/student.py` -- `FORGEStudent` class

### Bridge Attention Detail

Located in `src/forge/modules/bridge_attention.py`:

```
Vision tokens (B, 729, 1152)
         │
    ┌────▼────┐
    │ Linear  │  vision_proj: 1152 -> d_model
    └────┬────┘
         │
    ┌────▼────────────────┐
    │ Cross-Attention x4  │  64 learned queries attend over vision KV
    │  Pre-norm            │  Multi-head (8 heads), with FFN + residual
    │  Q: queries          │
    │  K,V: vision tokens  │
    └────┬────────────────┘
         │
    ┌────▼────┐
    │LayerNorm│
    └────┬────┘
         │
         ▼
    Compressed: (B, 64, d_model)
```

### Action Head Selection

Factory pattern in `src/forge/modules/action_head_factory.py`:

| Head Type | Class | Inference contract |
|-----------|-------|--------------------|
| `diffusion` | `DiffusionActionHead` | configurable iterative DDPM sampling |
| `flow` | `FlowMatchingActionHead` | configurable ODE integration |
| `chunk` | `ActionChunkHead` | one forward pass emitting an action chunk |
| `consistency` | `ConsistencyActionHead` | one forward pass after consistency training |

---

## Data Flow

### Training Data Flow

```
Teacher VLA (7B)
      │
      ▼
┌──────────────┐    HDF5 Files    ┌──────────────┐
│ Label Gen    │ ──────────────── │ TeacherLabel │
│ (teacher.py) │                  │ Dataset      │
└──────────────┘                  └──────┬───────┘
                                         │
     ForgeConfig ──┐                     │
                   ▼                     ▼
             ┌──────────┐         ┌──────────────┐
             │ Student  │ <────── │  Distill     │
             │ Model    │         │  Training    │
             └────┬─────┘         │  Loop        │
                  │               └──────┬───────┘
                  ▼                      │
          ┌───────────┐          Checkpoints (.pt)
          │ Prune     │                  │
          │ Quantize  │ <────────────────┘
          └─────┬─────┘
                │
                ▼
          ┌───────────┐
          │ Export     │ ── ONNX, TensorRT, MLX
          └─────┬─────┘
                │
                ▼
          ┌───────────┐
          │ Validate  │ ── Latency, throughput, correctness
          └───────────┘
```

### Inference Data Flow (Runtime)

```
Camera Frame
      │
      ▼
┌─────────────────┐
│ AsyncInference  │  Background model thread
│ Engine          │
│  submit_frame() │
└─────────┬───────┘
          │ Model forward pass
          ▼
┌─────────────────┐
│ ChunkBuffer     │  Thread-safe ring buffer
│  push(chunk)    │  Stores (H, D_action) chunks
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Robot Control   │  Non-blocking action lookup
│  get_action()   │
└─────────────────┘
```

---

## Multi-Teacher Architecture (PRD-21)

```
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ OpenVLA-7B   │  │ RDT2 7.5B    │  │ SmolVLA .45B │
│ local-only   │  │ local-only    │  │ local-only   │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                  │
       ▼                 ▼                  ▼
┌──────────────────────────────────────────────────┐
│              ConfidenceRouter                     │
│  Gumbel softmax over teacher confidence vectors   │
│  Learned per-sample weights w_i                   │
└──────────────────────┬───────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────┐
│          UniversalDistillationLoss                │
│  L = alpha_kd * sum(w_i * KD_i) +               │
│      alpha_task * task +                          │
│      alpha_div * diversity +                      │
│      alpha_con * consistency                      │
└──────────────────────────────────────────────────┘
```

---

## Training Phases

The production trainer (`src/forge/trainer.py`) manages three training phases:

```
Step 0          10%          83%          100%
  │              │            │             │
  ├──── Phase 1 ─┤── Phase 2 ─┤── Phase 3 ──┤
  │   Bridge     │   Full     │   Action    │
  │   warmup     │   distill  │   fine-tune │
  │              │            │             │
  │ Bridge +     │ Bridge +   │ Action head │
  │ Action head  │ LoRA +     │ only        │
  │ trainable    │ Action head│             │
  │              │ trainable  │             │
```

With curriculum learning enabled:

```
Difficulty
  1.0 ─────────────────────────────────────── ●
      │                                    ╱
      │                                 ╱
      │                              ╱
      │                           ╱
  0.3 ● ─────────────────────╱
      │
      └────────────────────────────────────────
      0         ramp_steps              max_steps
```

---

## Backend Abstraction

`src/forge/backend.py` provides platform-independent compute:

```
                  detect_backend()
                       │
              ┌────────┼────────┐
              ▼        ▼        ▼
         BackendType  BackendType  BackendType
           .CUDA       .MLX        .CPU
              │        │           │
              ▼        ▼           ▼
        TorchBackend  MLXBackend  TorchBackend
        (device=cuda)             (device=cpu)
```

Override with `FORGE_DEVICE` environment variable: `cuda`, `mlx`, `cpu`.

---

## Web Dashboard Architecture

Single-file SPA at `src/forge/web/dashboard.html`:

```
FastAPI (src/forge/web/api.py)
    │
    ├── GET  /                    → dashboard.html
    ├── GET  /api/status          → system status
    ├── GET  /api/config          → current config
    ├── GET  /api/teachers        → teacher list
    ├── POST /api/teachers/{n}/load   → load teacher
    ├── GET  /api/models          → model list
    ├── POST /api/train/start     → start training job
    ├── GET  /api/train/status    → training status
    ├── WS   /api/train/stream    → real-time training metrics
    ├── POST /api/compress/start  → start compression
    ├── POST /api/benchmarks/run  → run benchmarks
    ├── GET  /api/benchmarks      → benchmark history
    ├── GET  /api/embodiments     → robot profiles
    ├── POST /api/predict         → single prediction
    ├── WS   /api/stream          → streaming inference
    ├── GET  /api/experiments/auto_hp → auto-HP results
    ├── GET  /api/eval/results    → evaluation results
    └── POST /api/demo/run        → run demo
```

Design language: SACRED -- no border-radius, #FF3B00 orange accent, Oswald + JetBrains Mono fonts.

---

## Key Design Decisions

1. **Frozen vision encoder**: Canonical configs freeze the local SigLIP2 encoder. Its
   share of total parameters depends on the selected language-backbone variant; FORGE
   does not publish one fixed percentage for every student.

2. **Bridge attention**: Cross-attention with learned queries maps vision tokens into
   the language-backbone width. Query count and layer count come from the selected
   canonical config.

3. **Variant-specific LoRA**: Canonical micro, nano, small, and medium configs define
   their own LoRA ranks and target modules. Documentation and profiling read those
   values from config rather than assuming a fixed rank or trainable-parameter count.

4. **Action-head factory pattern**: `create_action_head(config)` selects diffusion,
   flow, chunk, or consistency behavior without changing the student call contract.

5. **Confidence-weighted KD loss**: Standard MSE loss fails for VLA because small errors compound into trajectory failures. Weighting by teacher confidence on each action dimension focuses learning on reliable supervision signals.

6. **Dual-platform backend**: The `forge.backend` abstraction allows the same codebase to run on CUDA (TensorRT/Jetson) and MLX (Apple Silicon) without conditional imports scattered through the code.

7. **Fail-closed runtime assets**: Production students and model servers require their
   configured local weights and tokenizer. Synthetic backbones are available only via
   the explicit test-only mock opt-in and retain mock provenance.

8. **HDF5 for teacher labels**: Teacher label generation and student training are decoupled via HDF5 files. This allows generating labels once on a powerful machine and training on a different machine.
