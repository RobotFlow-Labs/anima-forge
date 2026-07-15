# FORGE Model Profiler

Analytical profiler for FORGE student VLA models. Generates HuggingFace model cards, VRAM estimates, FLOPs breakdowns, and training config recommendations — all formula-based with no GPU required.

**Source**: `src/forge/profiler/`

---

## Overview

- No GPU or model weights needed — all estimates are computed analytically
- Supports all 4 variants: `micro`, `nano`, `small`, `medium`
- Outputs HuggingFace-compatible model cards (YAML front-matter + markdown body)
- Per-component FLOPs breakdown covering vision encoder, bridge, backbone, LoRA, and action head
- VRAM estimates for inference (FP16) and training (mixed precision)
- Hyperparameter recommendations scaled by GPU VRAM and dataset size

---

## Quick Start

```python
from forge.profiler import FORGEProfiler

profiler = FORGEProfiler(variant="nano")
card = profiler.generate_card(dataset_size=90000, gpu_vram_gb=24.0)
card.save_json("profiles/nano.json")
md = profiler.generate_markdown(card)
```

---

## CLI Commands

### `forge profile card` — Full profile card

```bash
forge profile card --variant nano --json
forge profile card --variant nano --output profiles/nano.json --markdown profiles/README.md
```

Flags: `--variant` (micro/nano/small/medium), `--json`, `--output <path>`, `--markdown <path>`.

### `forge profile vram` — VRAM estimation

```bash
forge profile vram --variant nano --gpu-vram 24
```

Prints inference and training VRAM requirements and GPU fit status.

### `forge profile recommend` — Hyperparameter recommendations

```bash
forge profile recommend --variant nano --dataset-size 90000 --gpu-vram 24 --objective balanced
```

Returns a YAML block with batch size, learning rate, LoRA rank, warmup steps, and bridge config.

### `forge profile benchmark` — Latency benchmarks

```bash
forge profile benchmark --variant nano --device cuda --samples 100
```

Delegates to `forge benchmark` and appends p50/p95/p99 latency and throughput to the profile card.

---

## What Gets Estimated

### FLOPs

Per-component analytical estimates (single forward pass, one image):

| Component | Formula |
|-----------|---------|
| Vision encoder (SigLIP) | `2 * params * seq_len` |
| Bridge attention (per layer) | `4 * n_queries * d_model * (d_vision + d_model)` |
| Language backbone (per layer) | `12 * d_model² * seq_len` |
| LoRA adapters (per module per layer) | `2 * d * rank` |
| Action head | `K_steps * (4 * d_hidden² * n_blocks)` |

Total reported in GFLOPs.

### VRAM

| Mode | Formula |
|------|---------|
| Inference FP16 | `total_params * 2 bytes` |
| Training FP16 (mixed precision) | `frozen_params * 2 + trainable_params * 16 + activations` |

GPU compatibility table covers 7 targets: L4 24GB, A100 40/80GB, RTX 4090 24GB, Jetson AGX Orin 64GB, T4 16GB, RTX 3090 24GB.

### Canonical Config Defaults

| Variant | Base LR | LoRA Rank | Bridge Layers | Config Batch |
|---------|---------|-----------|---------------|--------------|
| micro   | 2e-4    | 16        | 3             | 16           |
| nano    | 2e-4    | 32        | 4             | 16           |
| small   | 1e-4    | 64        | 6             | 8            |
| medium  | 1e-4    | 64        | 4             | 2            |

These values mirror `configs/forge_<variant>.yaml`. Runtime auto-sizing can select a
different batch from the visible GPU and dataset; an explicit CLI batch remains exact.

---

## HuggingFace Model Card

`generate_markdown(card)` produces a complete `README.md` with:

- **YAML front-matter** — `language`, `license`, `tags`, `pipeline_tag`, `model-index`
- **ASCII architecture diagram** — `SigLIP → Bridge → LoRA Backbone → Action Head`
- **Parameter table** — total, frozen, and trainable params per component
- **FLOPs breakdown** — GFLOPs per component + total
- **VRAM table** — inference and training estimates per GPU
- **Training config YAML** — ready-to-paste hyperparams block
- **GPU compatibility table** — pass/fail for each supported GPU target

---

## Module Structure

```
src/forge/profiler/
├── __init__.py          # Public API: FORGEProfiler + all dataclasses
├── dataclasses.py       # ComponentProfile, FLOPsEstimate, VRAMEstimate,
│                        # RecommendedHyperparams, ModelProfileCard
├── flops.py             # estimate_flops() + VARIANT_SPECS
├── vram.py              # estimate_vram() + GPU_PROFILES
├── recommend.py         # recommend_hyperparams() + VARIANT_DEFAULTS
├── markdown.py          # generate_markdown() + generate_ascii_diagram()
└── profiler.py          # FORGEProfiler class (orchestrates all above)
```

---

## Variant Specs Reference

| Variant | LM Backbone   | hidden_size | Layers | ~Params |
|---------|---------------|-------------|--------|---------|
| micro   | SmolLM2-135M  | 576         | 30     | 0.2B    |
| nano    | Qwen3-0.6B    | 1024        | 28     | 0.6B    |
| small   | Qwen3-1.7B    | 2048        | 28     | 1.7B    |
| medium  | Qwen3-4B      | 2560        | 36     | 4.0B    |

All variants use frozen SigLIP2-SO400M (about 400M params, 729 tokens, d=1152).

---

## API Reference

### `FORGEProfiler`

**Source**: `src/forge/profiler/profiler.py`

```python
from forge.profiler import FORGEProfiler

profiler = FORGEProfiler(
    variant="nano",     # micro | nano | small | medium
    model_dir=None,     # Optional — enables AutoSense dimension detection
    device="cpu",       # Reserved for forge profile benchmark
)

card = profiler.generate_card(dataset_size=90000, gpu_vram_gb=24.0)
md   = profiler.generate_markdown(card)   # -> str
```

### Dataclasses

**Source**: `src/forge/profiler/dataclasses.py`

- `ComponentProfile` — `name`, `param_count`, `flops`, `trainable`
- `FLOPsEstimate` — per-component int fields + `total`, `total_gflops`
- `VRAMEstimate` — `inference_fp16_gb`, `training_fp16_gb`, `gpu_compatibility: dict[str, bool]`
- `RecommendedHyperparams` — `learning_rate`, `lora_rank`, `bridge_n_layers`, `batch_size`, `warmup_steps`, `gradient_clip`, `objective`
- `ModelProfileCard` — full card with `to_dict()`, `save_json()`, `from_dict()`, `from_json()`

---

## Testing

```bash
uv run pytest tests/test_v2_profiler.py -v
# Focused profiler tests; no GPU required
```

Tests cover all 4 variants, FLOPs formulas, VRAM estimates, GPU compatibility, markdown generation, and JSON round-trip serialization.

---

## Notes

- All estimates are analytical — measured latency requires `forge profile benchmark`
- FLOPs represent a single forward pass on one image with a 32-token instruction
- Training VRAM assumes frozen vision encoder and LoRA-only backbone updates
- `model_dir` is optional; if absent, defaults from `VARIANT_SPECS` are used
- All CLI commands support `--json` for machine-readable output
