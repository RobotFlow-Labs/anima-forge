# FORGE Hyperparameter Guide

FORGE does not ship universal “best” numbers derived from synthetic inputs. Hardware,
robot embodiment, teacher fleet, and dataset all change the optimum. Search and validate
with genuine observations/actions from the target workload.

## Safe starting points

Use the canonical variant presets first; they select the supported SigLIP2 and language
backbones together:

| Variant | Language backbone | LoRA rank | Typical role |
|---------|-------------------|-----------|--------------|
| `micro` | SmolLM2-135M | 16 | smallest development student |
| `nano` | Qwen3-0.6B | 32 | default balanced student |
| `small` | Qwen3-1.7B | 64 | higher-capacity student |
| `medium` | Qwen3-4B bf16 | 64 | largest supported student |

Start with learning rate `2e-4`, then search around it on real data. Treat pruning ratios,
LoRA rank, action head, and quantization bit width as measured tradeoffs rather than fixed
recommendations.

## Run a real-data search

The automated search requires a LeRobot v3 dataset. Random inputs are accepted only with
the explicit test-only `--allow-mock` flag and must never be used for release claims.

```bash
forge hyperparam auto \
  --data-dir models/datasets/lerobot--pusht \
  --model-dir models \
  --trials 30 \
  --steps 100 \
  --seed 42 \
  --objective balanced \
  --device cuda \
  --json
```

The seed controls Optuna's TPE sampler and each trial's Python, NumPy, and Torch RNG
state. It defaults to 42 and is recorded in the search and trial metrics, so repeated
quality comparisons can use the same initialization, sample order, and stochastic model
path. The result also records `data_provenance`, genuine training sample count, measured
loss/FPS, and clearly labels theoretical packed-weight compression as an estimate.

## Validate candidates

Run the packaged suites and artifact-backed matrix before selecting a configuration:

```bash
forge benchmark all --device cuda --data-dir models/datasets/lerobot--pusht
forge benchmark matrix validation-manifest.json --device cuda
```

`forge benchmark all` requires the complete local runtime and four visible GPUs for the
multi-GPU, real-teacher, and 400-trial acceptance suites. Missing data, exports, teachers,
TensorRT, or devices produce a nonzero exit instead of an optional success.

Use prior genuine result files for ranking:

```bash
forge hyperparam recommend --objective balanced --top 3
forge hyperparam recommend --objective speed --top 3
forge hyperparam recommend --objective quality --top 3
forge hyperparam recommend --objective size --top 3 --json
```

Always re-run latency and quality checks on the deployment hardware. An estimated INT4
weight ratio is not a serialized size measurement; use the exported packed artifact and
runtime matrix for release evidence.
