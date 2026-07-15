# FORGE documentation

FORGE 3.0 distills large Vision-Language-Action teachers into compact students, applies
real pruning and packed INT4/INT8 quantization, then exports ONNX, TensorRT, and MLX
artifacts. The public CLI refuses silent mock fallbacks and stamps data/model provenance
into every trained checkpoint.

## Start here

```bash
pip install anima-forge
forge doctor
forge quickstart --yes
```

The v3 wheel and sample-label Hub dataset are release artifacts. Before publication, use
the locally built wheel and pass `forge quickstart --data-dir <real-labels>`. The detailed
path, expected artifacts, CPU fallback, and current publication status are in
[QUICKSTART.md](QUICKSTART.md).

## Current architecture

```text
real VLA teacher labels
        ↓
SigLIP2-SO400M (frozen) → bridge attention → LoRA Qwen3/SmolLM2 → action head
        ↓
chunk-aware pruning → packed QVLA/TurboQuant INT4 or INT8
        ↓
ONNX Runtime · TensorRT · MLX
```

Student variants:

| Variant | Language backbone | Intended role |
|---|---|---|
| `micro` | SmolLM2-135M | smallest development/edge student |
| `nano` | Qwen3-0.6B | default edge student |
| `small` | Qwen3-1.7B | larger edge/server student |
| `medium` | Qwen3-4B | largest canonical student |

The real teacher registry currently includes OpenVLA-7B, RDT2, SmolVLA, MolmoAct2, and
VLA-JEPA. Use `forge teacher list --json` for the runtime registry rather than relying on
a hard-coded list.

## Guides

| Document | Contents |
|---|---|
| [QUICKSTART.md](QUICKSTART.md) | Install to first real checkpoint |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Common failures with recovery commands |
| [CLI_REFERENCE.md](CLI_REFERENCE.md) | Public command reference |
| [PIPELINE.md](PIPELINE.md) | Label, distill, compress, export stages |
| [CONFIGURATION.md](CONFIGURATION.md) | YAML and environment configuration |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Modules, data flow, and provenance |
| [EVALUATION.md](EVALUATION.md) | LIBERO and other VLA harnesses |
| [DEPLOYMENT.md](DEPLOYMENT.md) | Runtime/export deployment |

## Source development

```bash
git clone https://github.com/RobotFlow-Labs/anima-forge-distillation-pipeline.git forge
cd forge
uv sync --locked --group dev
uv run forge doctor
uv run pytest -m "not gpu"
```

Python 3.12 is supported. Contributions target `develop`; see
[`CONTRIBUTING.md`](../CONTRIBUTING.md).
