# FORGE compatibility matrix

This document separates the supported dependency envelope from combinations that have
actually been exercised. A supported row without a check mark is a release target, not a
claim that the full FORGE suite has run on that exact combination.

## Python, PyTorch, and CUDA

| Python | PyTorch | CUDA runtime | July 2026 validation | Status |
| --- | --- | --- | --- | --- |
| 3.12.12 | 2.10.0+cu128 | CUDA 12.8 wheel, 4x NVIDIA L4 | full CPU CI plus real GPU distillation, export, and quantization execution | **Verified ✅** |

CPU-only use is supported with the same Python and PyTorch ranges. The CUDA column describes
the GPU build that was tested; it is not required for CPU-only commands.

## Verified validation records

The two checks below were run on the verified Python 3.12.12 / PyTorch 2.10.0+cu128 stack.
They cover different release surfaces and do not represent two distinct dependency
combinations.

| Validation surface | Evidence recorded on 2026-07-12 | Result |
| --- | --- | --- |
| CPU/full test surface | repository CI, wheel install, type check, lint, and format gates | **Verified ✅** |
| GPU runtime surface | Real L4 distillation and `forge quantize bench` completed with the CUDA 12.8 PyTorch wheel | **Verified ✅** |

## Release dependency envelope

The release-critical ranges are declared in `pyproject.toml`; `uv.lock` is the reproducible
installation artifact used by CI.

| Component | Supported range | Locked and locally exercised version |
| --- | --- | --- |
| Python | `>=3.12,<3.13` | 3.12.12 |
| PyTorch | `>=2.10,<2.11` | 2.10.0+cu128 |
| torchvision | `>=0.25,<0.26` | 0.25.0 |
| Transformers | `>=5.0,<6` | 5.5.4 |
| PEFT | `>=0.18,<1` | 0.18.1 |
| Accelerate | `>=1.13,<2` | 1.14.0 |
| ONNX | `>=1.20,<2` | 1.20.1 |
| ONNX Runtime | required on every platform; `>=1.24,<1.27` | GPU 1.24.4 locally; GPU 1.26.0 clean-wheel smoke |
| ONNX Script | `>=0.6,<1` | 0.6.2 |
| NumPy | `>=2.2,<2.3` | 2.2.6 |
| Typer | `>=0.24,<1` | 0.24.1 |
| FastAPI | `>=0.135,<1` | 0.135.2 |
| LeRobot | `>=0.6,<0.7` (required) | 0.6.0 |
| PyAV | `>=15,<16` (required) | 15.1.0 |
| TensorRT | `==10.16.0.72` on Linux CUDA installs | 10.16.0.72 |

## Runtime boundary decisions

- PyTorch 2.11 is excluded until the complete CPU/GPU/teacher/export matrix passes on it.
- ONNX Runtime 1.27 is excluded because its x86_64 GPU wheel selected CUDA 13 and failed
  to import on the supported CUDA 12.8 stack. The
  [official CUDA execution-provider documentation](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
  confirms that runtime packages must match the CUDA major version.
- TensorRT is pinned to the exact build used by the passing local engine evidence instead
  of allowing an unverified patch release at install time.

## Reproduction

Use the committed lockfile rather than asking the resolver for newer packages:

```bash
uv sync --locked --group dev
uv run pytest tests/ -q
```

The same command installs the mandatory platform-appropriate runtime. On Linux, that
includes the pinned CUDA/TensorRT stack; there is no optional CUDA extra:

```bash
uv sync --locked --group dev
```

CUDA compatibility depends on both the PyTorch wheel runtime and the host driver. The verified
machine used the `cu128` wheel successfully on its newer driver; this does not imply that every
CUDA 12.x wheel/driver pairing has been exercised.
