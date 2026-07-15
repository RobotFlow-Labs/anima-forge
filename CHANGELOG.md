# Changelog

All notable changes to FORGE are documented here.

## 3.0.0 — 2026-07

- Refreshed the student fleet to SmolLM2-135M and Qwen3 0.6B/1.7B/4B with SigLIP2.
- Added five real 2026 teacher adapters and fail-closed provenance across labels,
  checkpoints, compression, serving, benchmarking, and evaluation.
- Replaced placeholder compression outputs with reusable pruned checkpoints and genuine
  packed INT4/INT8 QVLA and TurboQuant artifacts.
- Added real artifact-driven battle validation, ONNX Runtime provider benchmarks, and
  TensorRT/MLX/ONNX export paths.
- Seeded training benchmarks and Optuna search across Python, NumPy, CPU Torch, and CUDA,
  with the reproducibility seed exposed through `forge hyperparam auto --seed`.
- Added `forge quickstart`, live training progress, recovery hints, a one-screen first-run
  banner, and commented starter configuration generation.
- Standardized on Python 3.12 and made every 2026 teacher runtime a required install dependency.
- Modernized the PyTorch 2.10, Transformers 5.x, and ONNX toolchain.

## 2.0.0

- Introduced the second-generation distillation, evaluation, and deployment architecture.
