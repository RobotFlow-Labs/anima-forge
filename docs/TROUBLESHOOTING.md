# Troubleshooting

FORGE errors include a recovery hint and exit nonzero. Use `--json` when another process
needs a single machine-readable result.

## Missing model weights

```bash
forge models fetch --all-students
forge doctor
```

Set a portable model location with `FORGE_MODEL_DIR=/path/to/models` when needed.

## CUDA is unavailable

Run `forge doctor --json` and fix the reported driver/runtime issue. For a slow contract
smoke, select `--device cpu`; do not relabel CPU results as GPU benchmarks.

## CUDA works but NVIDIA management/NVML is unhealthy

`forge doctor` checks both CUDA compute and the `nvidia-smi` management path. CUDA tensor
operations can keep working during a driver/library version mismatch even though NCCL,
multi-GPU execution, and Docker GPU attachment fail. Treat the doctor warning as a real
readiness failure for those features.

Finish or stop active GPU jobs before changing the driver. Then reload the matching NVIDIA
kernel modules or reboot the host, and require all three checks to pass before retrying a
multi-GPU or container workload:

```bash
nvidia-smi
forge doctor --json
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi
```

## Configuration file is missing or invalid

```bash
forge config init > forge.yaml
forge pipeline --config forge.yaml --help
```

The error identifies the failing path or value.

## Teacher labels are missing

Use `forge quickstart --data-dir <real-labels>` or generate labels with a locally installed
teacher. Synthetic labels require explicit `--allow-mock` and stamp every output as mock.

## A port is already in use

```bash
forge serve --checkpoint <checkpoint.pt> --port 8001
forge web --port 3001 --no-browser
```

## Hugging Face authentication is missing

```bash
hf auth login
hf auth whoami
```

Never paste a token into a tracked file. Model downloads for public repositories do not
require write access; publishing does.

## Disk space is low

Run `forge doctor` and free the amount it reports. Keep checkpoints and exports outside
the checkout when possible, for example `--output-dir /data/forge-runs/my-run`.

## The student variant is invalid

Canonical variants are `micro`, `nano`, `small`, and `medium`. Generate a valid starter
configuration with `forge config init > forge.yaml`.

## A checkpoint is incompatible or untrusted

Pass a checkpoint produced by the current FORGE pipeline. Mock or missing provenance is
rejected by serving, evaluation, benchmarking, compression, and export unless the command
explicitly exposes and receives `--allow-mock`.

## Training was interrupted

Foreground Ctrl-C exits cleanly. Detached training can be inspected and stopped with:

```bash
forge train status --json
forge train stop <run-id>
```

The stopped run writes a terminal heartbeat and `stopped.pt`, not a deceptive `final.pt`.
