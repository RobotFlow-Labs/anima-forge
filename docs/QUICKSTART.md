# Quickstart: first real distillation

This path produces a genuine student checkpoint. FORGE does not silently replace missing
weights or labels with random data.

## 1. Install and inspect the machine

```bash
curl -fsSL https://raw.githubusercontent.com/RobotFlow-Labs/anima-forge-distillation-pipeline/main/install.sh | sh
forge doctor
```

The installer is user-space only and creates an isolated uv tool environment. It
autodetects CUDA on Linux and uses the CPU build on macOS. Useful controls include
`--cpu`, `--cuda`, `--version 3.0.1`, `--backend uv|pipx`, `--no-modify-path`, and
`--uninstall`. Pipe flags with `sh -s --`, for example:

```bash
curl -fsSL https://raw.githubusercontent.com/RobotFlow-Labs/anima-forge-distillation-pipeline/main/install.sh \
  | sh -s -- --cpu --version 3.0.1
```

On Linux x64, `--cpu` resolves Torch and TorchVision from PyTorch's official CPU index;
it does not download or activate a CUDA-enabled Torch build. The complete mandatory
FORGE runtime dependency set is still installed.

The equivalent Windows CPU installer is:

```powershell
irm https://raw.githubusercontent.com/RobotFlow-Labs/anima-forge-distillation-pipeline/main/install.ps1 | iex
```

Manual alternative:

```bash
python -m venv .venv
. .venv/bin/activate
pip install anima-forge
```

Before the v3 package is published, replace the install command with the wheel built from
this checkout:

```bash
uv build
./install.sh --cpu --from-wheel dist/anima_forge-3.0.1-py3-none-any.whl
```

`forge doctor` reports the selected model directory, required disk space, CUDA readiness,
and each missing model with its recovery command.

## 2. Run the guided path

Once `robotflowlabs/forge-sample-labels` is published:

```bash
forge quickstart --yes
```

Until that publication gate is complete, point at an existing real FORGE label pack:

```bash
forge quickstart --yes \
  --data-dir /path/to/real-teacher-labels \
  --output-dir outputs/quickstart
```

The command performs readiness checks, fetches missing canonical nano/SigLIP2 weights,
validates label provenance, and runs 200 optimizer steps. Progress shows step, loss, ETA,
and CUDA memory. Use `--quiet` for a compact terminal result or `--json` for one final JSON
document.

## 3. Verify and continue

The final output names the checkpoint and next commands. Verify readiness and inspect the
artifact before compression:

```bash
forge info
forge pipeline \
  --config configs/forge_nano.yaml \
  --stage compress \
  --checkpoint outputs/quickstart/checkpoints/final.pt \
  --data-dir /path/to/real-teacher-labels \
  --output-dir outputs/quickstart-compressed
```

Compression calibrates pruning against the real teacher-label pack; the directory must
contain `teacher_labels/metadata.json`. Then export the packed QVLA checkpoint produced
by that stage with the same real data root:

```bash
forge pipeline \
  --config configs/forge_nano.yaml \
  --stage export \
  --checkpoint outputs/quickstart-compressed/compressed/qvla_4bit.pt \
  --data-dir /path/to/real-teacher-labels \
  --output-dir outputs/quickstart-export
```

## CPU smoke

CPU mode is useful for contract verification but is not the ten-minute target:

```bash
forge quickstart --yes --device cpu --max-steps 1 --batch-size 1 \
  --data-dir /path/to/real-teacher-labels \
  --output-dir outputs/quickstart-cpu
```

CPU smoke verifies the command and artifact contracts; it is not a performance
benchmark. Public timing, memory, and quality measurements are published only in the
README launch table after their exact training, compression, and runtime artifacts pass
the release evidence gates. Publication of the sanitized sample-label dataset remains
required before the zero-argument download path can be exercised from a fresh machine.
