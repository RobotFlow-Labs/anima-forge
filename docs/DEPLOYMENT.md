# FORGE Deployment Guide

Deploying compressed FORGE students to edge devices (Jetson, Apple Silicon) and serving via runtime APIs.

---

## Deployment Targets

| Platform | Backend | Export Format | Expected Latency |
|----------|---------|---------------|-----------------|
| NVIDIA Jetson Orin | TensorRT | `.engine` (INT8/FP16) | Roadmap; not device-validated |
| NVIDIA GPU Server | ONNX Runtime / TensorRT | `.onnx` / `.engine` | L4 fp16 validated |
| Apple Silicon M1-M4 | MLX | `.npz` (FP16) | Artifact validated; device latency pending |
| CPU (any) | ONNX Runtime | `.onnx` + external weights | Run the packaged benchmark locally |

---

## Export Formats

### ONNX Export

**Source**: `src/forge/export/onnx_export.py`

Universal format that runs on any platform via ONNX Runtime.

```bash
# Via pipeline
forge pipeline --config configs/forge_nano.yaml --stage export \
  --checkpoint outputs/checkpoints/final.pt --output-dir outputs/export
```

```python
import torch

from forge.export.onnx_export import export_onnx, validate_onnx
from forge.student import FORGEStudent
from forge.config import ForgeConfig

# Load trained student
config = ForgeConfig.from_yaml("configs/forge_nano.yaml")
student = FORGEStudent(config.student, model_dir="/path/to/models")
checkpoint = torch.load("outputs/checkpoints/best.pt", map_location="cpu")
student.load_state_dict(checkpoint["model_state_dict"])

# Export
onnx_path = export_onnx(
    student,
    output_path="./outputs/forge.onnx",
    image_size=384,
    max_seq_len=128,
    opset_version=19,
    optimize=True,          # Apply ORT graph optimizations
)

# Validate (compares PyTorch vs ONNX outputs)
validation = validate_onnx(student, onnx_path, n_samples=10, tolerance=0.01)
print(validation)  # {"status": "passed", "max_diff": 0.001, ...}
```

**Dynamic shapes**: Batch size and sequence length are dynamic for flexible deployment.

**Release evidence**: launch measurements are temporarily withheld after the image
preprocessing audit invalidated the first comparison artifacts. The project README will
publish results only after corrected training plus ONNX Runtime and TensorRT execution
pass on the regenerated checkpoints.

### TensorRT Export

**Source**: `src/forge/export/tensorrt_export.py`

NVIDIA-specific inference engine format. Required for Jetson deployment.

```python
from forge.export.tensorrt_export import export_tensorrt, check_tensorrt_available

if check_tensorrt_available():
    engine_path = export_tensorrt(
        onnx_path="./outputs/forge.onnx",
        output_path="./outputs/forge.engine",
        precision="fp16",           # "fp16" or "int8"
        workspace_mb=2048,          # TensorRT workspace memory
        calibration_data=None,      # Required for INT8
    )
```

**Requirements**: NVIDIA GPU + TensorRT SDK. Not available on Mac.

**Precision options**:
- `fp16`: mixed-precision engine; the builder retains sensitive reductions in fp32
- `int8`: requires real calibration data and is not claimed by launch validation yet

### MLX Export

**Source**: `src/forge/export/mlx_export.py`

Apple Silicon native format using Metal GPU acceleration.

```python
from forge.export.mlx_export import export_mlx, load_mlx_weights, validate_mlx_export

# Export
output_dir = export_mlx(
    student,
    output_dir="./outputs/forge-mlx/",
    config={"variant": "nano"},
)
# Creates:
#   outputs/forge-mlx/weights.npz    (FP16, uncompressed NPZ container)
#   outputs/forge-mlx/config.json    (architecture config)
#   outputs/forge-mlx/metadata.json  (export metadata)

# Validate
validation = validate_mlx_export(student, "./outputs/forge-mlx/")
print(validation)  # {"status": "passed", "n_pytorch_params": 42, ...}

# Load weights
weights = load_mlx_weights("./outputs/forge-mlx/")
# Returns dict[str, np.ndarray]
```

---

## Runtime Server

### FastAPI Inference Server

**Source**: `src/forge/serve.py`

The maintained HTTP server loads only a provenance-verified trained checkpoint. Its
model variant, action horizon, and action dimension come from that checkpoint's
`student_config` metadata.

```bash
forge serve --host 0.0.0.0 --port 8080 --device cuda --checkpoint outputs/checkpoints/final.pt
```

**Endpoints**:

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Package, checkpoint, model, device, and action-shape health metadata |
| POST | `/predict` | One `image` plus required `instruction`; returns one `(H,D)` action chunk |
| POST | `/batch_predict` | Repeated `images` plus required `instruction`; returns `(B,H,D)` actions |

**Prediction request**:
```bash
curl -X POST http://localhost:8080/predict \
  -F "image=@camera_frame.jpg" \
  -F "instruction=pick up the red block"
```

**Response**:
```json
{
  "actions": [[0.01, -0.02, 0.05, 0.00, 0.03, -0.01, 0.8]],
  "action_horizon": 1,
  "action_dim": 7,
  "instruction": "pick up the red block",
  "model": "FORGE-nano",
  "version": "3.0.1"
}
```

**Batch request**:
```bash
curl -X POST http://localhost:8080/batch_predict \
  -F "images=@camera_1.jpg" \
  -F "images=@camera_2.jpg" \
  -F "instruction=pick up the red block"
```

### Async Inference Engine

**Source**: `src/forge/runtime/async_engine.py`

Decouples model inference from action delivery for asynchronous robot control.

```python
from forge.runtime.async_engine import AsyncInferenceEngine, RuntimeConfig

config = RuntimeConfig(
    max_buffer_size=4,           # Max action chunks in buffer
    vision_timeout_ms=100,       # Max wait for vision processing
    action_horizon=8,            # Actions per chunk (H)
    chunk_overlap=2,             # Overlap for blending
    target_hz=50,                # Target action frequency
    action_dim=7,                # Robot action dimensions
)

engine = AsyncInferenceEngine(model, config)
engine.start()

# In robot control loop (50 Hz):
engine.submit_frame(camera_image, instruction="pick up cup")
action = engine.get_action()    # Non-blocking; returns None when no action is buffered
if action is not None:
    robot.execute(action)

# Check health
status = engine.get_status()
print(f"Buffer: {status.buffer_size}, Vision: {status.avg_vision_ms:.1f}ms")

engine.stop()
```

**Threading model**:
- **Vision thread** (background): Processes camera frames through the full model pipeline and updates shared feature state.
- **Action delivery** (main thread): Serves actions from the thread-safe `ChunkBuffer` ring buffer.

### ChunkBuffer

Thread-safe ring buffer for action chunks:

```python
from forge.runtime.async_engine import ChunkBuffer

buffer = ChunkBuffer(max_size=4, horizon=8, action_dim=7)
buffer.push(chunk)          # (H, D_action) array
action = buffer.pop_action()  # (D_action,) single action
print(buffer.size)          # Chunks remaining
buffer.clear()              # Reset
```

---

## Edge Deployment Checklist

### NVIDIA Jetson Orin Nano — roadmap, not device-validated

1. **Export to ONNX**:
   ```bash
   forge pipeline --config configs/forge_nano.yaml --stage export \
     --checkpoint outputs/checkpoints/final.pt --output-dir outputs/export
   ```

2. **Convert to TensorRT** (on Jetson or matching GPU):
   ```python
   from forge.export.tensorrt_export import export_tensorrt
   export_tensorrt("forge.onnx", "forge.engine", precision="fp16")
   ```

3. **Deploy runtime server**:
   ```bash
   forge serve --port 8080 --device cuda --checkpoint outputs/checkpoints/final.pt
   ```

4. **Connect robot control loop** via the HTTP endpoints above.

### Apple Silicon (M1-M4)

1. **Export to MLX**:
   ```python
   from forge.export.mlx_export import export_mlx
   export_mlx(student, "./forge-mlx/")
   ```

2. **Load and run with MLX**:
   ```python
   import mlx.core as mx
   from forge.export.mlx_export import load_mlx_weights
   weights = load_mlx_weights("./forge-mlx/")
   ```

3. **Or use the runtime server**:
   ```bash
   forge serve --port 8080 --device mps --checkpoint outputs/checkpoints/final.pt
   ```

---

## Embodiment Configuration

Generate robot-specific configs for deployment:

```bash
# List available profiles
forge embodiment list

# Generate config for a specific robot
forge embodiment config franka --output franka_config.yaml
forge embodiment config aloha --output aloha_config.yaml
```

| Robot | DoF | Action Dim | Control Hz | Recommended Head |
|-------|-----|-----------|------------|-----------------|
| Franka Emika Panda | 7 | 7 | 20 Hz | flow (H=8) |
| ALOHA (bimanual) | 14 | 14 | 50 Hz | chunk (H=16) |
| xArm | 6 | 6 | 100 Hz | flow (H=4) |
| UR5e | 6 | 6 | 125 Hz | flow (H=4) |

### Cross-Embodiment Transfer

Transfer a trained model between robots:

```bash
forge transfer info --source franka --target ur5e --strategy linear
```

```python
from forge.cross_embodiment import EmbodimentTransfer, TransferConfig

transfer = EmbodimentTransfer(
    source_profile=franka,
    target_profile=ur5e,
    config=TransferConfig(mapping_strategy="linear"),
)
target_actions = transfer.map_actions(source_actions)
```

**Strategies**:
- `linear`: Pad/trim + scale. No training needed. Good for similar robots.
- `joint_name`: Match joints by name. For same-family robots.
- `learned`: MLP adapter that requires paired demonstrations and task-specific evaluation.

---

## Docker Deployment

```bash
# Stamp the image with the exact source revision without copying .git.
export FORGE_GIT_SHA="$(git rev-parse HEAD)"

# Build the CPU validation container.
docker compose build forge-dev

# Run tests in container
docker compose up forge-dev

# Build the CUDA training and inference container.
docker compose --profile gpu build forge-train
```

The container definitions live at the repository root: `Dockerfile` and
`docker-compose.yml`. Image builds fail closed when `FORGE_GIT_SHA` is absent or
is not a full lowercase 40-character commit SHA.

---

## Inference Telemetry

Monitor deployed model performance:

```python
from forge.telemetry import InferenceTelemetry, TelemetryConfig

telemetry = InferenceTelemetry(TelemetryConfig(
    window_size=1000,
    anomaly_threshold=3.0,
    log_interval=100,
))

# In inference loop
telemetry.record_inference(latency_ms=12.3)
info = telemetry.record_action(action)  # anomaly detection
telemetry.record_buffer(buffer_fill)

# Export
summary = telemetry.summary()
telemetry.export_json("./telemetry.json")
```

```bash
forge telemetry summary --export-path ./telemetry.json
```

**Components**:
- `LatencyTracker`: p50/p95/p99 rolling latency
- `ThroughputTracker`: FPS over time window
- `ActionMonitor`: Action magnitude + z-score anomaly detection
- `BufferHealthMonitor`: Starvation/overflow detection

---

## Model Size Reference

Launch-week artifact sizes are withheld until corrected-preprocessing checkpoints pass
the complete compression and runtime matrix:

| Variant | Pruned checkpoint | Packed INT4 | Packed INT8 | ONNX total |
|---|---:|---:|---:|---:|

All four variant matrices and physical edge-device measurements remain pending. The
[project README](../README.md#how-it-works) is the only public launch-claim source.
