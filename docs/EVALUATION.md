# FORGE VLA Evaluation Harness (PRD-32)

The evaluation harness runs FORGE students against standardized VLA benchmarks using Docker containers from the `vla-evaluation-harness` project. Results feed into the model registry and experiment log.

**Source**: `src/forge/eval/`

---

## Architecture

```
┌────────────────────┐     WebSocket/msgpack     ┌──────────────────────┐
│  ForgeModelServer  │ <───────────────────────── │  Docker Benchmark    │
│  (ws://0.0.0.0:8k) │                           │  Container           │
│                    │ ────────────────────────── │  (LIBERO/SimplerEnv/ │
│  Loads FORGE       │     predict → actions      │   VLABench)          │
│  student from      │                           └──────────┬───────────┘
│  checkpoint        │                                      │
└────────────────────┘                             JSON results
                                                           │
┌────────────────────┐                                     ▼
│  EvalRunner        │ <───────── parse ──────── results.json
│  orchestrates      │
│  server + docker   │ ────────── save ────────> EvalResult
│  + results parsing │                           ↓
└────────────────────┘                    outputs/eval/report.md
```

---

## Setup

### Prerequisites

- Docker installed and running
- FORGE installed with its complete runtime dependencies: `uv sync`
- A trained student checkpoint

### Pull Benchmark Images

```bash
# Pull all 3 images (download size depends on the current upstream images)
forge eval setup

# Images:
#   ghcr.io/allenai/vla-evaluation-harness/libero:latest
#   ghcr.io/allenai/vla-evaluation-harness/simpler:latest
#   ghcr.io/allenai/vla-evaluation-harness/vlabench:latest
```

---

## Available Benchmarks

| Name | Docker Image | Suite | Tasks |
|------|-------------|-------|-------|
| `libero` | `ghcr.io/allenai/vla-evaluation-harness/libero:latest` | `libero_spatial` | 10 spatial tasks |
| `simpler` | `ghcr.io/allenai/vla-evaluation-harness/simpler:latest` | SimplerEnv | ManiSkill2 real-to-sim tasks |
| `vlabench` | `ghcr.io/allenai/vla-evaluation-harness/vlabench:latest` | VLABench | Long-horizon language tasks |

---

## CLI Usage

### Start Model Server (Manual Testing)

```bash
forge eval serve \
  --checkpoint ./outputs/checkpoints/best.pt \
  --variant nano \
  --device cuda \
  --port 8000
```

The server exposes a WebSocket endpoint at `ws://0.0.0.0:8000` using the msgpack protocol expected by the vla-evaluation-harness containers.

### Run a Single Benchmark

```bash
forge eval run libero \
  --checkpoint ./outputs/checkpoints/best.pt \
  --variant nano \
  --device cuda \
  --episodes 20 \
  --max-tasks 10 \
  --output-dir ./outputs/eval
```

This:
1. Starts `ForgeModelServer` in a background thread
2. Launches the Docker benchmark container with appropriate env vars
3. Waits for evaluation to complete (up to 1 hour timeout)
4. Parses results into `EvalResult` dataclass
5. Stops the server

### Run All Benchmarks

```bash
forge eval run-all \
  --checkpoint ./outputs/checkpoints/best.pt \
  --variant nano \
  --device cuda \
  --json
```

Runs LIBERO, SimplerEnv, and VLABench sequentially.

### Compare Checkpoints

```bash
forge eval compare \
  --a ./outputs/ckpt_v1.pt \
  --b ./outputs/ckpt_v2.pt \
  --benchmark libero \
  --variant nano
```

Output includes delta success rate and per-task comparison.

### View Results

```bash
# Human-readable Markdown table
forge eval results --output-dir ./outputs/eval

# JSON output
forge eval results --output-dir ./outputs/eval --json
```

Both forms preserve each run's `status`. A simulator or harness exception is shown as
`failed` with its bounded diagnostic; an episode that executes normally but does not
solve the task remains `completed` with its measured success rate. Viewing historical
results is an inspection command and therefore exits zero even when a stored row failed.

---

## Python API

### ForgeModelServer

**Source**: `src/forge/eval/model_server.py`

```python
from forge.eval.model_server import ForgeModelServer

server = ForgeModelServer(
    checkpoint_path="./outputs/checkpoints/best.pt",
    variant="nano",
    model_dir="/path/to/models",     # Required local model directory
    device="cuda",
    port=8000,
    host="0.0.0.0",
    chunk_size=1,                    # Action chunk size
    image_size=384,                  # Input image resize
    action_scale=1.0,               # Action denormalization scale
    action_offset=0.0,              # Action denormalization offset
)

# Blocking mode (for CLI)
server.start(blocking=True)

# Background mode (for programmatic use)
server.start(blocking=False)

# Direct prediction (bypasses WebSocket)
result = server.predict({
    "images": {"base_camera": numpy_image},
    "task_description": "pick up the red block",
})
actions = result["actions"]  # np.ndarray shape (7,) or (H, 7)

server.stop()
```

**Startup loading**: `start()` loads and verifies the checkpoint and mandatory local
backbones before binding the WebSocket port. A client cannot connect to a partially
initialized model server.

**WebSocket protocol** (msgpack):
- The official `vla_eval.protocol.messages.Message` codec carries typed `hello`,
  `observation`, `action`, lifecycle, info, and error messages.
- Observation payloads contain `images` plus `task_description`; official camera keys
  include `agentview` for LIBERO and `primary` for SimplerEnv/VLABench.
- Action replies contain finite 7-D `actions` and preserve the request sequence number.

### EvalRunner

**Source**: `src/forge/eval/runner.py`

```python
from forge.eval.runner import EvalRunner

runner = EvalRunner(
    checkpoint_path="./outputs/checkpoints/best.pt",
    variant="nano",
    model_dir="./models",
    device="cuda",
    output_dir="./outputs/eval",
    port=8000,
)

# Single benchmark
result = runner.run_benchmark(
    benchmark="libero",              # "libero", "simpler", "vlabench"
    config_path=None,                # Optional custom config YAML
    episodes_per_task=20,
    max_tasks=10,
)
# result: dict with success_rate, per_task_rates, latency_p50_ms, etc.

# All benchmarks
results = runner.run_all(benchmarks=["libero", "simpler", "vlabench"])

# Compare two checkpoints
comparison = runner.compare(
    other_checkpoint="./other.pt",
    benchmark="libero",
    episodes_per_task=20,
    max_tasks=10,
)
# comparison: dict with success_rate_a, success_rate_b, delta_success_rate

# Utilities
EvalRunner.check_docker()  # -> bool
EvalRunner.pull_images()   # -> dict[name -> status]
```

### EvalResult

**Source**: `src/forge/eval/results.py`

```python
from forge.eval.results import EvalResult, load_results, results_to_table, append_to_report

@dataclass
class EvalResult:
    benchmark: str = ""
    success_rate: float = 0.0
    tasks: int = 0
    episodes_per_task: int = 0
    per_task_rates: dict[str, float] = {}
    latency_p50_ms: float = 0.0
    student_variant: str = ""
    checkpoint: str = ""
    timestamp: str = ""
    status: str = "completed"
    error: str = ""

    def to_dict(self) -> dict: ...
    def to_json(self, indent=2) -> str: ...
    def to_report_markdown(self) -> str: ...

# Load all results from output directory
results = load_results("./outputs/eval")

# Format as markdown table
print(results_to_table(results))

# Append to experiment log
append_to_report(result, "outputs/eval/report.md")
```

---

## Config Files

Benchmark-specific YAML configs in `configs/eval/`:

### `configs/eval/forge_student.yaml`
Model server configuration.

### `configs/eval/libero_forge.yaml`
LIBERO benchmark configuration.

### `configs/eval/simpler_forge.yaml`
SimplerEnv benchmark configuration.

### `configs/eval/vlabench_forge.yaml`
VLABench benchmark configuration.

---

## Web Dashboard Integration

The evaluation page is available at `#/eval` in the Command Center dashboard.

**API endpoint**: `GET /api/eval/results` returns a JSON array of `EvalResult` dicts.

```bash
# Launch dashboard
forge web --port 3000

# API directly
curl http://localhost:3000/api/eval/results
```

---

## Dependencies

```bash
# Install FORGE and all runtime dependencies
uv sync
```

Required packages:
- `websockets>=12.0` -- WebSocket server for model serving
- `msgpack>=1.0` -- Binary serialization protocol

---

## Testing

```bash
uv run pytest tests/test_v2_eval.py tests/test_v3_eval_truthfulness.py \
  tests/test_student_language_text.py -v
```

Tests cover:
- `ForgeModelServer` construction, startup loading, prediction
- `EvalRunner` construction, Docker check
- `EvalResult` serialization, parsing, report generation
- `results_to_table` formatting
- CLI command registration

---

## Notes

- Docker is required for running actual benchmarks against simulation environments
- The model server verifies and loads the checkpoint before accepting connections
- Results default to the ignored `outputs/eval/report.md` artifact when using
  `append_to_report()`; generated reports include status and error evidence.
- All CLI commands support `--json` for machine-readable output
- Evaluation is deliberately decoupled from the pipeline -- it is never auto-run
