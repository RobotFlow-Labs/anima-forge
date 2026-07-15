# FORGE Python API Reference

Complete reference for all public Python classes and functions.

---

## Table of Contents

- [forge.student](#forgestudent)
- [forge.config](#forgeconfig)
- [forge.pipeline](#forgepipeline)
- [forge.teacher](#forgeteacher)
- [forge.distill](#forgedistill)
- [forge.losses](#forgelosses)
- [forge.trainer](#forgetrainer)
- [forge.curriculum](#forgecurriculum)
- [forge.universal_distill](#forgeuniversal_distill)
- [forge.modules](#forgemodules)
- [forge.autosense](#forgeautosense)
- [forge.backend](#forgebackend)
- [forge.model_registry](#forgemodel_registry)
- [forge.hyperparam](#forgehyperparam)
- [forge.auto_hyperparam](#forgeauto_hyperparam)
- [forge.finetune](#forgefinetune)
- [forge.telemetry](#forgetelemetry)
- [forge.cross_embodiment](#forgecross_embodiment)
- [forge.metrics](#forgemetrics)
- [forge.export](#forgeexport)
- [forge.runtime](#forgeruntime)
- [forge.eval](#forgeeval)
- [forge.web](#forgeweb)

---

## forge.student

**Source**: `src/forge/student.py`

### FORGEStudent

```python
class FORGEStudent(nn.Module):
    """FORGE student VLA model.

    Architecture:
        SigLIP2-SO400M (frozen) -> Bridge Attention -> Qwen3/SmolLM2 (LoRA) -> Action Head
    """

    def __init__(self, config: StudentConfig, model_dir: str | Path | None = None):
        """
        Args:
            config: StudentConfig with variant, dimensions, action head type
            model_dir: Path to local model weights. Missing real weights raise
                       ForgeModelNotFoundError unless mock use is explicitly enabled.
        """

    def forward(
        self,
        images: torch.Tensor,                      # (B, C, H, W)
        language_ids: torch.Tensor | None = None,   # (B, seq_len)
        language_text: str | None = None,            # Raw text
        proprioception: torch.Tensor | None = None,  # (B, D_proprio)
        gt_actions: torch.Tensor | None = None,      # (B, D_action) for training
    ) -> dict:
        """
        Returns:
            dict with keys:
            - 'actions': (B, D_action) or (B, H, D_action) predicted actions
            - 'vision_features': (B, n_queries, D_model) compressed vision
            - 'loss': scalar, present only when gt_actions provided (diffusion/flow heads)
        """

    @property
    def total_params(self) -> int: ...
    @property
    def trainable_params(self) -> int: ...
    def trainable_parameters(self) -> list[nn.Parameter]: ...
```

**Usage**:
```python
from forge.student import FORGEStudent
from forge.config import ForgeConfig

config = ForgeConfig.default()
student = FORGEStudent(config.student, model_dir="/path/to/models")

# Training
out = student(images, language_ids=tokens, gt_actions=actions)
loss = out["loss"]  # from action head
actions = out["actions"]

# Inference
with torch.no_grad():
    out = student(images, language_text="pick up the red block")
    actions = out["actions"]  # (B, 7)
```

---

## forge.config

**Source**: `src/forge/config.py`

### ForgeConfig

Master configuration combining all pipeline stages.

```python
@dataclass
class ForgeConfig:
    paths: ModelPaths
    teacher: TeacherConfig
    student: StudentConfig
    vision: VisionConfig
    distill: DistillConfig
    pruning: PruningConfig
    quant: QuantConfig
    export: ExportConfig
    web: WebConfig
    universal: UniversalDistillConfig
    curriculum: CurriculumConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> ForgeConfig: ...
    @classmethod
    def default(cls) -> ForgeConfig: ...
```

### Sub-configs

```python
@dataclass
class ModelPaths:
    model_dir: str          # Base directory for all model weights
    teacher: str            # Teacher model subdirectory name
    vision_encoder: str     # Vision encoder subdirectory name
    language_model: str     # Language model subdirectory name
    output_dir: str         # Pipeline output directory
    data_dir: str           # Teacher label data directory

    @property
    def teacher_path(self) -> Path: ...
    @property
    def vision_encoder_path(self) -> Path: ...
    @property
    def language_model_path(self) -> Path: ...

@dataclass
class StudentConfig:
    variant: str = "nano"                   # nano, small, micro, medium
    vision_encoder: str                     # HF model ID
    language_model: str                     # HF model ID
    bridge_d_vision: int = 1152             # Vision encoder output dim
    bridge_d_model: int = 896               # Language model hidden dim
    bridge_n_queries: int = 64              # Compressed token count
    bridge_n_heads: int = 8                 # Attention heads in bridge
    bridge_n_layers: int = 4                # Bridge attention layers
    action_dim: int = 7                     # Robot action dimensions
    action_head_layers: int = 4             # MLP layers in action head
    action_diffusion_steps: int = 10        # DDPM steps (diffusion head)
    lora_rank: int = 32                     # LoRA adapter rank
    lora_alpha: int = 64                    # LoRA alpha scaling
    lora_target_modules: list[str]          # LoRA target layers
    action_horizon: int = 1                 # H=1 single-step (v1 compat)
    chunk_overlap: int = 2                  # Chunk blending overlap
    action_head_type: str = "diffusion"     # diffusion | flow | chunk | consistency
    flow_inference_steps: int = 4           # ODE steps for flow head
    autosense: bool = True                  # Auto-detect model dims

@dataclass
class DistillConfig:
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_steps: int = 50000
    batch_size: int = 16
    gradient_accumulation_steps: int = 4
    temperature: float = 4.0
    alpha_kd: float = 0.4
    alpha_task: float = 0.3
    alpha_feat: float = 0.2
    alpha_action: float = 0.1
    eval_every: int = 1000
    save_every: int = 2000

@dataclass
class PruningConfig:
    target_layers: int = 8
    calibration_samples: int = 500
    recovery_lr: float = 5e-5
    recovery_steps: int = 5000
    keep_first_n: int = 2
    keep_last_n: int = 2

@dataclass
class QuantConfig:
    target_avg_bits: float = 4.0
    min_bits: int = 2
    max_bits: int = 8
    calibration_samples: int = 200
    post_quant_finetune_steps: int = 1000

@dataclass
class ExportConfig:
    formats: list[str] = ["onnx", "mlx"]
    tensorrt_precision: str = "int8"
    tensorrt_workspace_mb: int = 2048
    coreml_min_target: str = "macOS14"
    onnx_opset: int = 17

@dataclass
class CurriculumConfig:
    enabled: bool = True
    difficulty_metric: str = "loss"
    initial_difficulty: float = 0.3
    final_difficulty: float = 1.0
    ramp_steps: int = 50000
    ramp_schedule: str = "linear"
    plateau_window: int = 2000
    plateau_threshold: float = 0.01
    plateau_lr_factor: float = 0.5
    max_plateaus: int = 3
    teacher_dropout: bool = False
    teacher_dropout_start: float = 0.0
    teacher_dropout_end: float = 0.3
    teacher_dropout_ramp_steps: int = 30000
    hard_example_mining: bool = True
    hard_example_ratio: float = 0.3
    loss_history_size: int = 10000

@dataclass
class UniversalDistillConfig:
    teacher_names: list[str] = ["openvla-7b", "rdt2-fm", "smolvla-base"]
    gpu_assignment: dict[str, int] = {}
    router_temperature: float = 1.0
    use_gumbel: bool = True
    max_steps: int = 100000
    batch_size: int = 8
    staged: bool = False
    teachers_per_stage: int = 3
    steps_per_stage: int = 25000
    alpha_task: float = 0.3
    alpha_diversity: float = 0.05
    alpha_consistency: float = 0.1

@dataclass
class WebConfig:
    host: str = "127.0.0.1"
    port: int = 3000
    cors_origins: list[str] = ["http://localhost:3000"]
    auto_open_browser: bool = True
```

---

## forge.pipeline

**Source**: `src/forge/pipeline.py`

```python
def run_pipeline(
    config: ForgeConfig,
    device: str | None = None,
    skip_labels: bool = False,
    stage: str | None = None,           # None, "labels", "distill", "compress", "export", "validate"
    max_distill_steps: int | None = None,
    max_recovery_steps: int | None = None,
) -> dict:
    """Run complete FORGE pipeline. Returns summary dict with per-stage results."""
```

---

## forge.teacher

**Source**: `src/forge/teacher.py`

```python
def load_teacher(model_path: str | Path, device: str = "cpu", dtype = torch.bfloat16) -> nn.Module:
    """Load a teacher VLA model via HuggingFace AutoModelForVision2Seq."""

def load_processor(model_path: str | Path) -> Any:
    """Load processor/tokenizer for teacher."""

def generate_teacher_labels(
    config: ForgeConfig,
    teacher_override: str | None = None,
    max_episodes: int | None = None,
    device: str | None = None,
) -> dict:
    """Generate soft labels. Returns summary dict."""

def compute_action_confidence(action_std: np.ndarray) -> np.ndarray:
    """Confidence = 1 / (1 + std). Normalized to [0, 1]."""

class ExtractionHooks:
    """Forward hooks to capture intermediate teacher representations."""
    def register(self, model: nn.Module, extract_vision: bool = True) -> None: ...
    def remove(self) -> None: ...
    def reset(self) -> None: ...
    # After forward: self.vision_features: torch.Tensor | None
```

---

## forge.distill

**Source**: `src/forge/distill.py`

```python
def train_forge(
    config: ForgeConfig,
    device: str | None = None,
    max_steps: int | None = None,
    checkpoint_dir: str | None = None,
    resume_from: str | None = None,
) -> dict:
    """Main KD training loop. Returns summary with final_loss, best_loss, etc."""
```

---

## forge.losses

**Source**: `src/forge/losses.py`

```python
class ForgeDistillationLoss(nn.Module):
    """Composite KD loss: L = alpha_kd*KD + alpha_task*task + alpha_feat*feat + alpha_action*action"""

    def __init__(
        self,
        temperature: float = 4.0,
        alpha_kd: float = 0.4,
        alpha_task: float = 0.3,
        alpha_feat: float = 0.2,
        alpha_action: float = 0.1,
        feature_proj_dim: int | None = None,
    ): ...

    def forward(
        self,
        student_actions, teacher_action_logits, ground_truth_actions,
        student_vision_features=None, teacher_vision_features=None,
        teacher_action_mean=None, teacher_action_std=None, teacher_confidence=None,
    ) -> dict[str, torch.Tensor]:
        """Returns dict with 'total', 'kd', 'task', 'feature', 'action' losses."""

def kd_loss(student, teacher, temperature=4.0) -> torch.Tensor: ...
def task_loss(predicted, ground_truth) -> torch.Tensor: ...
def feature_alignment_loss(student_feat, teacher_feat, projector=None) -> torch.Tensor: ...
def action_distribution_loss(student, mean, std, confidence=None) -> torch.Tensor: ...
def chunk_aware_kd_loss(student, teacher, temperature=4.0, decay_factor=0.95) -> torch.Tensor: ...
```

---

## forge.trainer

**Source**: `src/forge/trainer.py`

```python
class ProductionTrainer:
    """Unified training orchestrator with curriculum, plateau, mining, phases."""

    def __init__(
        self,
        student: nn.Module,
        dataset: Dataset,
        loss_fn: nn.Module,
        config: ForgeConfig,
        device: str = "cpu",
        n_teachers: int = 1,
        checkpoint_dir: str | None = None,
    ): ...

    def train(
        self,
        max_steps: int | None = None,
        log_every: int = 100,
        checkpoint_every: int | None = None,
        eval_fn: Any | None = None,
    ) -> TrainingReport: ...

    def save_checkpoint(self, tag: str | None = None) -> Path: ...
    def load_checkpoint(self, path: str | Path) -> None: ...
    def get_status(self) -> dict[str, Any]: ...

@dataclass
class TrainingState:
    global_step: int = 0
    best_loss: float = inf
    phase: int = 1
    plateau_count: int = 0
    lr_multiplier: float = 1.0

@dataclass
class TrainingReport:
    total_steps: int = 0
    elapsed_seconds: float = 0.0
    final_loss: float = 0.0
    best_loss: float = inf
    final_lr: float = 0.0
    plateaus_detected: int = 0
    phase_transitions: list[dict] = []
    curriculum_stats: dict = {}

class AdaptiveLRScheduler:
    """Cosine + warmup + plateau-based reduction."""
    def step(self, loss: float | None = None) -> None: ...
    def get_lr(self) -> float: ...

def get_phase(step: int, max_steps: int) -> int:
    """Phase 1 (0-10%), Phase 2 (10-83%), Phase 3 (83-100%)."""

def set_trainable_for_phase(student: nn.Module, phase: int) -> None: ...
```

---

## forge.curriculum

**Source**: `src/forge/curriculum.py`

```python
class DifficultyScorer:
    """Score samples by loss, confidence, or teacher_disagreement."""
    def __init__(self, metric: str = "loss"): ...

class CurriculumScheduler:
    """Difficulty schedule: linear, cosine, or step."""
    def get_difficulty(self, step: int) -> float: ...

class CurriculumSampler(torch.utils.data.Sampler):
    """PyTorch sampler that filters by difficulty threshold."""
    def set_step(self, step: int) -> None: ...
    def update_difficulty_scores(self, scores: torch.Tensor) -> None: ...

class PlateauDetector:
    """Detects loss plateaus and triggers LR reduction."""
    def update(self, loss: float) -> None: ...
    def check_plateau(self, step: int) -> bool: ...
    def get_lr_multiplier(self) -> float: ...

class TeacherDropout:
    """Progressive teacher dropping for multi-teacher robustness."""
    def get_active_mask(self, step: int) -> list[bool]: ...
    def get_dropout_rate(self, step: int) -> float: ...

class HardExampleMiner:
    """Track per-sample losses and re-sample high-loss examples."""
    def update_losses(self, indices, losses) -> None: ...
    def get_difficulty_scores(self) -> torch.Tensor: ...
```

---

## forge.universal_distill

**Source**: `src/forge/universal_distill.py`

```python
class ConfidenceRouter(nn.Module):
    """Routes using student features + teacher confidence vectors."""

class UniversalDistillationLoss(nn.Module):
    """Multi-teacher KD with confidence routing."""
    def __init__(self, n_teachers, d_student, confidence_dim=7): ...
    def forward(self, student_actions, teacher_actions_list, ground_truth_actions,
                student_features, teacher_confidences) -> dict: ...

class UniversalRunner:
    """Full training loop with multi-teacher checkpointing."""

def plan_gpu_placement(teacher_names, n_gpus) -> dict: ...
```

---

## forge.modules

### BridgeAttention

**Source**: `src/forge/modules/bridge_attention.py`

```python
class BridgeAttention(nn.Module):
    def __init__(self, d_vision=1152, d_model=896, n_queries=64, n_heads=8, n_layers=4): ...
    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """(B, N_vis, d_vision) -> (B, n_queries, d_model)"""
    def param_count(self) -> int: ...
```

### Action Heads

```python
# Factory
from forge.modules.action_head_factory import create_action_head
head = create_action_head(config)  # Returns nn.Module based on config.action_head_type

# All heads share the same interface:
# forward(action_features: Tensor, gt_actions: Tensor | None = None) -> dict
# Returns: {"actions": Tensor, "loss": Tensor (training only)}
```

---

## forge.autosense

**Source**: `src/forge/autosense.py`

```python
def sense_vision_encoder(model_path: Path) -> dict | None:
    """Returns {d_output, n_tokens, patch_size, image_size}."""

def sense_language_model(model_path: Path) -> dict | None:
    """Returns {d_model, vocab_size, n_layers, n_heads}."""

def sense_teacher(model_path: Path) -> dict | None:
    """Returns {action_dim, action_horizon, param_count}."""

def autosense_config(model_dir, vision_name, lm_name) -> dict:
    """Returns config override dict (e.g., {bridge_d_vision: 1152})."""

def apply_autosense(config, model_dir) -> config:
    """Mutate config in-place with auto-detected values."""
```

---

## forge.backend

**Source**: `src/forge/backend.py`

```python
class BackendType(Enum):
    CUDA = "cuda"
    MLX = "mlx"
    CPU = "cpu"

def detect_backend() -> BackendType:
    """Auto-detect. Respects FORGE_DEVICE env var."""

def get_backend() -> TorchBackend | MLXBackend:
    """Singleton backend instance."""

class TorchBackend:
    def zeros(self, *shape) -> Tensor: ...
    def from_numpy(self, arr) -> Tensor: ...
    def to_numpy(self, tensor) -> ndarray: ...
    def to_device(self, tensor) -> Tensor: ...
    def get_device_info(self) -> DeviceInfo: ...
    def save(self, obj, path) -> None: ...
    def load(self, path) -> Any: ...
```

---

## forge.model_registry

**Source**: `src/forge/model_registry.py`

```python
class ModelRegistry:
    def __init__(self, registry_dir: str = "./outputs/registry"): ...
    def register(self, checkpoint_path, variant, config, training_report, metrics) -> entry: ...
    def list_models(self, variant=None, tag=None) -> list: ...
    def get(self, model_id: str) -> entry | None: ...
    def best(self, by="best_loss", variant=None, lower_is_better=True) -> entry | None: ...
    def promote(self, model_id, tag="production") -> entry | None: ...
    def compare(self, id1, id2) -> dict: ...
```

---

## forge.hyperparam

**Source**: `src/forge/hyperparam.py`

```python
class SearchSpace:
    def add_choice(self, name, values) -> SearchSpace: ...
    def add_range(self, name, low, high, log_scale=False) -> SearchSpace: ...
    def add_categorical(self, name, values) -> SearchSpace: ...

class HyperparamSearch:
    def __init__(self, space, objective="best_loss", results_dir="./outputs/hp"): ...
    def grid_search(self) -> list[Trial]: ...
    def random_search(self, n_trials, seed=42) -> list[Trial]: ...
    def apply_to_config(self, params, config) -> None: ...
    def start_trial(self, trial) -> None: ...
    def complete_trial(self, trial, objective_value) -> None: ...
    def best_trial(self) -> Trial: ...
    def top_trials(self, n=5) -> list[Trial]: ...
    def summary(self) -> dict: ...

def recommend_config(results_dir, objective="balanced", top_n=3) -> list[dict]: ...
```

---

## forge.auto_hyperparam

**Source**: `src/forge/auto_hyperparam.py`

```python
def run_auto_search(
    objective="balanced", n_trials=30, train_steps=100, device="cuda",
    model_dir=None, output_dir="./outputs/auto_hp", pruner="median",
    storage=None, wandb_project=None, wandb_entity=None,
) -> dict:
    """Full automated Optuna search. Returns results dict."""

def create_forge_study(storage=None) -> optuna.Study: ...
def export_best_yaml(study, output_path) -> None: ...
def get_search_summary(output_dir) -> dict | None: ...
```

---

## forge.finetune

**Source**: `src/forge/finetune.py`

```python
@dataclass
class FinetuneConfig:
    strategy: str = "lora"          # lora | action_head | full
    lr: float = 5e-5
    max_steps: int = 5000
    ewc_enabled: bool = True        # Elastic Weight Consolidation
    ewc_lambda: float = 1000.0
    replay_enabled: bool = True     # Experience replay
    replay_buffer_size: int = 5000
    replay_ratio: float = 0.2

class FinetuneTrainer:
    def __init__(self, student, config: FinetuneConfig, device="cuda"): ...
    def load_pretrained(self, path) -> None: ...
    def train(self, dataset) -> FinetuneReport: ...

class ReplayBuffer: ...
class EWCPenalty: ...
```

---

## forge.telemetry

**Source**: `src/forge/telemetry.py`

```python
class InferenceTelemetry:
    def __init__(self, config: TelemetryConfig): ...
    def record_inference(self, latency_ms: float) -> None: ...
    def record_action(self, action: np.ndarray) -> dict: ...
    def record_buffer(self, fill_level: float) -> None: ...
    def summary(self) -> dict: ...
    def export_json(self, path: str) -> None: ...
```

---

## forge.cross_embodiment

**Source**: `src/forge/cross_embodiment.py`

```python
@dataclass
class EmbodimentProfile:
    name: str
    action_dim: int
    joint_names: list[str]
    joint_min: list[float]
    joint_max: list[float]
    has_gripper: bool

class EmbodimentTransfer:
    def __init__(self, source, target, config: TransferConfig): ...
    def map_actions(self, source_actions: Tensor) -> Tensor: ...
    def info(self) -> dict: ...
```

---

## forge.metrics

**Source**: `src/forge/metrics.py`

```python
class TrainingMonitor:
    def __init__(self, log_dir="./logs", log_every=100): ...
    def record(self, metrics: dict, step: int) -> None: ...
    def get_summary(self) -> dict: ...
    def close(self) -> None: ...

class JSONLogger:
    @staticmethod
    def load(path) -> list[dict]: ...
```

---

## forge.export

```python
# MLX
from forge.export.mlx_export import export_mlx, load_mlx_weights, validate_mlx_export
mlx_path = export_mlx(model, output_dir, config=None)

# ONNX
from forge.export.onnx_export import export_onnx, validate_onnx
onnx_path = export_onnx(model, output_path, image_size=384, opset_version=19, optimize=True)

# TensorRT
from forge.export.tensorrt_export import export_tensorrt, check_tensorrt_available
engine_path = export_tensorrt(onnx_path, output_path, precision="fp16", workspace_mb=2048)
```

---

## forge.runtime

```python
from forge.runtime.async_engine import AsyncInferenceEngine, RuntimeConfig, ChunkBuffer

engine = AsyncInferenceEngine(model, RuntimeConfig(target_hz=50, action_horizon=8))
engine.start()
engine.submit_frame(image, instruction="pick up cup")
action = engine.get_action()  # Non-blocking; returns None when no action is buffered
status = engine.get_status()   # RuntimeStatus dataclass
engine.stop()
```

The async engine is an embedded Python runtime. The maintained public HTTP surface is
`forge.serve` and always requires a provenance-verified trained checkpoint:

```python
from forge.serve import create_app, start_server

app = create_app(
    checkpoint="outputs/checkpoints/final.pt",
    model_dir="models",
    device="cuda",
)

start_server(
    checkpoint="outputs/checkpoints/final.pt",
    host="0.0.0.0",
    port=8080,
    model_dir="models",
    device="cuda",
)
```

Maintained endpoints:

| Method | Path | Multipart contract | Response actions |
|--------|------|--------------------|------------------|
| GET | `/health` | None | Model/version/checkpoint and exact `action_horizon`/`action_dim` |
| POST | `/predict` | `image` file and required non-empty `instruction` | `(H,D)` |
| POST | `/batch_predict` | Repeated `images` files and required non-empty `instruction` | `(B,H,D)` |

---

## forge.eval

```python
from forge.eval.model_server import ForgeModelServer
server = ForgeModelServer(checkpoint_path="best.pt", variant="nano", device="cuda")
server.start(blocking=False)
result = server.predict({"images": {"base_camera": image}, "task_description": "pick up"})
server.stop()

from forge.eval.runner import EvalRunner
runner = EvalRunner(checkpoint_path="best.pt", variant="nano")
result = runner.run_benchmark("libero", episodes_per_task=20, max_tasks=10)
results = runner.run_all()
comparison = runner.compare("other.pt", benchmark="libero")
EvalRunner.pull_images()  # Pull Docker images

from forge.eval.results import EvalResult, load_results, results_to_table, append_to_report
results = load_results("./outputs/eval")
print(results_to_table(results))
append_to_report(result, "outputs/eval/report.md")
```

`results_to_table()` and `append_to_report()` retain `status` and bounded `error`
evidence; task failure rates and harness failures are reported as distinct outcomes.

---

## forge.web

```python
from forge.web.api import create_app
from forge.config import ForgeConfig

config = ForgeConfig.default()
app = create_app(config)

# Run with uvicorn
import uvicorn
uvicorn.run(app, host="0.0.0.0", port=3000)
```

**API Routes** (26 endpoints):

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Dashboard HTML |
| GET | `/api/status` | System status |
| GET | `/api/config` | Current config |
| PUT | `/api/config` | Update config |
| GET | `/api/teachers` | List teachers |
| POST | `/api/teachers/{name}/load` | Load teacher |
| POST | `/api/teachers/{name}/unload` | Unload teacher |
| GET | `/api/models` | List models |
| POST | `/api/train/start` | Start training |
| POST | `/api/train/stop` | Stop training |
| GET | `/api/train/status` | Training status |
| WS | `/api/train/stream` | Training metrics stream |
| POST | `/api/compress/start` | Start compression |
| GET | `/api/compress/status` | Compression status |
| POST | `/api/benchmarks/run` | Run benchmarks |
| GET | `/api/benchmarks` | Benchmark history |
| GET | `/api/benchmarks/{id}` | Single benchmark |
| GET | `/api/embodiments` | List embodiments |
| GET | `/api/embodiments/{name}` | Get embodiment |
| GET | `/api/embodiments/{name}/config` | Get YAML config |
| POST | `/api/predict` | Single prediction |
| GET | `/api/runtime/status` | Runtime status |
| WS | `/api/stream` | Inference stream |
| GET | `/api/experiments/auto_hp` | Auto-HP results |
| GET | `/api/eval/results` | Eval results |
| POST | `/api/demo/run` | Run demo |
