"""CLI command modules for FORGE."""

from .benchmark import benchmark_app
from .config_cli import config_app
from .curriculum import curriculum_app
from .demo import demo_app
from .doctor import doctor_command
from .embodiment import embodiment_app
from .eval import eval_app
from .fetch import models_fetch
from .finetune import finetune_app
from .hyperparam import hyperparam_app
from .logging_config import setup_cli_logging
from .metrics import metrics_app
from .models import models_app
from .profile import profile_app
from .quantize import quantize_app
from .quickstart import quickstart_command
from .shared import get_json_payload, load_forge_config
from .students import students_app
from .teacher import teacher_app
from .telemetry import telemetry_app
from .top import status_command, top, top_agent
from .train import train_app
from .transfer import transfer_app

__all__ = [
    "benchmark_app",
    "config_app",
    "curriculum_app",
    "demo_app",
    "doctor_command",
    "embodiment_app",
    "finetune_app",
    "telemetry_app",
    "transfer_app",
    "eval_app",
    "models_fetch",
    "hyperparam_app",
    "setup_cli_logging",
    "metrics_app",
    "models_app",
    "students_app",
    "profile_app",
    "quantize_app",
    "quickstart_command",
    "get_json_payload",
    "load_forge_config",
    "status_command",
    "teacher_app",
    "top",
    "top_agent",
    "train_app",
]
