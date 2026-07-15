"""Exhaustive subprocess stream contract for every public ``--json`` command."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
from typer.main import get_command

from forge.cli import app
from forge.training_runtime import atomic_write_json


@dataclass(frozen=True)
class JsonCase:
    path: str
    args: tuple[str, ...]
    stream: str = "stdout"
    exit_codes: tuple[int, ...] = (0,)


CASES = (
    JsonCase("agent", ("agent", "--no-jobs", "--json")),
    JsonCase("agent-top", ("agent-top", "--no-jobs", "--json")),
    JsonCase("autosense", ("autosense", "--model-dir", "{tmp}/missing-models", "--json"), "stderr", (2,)),
    JsonCase(
        "benchmark all",
        ("benchmark", "all", "--results-dir", "{tmp}/benchmarks", "--device", "invalid", "--json"),
        "stderr",
        (2,),
    ),
    JsonCase(
        "benchmark aggregate",
        ("benchmark", "aggregate", "--results-dir", "{tmp}/benchmarks", "--json"),
        "stderr",
        (2,),
    ),
    JsonCase("benchmark compare", ("benchmark", "compare", "{tmp}/report-a.json", "{tmp}/report-b.json", "--json")),
    JsonCase("benchmark list", ("benchmark", "list", "--json")),
    JsonCase(
        "benchmark matrix",
        ("benchmark", "matrix", "{tmp}/missing-manifest.json", "--json"),
        "stderr",
        (2,),
    ),
    JsonCase(
        "benchmark run",
        ("benchmark", "run", "--device", "cpu", "--checkpoint", "{tmp}/missing.pt", "--json"),
        "stderr",
        (2,),
    ),
    JsonCase("benchmark suite", ("benchmark", "suite", "missing", "--json"), "stderr", (2,)),
    JsonCase("curriculum simulate", ("curriculum", "simulate", "--steps", "4", "--json")),
    JsonCase("curriculum status", ("curriculum", "status", "--run-dir", "{tmp}/run", "--json")),
    JsonCase("doctor", ("doctor", "--json"), "stdout", (0, 1, 2)),
    JsonCase("embodiment list", ("embodiment", "list", "--json")),
    JsonCase("embodiments list", ("embodiments", "list", "--json")),
    JsonCase("embodyments list", ("embodyments", "list", "--json")),
    JsonCase(
        "eval compare",
        ("eval", "compare", "--a", "{tmp}/missing-a.pt", "--b", "{tmp}/missing-b.pt", "--device", "cpu", "--json"),
        "stderr",
        (2,),
    ),
    JsonCase("eval results", ("eval", "results", "--output-dir", "{tmp}/eval", "--json")),
    JsonCase(
        "eval run",
        ("eval", "run", "libero", "--checkpoint", "{tmp}/missing.pt", "--device", "cpu", "--json"),
        "stderr",
        (2,),
    ),
    JsonCase(
        "eval run-all",
        ("eval", "run-all", "--checkpoint", "{tmp}/missing.pt", "--device", "cpu", "--json"),
        "stderr",
        (2,),
    ),
    JsonCase(
        "eval smoke", ("eval", "smoke", "--checkpoint", "{tmp}/missing.pt", "--device", "cpu", "--json"), "stderr", (2,)
    ),
    JsonCase("finetune list", ("finetune", "list", "--output-dir", "{tmp}/finetune", "--json")),
    JsonCase("finetune status", ("finetune", "status", "--output-dir", "{tmp}/finetune", "--json")),
    JsonCase(
        "hyperparam auto",
        ("hyperparam", "auto", "--show-best", "--output-dir", "{tmp}/auto-hp", "--json"),
        "stderr",
        (1,),
    ),
    JsonCase("hyperparam recommend", ("hyperparam", "recommend", "--results-dir", "{tmp}/recommend", "--json")),
    JsonCase("hyperparam status", ("hyperparam", "status", "--results-dir", "{tmp}/hyperparam", "--json")),
    JsonCase("hyperparam top", ("hyperparam", "top", "--results-dir", "{tmp}/hyperparam", "--json")),
    JsonCase("info", ("info", "--json")),
    JsonCase(
        "metrics export",
        ("metrics", "export", "--log-dir", "{tmp}/logs", "--output", "{tmp}/metrics-export.json", "--json"),
    ),
    JsonCase("metrics summary", ("metrics", "summary", "--log-dir", "{tmp}/logs", "--json")),
    JsonCase("models best", ("models", "best", "--registry-dir", "{tmp}/registry", "--json"), "stderr", (1,)),
    JsonCase(
        "models compare",
        ("models", "compare", "missing-a", "missing-b", "--registry-dir", "{tmp}/registry", "--json"),
        "stderr",
        (2,),
    ),
    JsonCase(
        "models fetch",
        ("models", "fetch", "--model-dir", "{tmp}/models", "--cache-dir", "{tmp}/cache", "--json"),
        "stderr",
        (2,),
    ),
    JsonCase("models list", ("models", "list", "--registry-dir", "{tmp}/registry", "--json")),
    JsonCase(
        "models promote", ("models", "promote", "missing", "--registry-dir", "{tmp}/registry", "--json"), "stderr", (2,)
    ),
    JsonCase(
        "models show", ("models", "show", "missing", "--registry-dir", "{tmp}/registry", "--json"), "stderr", (2,)
    ),
    JsonCase("profile card", ("profile", "card", "--variant", "nano", "--json")),
    JsonCase("profile recommend", ("profile", "recommend", "--variant", "nano", "--json")),
    JsonCase("profile vram", ("profile", "vram", "--variant", "nano", "--json")),
    JsonCase(
        "quickstart",
        (
            "quickstart",
            "--device",
            "cpu",
            "--model-dir",
            "{tmp}/missing-quickstart-models",
            "--data-dir",
            "{tmp}/missing-quickstart-data",
            "--json",
        ),
        "stderr",
        (2,),
    ),
    JsonCase(
        "quantize bench",
        ("quantize", "bench", "--device", "cpu", "--checkpoint", "{tmp}/missing.pt", "--json"),
        "stderr",
        (2,),
    ),
    JsonCase(
        "quantize run",
        ("quantize", "run", "--device", "cpu", "--checkpoint", "{tmp}/missing.pt", "--json"),
        "stderr",
        (2,),
    ),
    JsonCase("status", ("status", "--json")),
    JsonCase("students list", ("students", "list", "--registry-dir", "{tmp}/registry", "--json")),
    JsonCase(
        "students package",
        (
            "students",
            "package",
            "{tmp}/missing.pt",
            "--training-summary",
            "{tmp}/missing-summary.json",
            "--output-dir",
            "{tmp}/hub-package",
            "--json",
        ),
        "stderr",
        (2,),
    ),
    JsonCase("teacher list", ("teacher", "list", "--json")),
    JsonCase("telemetry summary", ("telemetry", "summary", "--export-path", "{tmp}/telemetry.json", "--json")),
    JsonCase("top", ("top", "--no-jobs", "--json")),
    JsonCase("top-agent", ("top-agent", "--no-jobs", "--json")),
    JsonCase(
        "train start",
        (
            "train",
            "start",
            "--device",
            "cpu",
            "--max-steps",
            "1",
            "--output-dir",
            "{tmp}/train",
            "--data-dir",
            "{tmp}/missing-data",
            "--json",
        ),
        "stderr",
        (2,),
    ),
    JsonCase("train status", ("train", "status", "--output-dir", "{tmp}/missing-runs", "--json")),
    JsonCase("train stop", ("train", "stop", "--output-dir", "{tmp}/missing-runs", "--json"), "stderr", (2,)),
    JsonCase("transfer info", ("transfer", "info", "--source", "missing", "--json"), "stderr", (2,)),
)


def _registered_json_paths() -> set[str]:
    root = get_command(app)
    paths: set[str] = set()

    def walk(command: object, prefix: tuple[str, ...]) -> None:
        children = getattr(command, "commands", None)
        if isinstance(children, dict):
            for name, child in children.items():
                walk(child, (*prefix, name))
            return
        if any(
            getattr(parameter, "name", None) == "output_json" or "--json" in getattr(parameter, "opts", ())
            for parameter in getattr(command, "params", ())
        ):
            paths.add(" ".join(prefix))

    walk(root, ())
    return paths


@pytest.fixture
def json_workspace(tmp_path: Path) -> Path:
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "metrics.jsonl").write_text(
        '{"step": 1, "loss": 0.5}\n',
        encoding="utf-8",
    )
    (tmp_path / "report-a.json").write_text('{"model_name": "a"}\n', encoding="utf-8")
    (tmp_path / "report-b.json").write_text('{"model_name": "b"}\n', encoding="utf-8")
    (tmp_path / "telemetry.json").write_text('{"latency_ms": 4.5}\n', encoding="utf-8")
    atomic_write_json(
        tmp_path / "run" / "train_state.json",
        {
            "status": "completed",
            "step": 1,
            "curriculum": {
                "enabled": False,
                "difficulty": None,
                "difficulty_metric": "loss",
                "initial_difficulty": 0.3,
                "final_difficulty": 1.0,
                "ramp_schedule": "linear",
                "ramp_steps": 10,
                "hard_example_mining": False,
                "hard_examples_seen": 0,
                "plateau_detection": False,
                "plateaus": 0,
                "teacher_dropout": False,
                "teacher_dropout_rate": None,
            },
        },
    )
    return tmp_path


def test_subprocess_matrix_covers_every_registered_json_command() -> None:
    matrix_paths = {case.path for case in CASES}
    assert len(CASES) == len(matrix_paths)
    assert matrix_paths == _registered_json_paths()


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.path.replace(" ", "-"))
def test_every_json_command_is_one_parseable_subprocess_document(
    case: JsonCase,
    json_workspace: Path,
) -> None:
    args = [item.format(tmp=json_workspace) for item in case.args]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["FORGE_ALLOW_CPU_FALLBACK"] = "1"
    result = subprocess.run(
        [sys.executable, "-m", "forge.cli_v2", *args],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        capture_output=True,
        text=True,
        timeout=55,
        check=False,
    )
    assert result.returncode in case.exit_codes, (
        f"{case.path}: exit={result.returncode}\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    selected = result.stdout if case.stream == "stdout" else result.stderr
    other = result.stderr if case.stream == "stdout" else result.stdout
    payload = json.loads(selected)
    assert payload is not None
    if case.stream == "stderr":
        assert other == ""
        assert set(payload) == {"error"}
