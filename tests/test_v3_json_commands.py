"""Strict JSON stream contracts for v3 CLI command groups."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
from typer.testing import CliRunner

from forge.cli import app
from forge.cli_commands.eval import eval_app
from forge.cli_commands.hyperparam import hyperparam_app
from forge.cli_commands.profile import profile_app


@pytest.fixture(autouse=True)
def _disable_cli_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep tests focused on command payloads rather than configured log handlers."""
    monkeypatch.setattr("forge.cli_v2_root.setup_cli_logging", lambda **_kwargs: None)


def _assert_json_success(result) -> object:
    assert result.exit_code == 0, result.output
    assert result.stdout.lstrip().startswith(("{", "[")), result.stdout
    return json.loads(result.stdout)


def _assert_json_error(result, exit_code: int) -> dict:
    assert result.exit_code == exit_code, result.output
    assert result.stdout == ""
    payload = json.loads(result.stderr)
    assert set(payload) == {"error"}
    return payload


@pytest.mark.parametrize(
    "command",
    [
        ["eval", "smoke", "--checkpoint", "{checkpoint}", "--json"],
        ["eval", "run", "libero", "--checkpoint", "{checkpoint}", "--json"],
        ["hyperparam", "auto", "--trials", "1", "--steps", "1", "--json"],
        ["quantize", "run", "--checkpoint", "{checkpoint}", "--method", "qvla", "--json"],
    ],
)
def test_runtime_preflight_failures_are_one_stderr_json_document(
    command: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"not loaded because device preflight fails first")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("FORGE_DEVICE", "cuda")
    monkeypatch.delenv("FORGE_ALLOW_CPU_FALLBACK", raising=False)

    args = [part.format(checkpoint=checkpoint) for part in command]
    result = CliRunner().invoke(app, args)

    payload = _assert_json_error(result, 2)
    assert "no CUDA device" in payload["error"]
    assert "Traceback" not in result.stderr


def test_cheap_registered_json_commands_emit_one_document(tmp_path: Path) -> None:
    telemetry_path = tmp_path / "telemetry.json"
    telemetry_path.write_text('{"latency_ms": 4.5}', encoding="utf-8")

    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "metrics.jsonl").write_text(
        '{"step": 1, "loss": 0.5}\n',
        encoding="utf-8",
    )

    report_a = tmp_path / "report-a.json"
    report_b = tmp_path / "report-b.json"
    report_a.write_text('{"model_name": "a"}', encoding="utf-8")
    report_b.write_text('{"model_name": "b"}', encoding="utf-8")

    commands = [
        ["info", "--json"],
        ["teacher", "list", "--json"],
        ["embodiment", "list", "--json"],
        ["curriculum", "simulate", "--steps", "8", "--json"],
        ["finetune", "status", "--output-dir", str(tmp_path / "finetune"), "--json"],
        ["finetune", "list", "--output-dir", str(tmp_path / "finetune"), "--json"],
        ["models", "list", "--registry-dir", str(tmp_path / "registry"), "--json"],
        ["eval", "results", "--output-dir", str(tmp_path / "eval"), "--json"],
        ["hyperparam", "status", "--results-dir", str(tmp_path / "hp"), "--json"],
        ["hyperparam", "top", "--results-dir", str(tmp_path / "hp"), "--json"],
        ["telemetry", "summary", "--export-path", str(telemetry_path), "--json"],
        ["metrics", "summary", "--log-dir", str(log_dir), "--json"],
        ["benchmark", "compare", str(report_a), str(report_b), "--json"],
        ["transfer", "info", "--source", "franka", "--target", "ur5e", "--json"],
        ["status", "--max-output-bytes", "1", "--json"],
        ["top", "--max-output-bytes", "1", "--json"],
    ]

    runner = CliRunner()
    for command in commands:
        payload = _assert_json_success(runner.invoke(app, command))
        assert payload is not None


def test_eval_smoke_redirects_loader_chatter_and_sanitizes_non_finite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NoisyEvalRunner:
        def __init__(self, **_kwargs) -> None:
            print("eval loader chatter")

        def run_benchmark(self, **_kwargs) -> dict:
            print("eval runtime chatter")
            return {"status": "completed", "success_rate": math.inf}

    monkeypatch.setattr("forge.eval.runner.EvalRunner", NoisyEvalRunner)
    monkeypatch.setattr("forge.cli_commands.eval._resolve_checkpoint", lambda *_args, **_kwargs: "model.pt")

    result = CliRunner().invoke(eval_app, ["smoke", "--device", "cpu", "--json"])

    payload = _assert_json_success(result)
    assert payload["success_rate"] is None
    assert "note" in payload
    assert "eval loader chatter" in result.stderr
    assert "Running LIBERO smoke test" not in result.stdout


def test_eval_smoke_discards_chatter_before_strict_json_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingEvalRunner:
        def __init__(self, **_kwargs) -> None:
            print("loader chatter that must not corrupt the error")

        def run_benchmark(self, **_kwargs) -> dict:
            print("runtime chatter that must not corrupt the error")
            raise RuntimeError("evaluation runtime unavailable")

    monkeypatch.setattr("forge.eval.runner.EvalRunner", FailingEvalRunner)
    monkeypatch.setattr("forge.cli_commands.eval._resolve_checkpoint", lambda *_args, **_kwargs: "model.pt")

    result = CliRunner().invoke(eval_app, ["smoke", "--device", "cpu", "--json"])

    assert _assert_json_error(result, 2) == {"error": "evaluation runtime unavailable"}


def test_hyperparam_auto_redirects_progress_and_sanitizes_non_finite(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: dict[str, object] = {}

    def noisy_search(**kwargs) -> dict:
        observed.update(kwargs)
        print("optuna progress")
        return {"completed": 1, "best_score": math.nan}

    monkeypatch.setattr("forge.auto_hyperparam.run_auto_search", noisy_search)

    result = CliRunner().invoke(
        hyperparam_app,
        [
            "auto",
            "--trials",
            "1",
            "--seed",
            "7",
            "--steps",
            "1",
            "--device",
            "cpu",
            "--output-dir",
            str(tmp_path),
            "--json",
        ],
    )

    payload = _assert_json_success(result)
    assert payload["best_score"] is None
    assert "note" in payload
    assert "optuna progress" in result.stderr
    assert "FORGE Auto-HP Search" not in result.stdout
    assert observed["random_seed"] == 7


def test_hyperparam_auto_discards_progress_before_strict_json_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing_search(**_kwargs) -> dict:
        print("optuna progress that must not corrupt the error")
        raise RuntimeError("search runtime unavailable")

    monkeypatch.setattr("forge.auto_hyperparam.run_auto_search", failing_search)

    result = CliRunner().invoke(
        hyperparam_app,
        ["auto", "--trials", "1", "--steps", "1", "--device", "cpu", "--json"],
    )

    assert _assert_json_error(result, 2) == {"error": "search runtime unavailable"}


def test_hyperparam_auto_exports_from_selected_storage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import forge.auto_hyperparam as auto_hyperparam

    observed: dict[str, object] = {}
    selected_storage = f"sqlite:///{tmp_path / 'selected.db'}"
    export_path = tmp_path / "best.yaml"
    sentinel_study = object()

    monkeypatch.setattr(
        auto_hyperparam,
        "run_auto_search",
        lambda **_kwargs: {"best_trial": {"number": 0}},
    )

    def fake_create_study(**kwargs):
        observed.update(kwargs)
        return sentinel_study

    monkeypatch.setattr(auto_hyperparam, "create_forge_study", fake_create_study)
    monkeypatch.setattr(
        auto_hyperparam,
        "export_best_yaml",
        lambda study, path: observed.update(export_study=study, export_path=path),
    )

    result = CliRunner().invoke(
        hyperparam_app,
        [
            "auto",
            "--trials",
            "1",
            "--steps",
            "1",
            "--device",
            "cpu",
            "--allow-mock",
            "--storage",
            selected_storage,
            "--export-yaml",
            str(export_path),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert observed["storage"] == selected_storage
    assert observed["random_seed"] == 42
    assert observed["export_study"] is sentinel_study
    assert observed["export_path"] == str(export_path)


def test_profile_card_redirects_generation_and_artifact_chatter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeCard:
        def save_json(self, output: str) -> None:
            print("profile save chatter")
            Path(output).write_text("{}", encoding="utf-8")

        def to_dict(self) -> dict:
            return {"score": math.inf}

    class NoisyProfiler:
        def __init__(self, **_kwargs) -> None:
            print("profile init chatter")

        def generate_card(self, **_kwargs) -> FakeCard:
            print("profile generation chatter")
            return FakeCard()

        def generate_markdown(self, _card: FakeCard) -> str:
            print("profile markdown chatter")
            return "# card\n"

    monkeypatch.setattr("forge.profiler.FORGEProfiler", NoisyProfiler)
    output = tmp_path / "card.json"
    markdown = tmp_path / "card.md"

    result = CliRunner().invoke(
        profile_app,
        ["card", "--output", str(output), "--markdown", str(markdown), "--json"],
    )

    payload = _assert_json_success(result)
    assert payload["score"] is None
    assert "note" in payload
    assert "profile generation chatter" in result.stderr
    assert output.is_file()
    assert markdown.is_file()


@pytest.mark.parametrize(
    ("args", "exit_code"),
    [
        (["telemetry", "summary", "--json"], 2),
        (["metrics", "summary", "--log-dir", "{tmp}/missing-logs", "--json"], 2),
        (["models", "show", "missing", "--registry-dir", "{tmp}/registry", "--json"], 2),
        (["models", "best", "--registry-dir", "{tmp}/registry", "--json"], 1),
        (["transfer", "info", "--source", "missing", "--json"], 2),
        (["benchmark", "compare", "{tmp}/missing-a", "{tmp}/missing-b", "--json"], 2),
        (["curriculum", "simulate", "--schedule", "invalid", "--json"], 2),
        (["quantize", "bench", "--device", "cpu", "--checkpoint", "{tmp}/missing.pt", "--json"], 2),
    ],
)
def test_clear_cli_errors_use_stderr_json(
    args: list[str],
    exit_code: int,
    tmp_path: Path,
) -> None:
    resolved = [item.format(tmp=tmp_path) for item in args]
    payload = _assert_json_error(CliRunner().invoke(app, resolved), exit_code)
    assert payload["error"]


def test_malformed_telemetry_uses_stderr_json(tmp_path: Path) -> None:
    telemetry_path = tmp_path / "broken.json"
    telemetry_path.write_text("{broken", encoding="utf-8")

    result = CliRunner().invoke(
        app,
        ["telemetry", "summary", "--export-path", str(telemetry_path), "--json"],
    )

    payload = _assert_json_error(result, 2)
    assert "Could not read telemetry JSON" in payload["error"]
