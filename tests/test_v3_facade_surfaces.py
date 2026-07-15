"""Contracts closing the remaining PRD-35 facade command surfaces."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from forge.training_runtime import atomic_write_json


def _heartbeat(run_dir: Path, **overrides) -> None:
    payload = {
        "status": "completed",
        "step": 2,
        "curriculum": {
            "enabled": False,
            "difficulty": None,
            "difficulty_metric": "loss",
            "initial_difficulty": 0.3,
            "final_difficulty": 1.0,
            "ramp_schedule": "linear",
            "ramp_steps": 100,
            "hard_example_mining": False,
            "hard_examples_seen": 0,
            "plateau_detection": False,
            "plateaus": 0,
            "teacher_dropout": False,
            "teacher_dropout_rate": None,
        },
        **overrides,
    }
    atomic_write_json(run_dir / "train_state.json", payload)


def test_curriculum_status_reports_disabled_run(tmp_path: Path) -> None:
    from forge.cli_v2 import curriculum_app

    run_dir = tmp_path / "run"
    _heartbeat(run_dir)
    result = CliRunner().invoke(
        curriculum_app,
        ["status", "--run-dir", str(run_dir), "--json"],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["enabled"] is False
    assert payload["difficulty"] is None
    assert payload["plateau_detection"] is False


def test_curriculum_status_marks_stale_process(tmp_path: Path) -> None:
    from forge.cli_v2 import curriculum_app

    run_dir = tmp_path / "run"
    _heartbeat(run_dir, status="running", pid=999_999_999)
    result = CliRunner().invoke(
        curriculum_app,
        ["status", "--run-dir", str(run_dir), "--json"],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["run_status"] == "stale"
    assert payload["process_running"] is False


def test_curriculum_status_rejects_legacy_and_missing_runs(tmp_path: Path) -> None:
    from forge.cli_v2 import curriculum_app

    legacy = tmp_path / "legacy"
    atomic_write_json(legacy / "train_state.json", {"status": "completed", "step": 1})
    legacy_result = CliRunner().invoke(
        curriculum_app,
        ["status", "--run-dir", str(legacy), "--json"],
    )
    assert legacy_result.exit_code == 1
    assert legacy_result.stdout == ""
    assert "legacy heartbeat" in json.loads(legacy_result.stderr)["error"]

    missing_result = CliRunner().invoke(
        curriculum_app,
        ["status", "--run-dir", str(tmp_path / "missing"), "--json"],
    )
    assert missing_result.exit_code == 2
    assert missing_result.stdout == ""
    assert "Training heartbeat not found" in json.loads(missing_result.stderr)["error"]


def test_unwired_duplicate_facades_are_not_registered() -> None:
    from typer.main import get_command

    from forge.cli_v2 import app

    runner = CliRunner()
    root_help = runner.invoke(app, ["--help"])
    assert root_help.exit_code == 0
    assert "universal-distill" not in root_help.output
    assert "export" not in get_command(app).commands

    profile_help = runner.invoke(app, ["profile", "--help"])
    assert profile_help.exit_code == 0
    assert "benchmark" not in profile_help.output

    assert runner.invoke(app, ["universal-distill", "start"]).exit_code == 2
    assert runner.invoke(app, ["profile", "benchmark"]).exit_code == 2
    assert runner.invoke(app, ["export"]).exit_code == 2


def test_report_requires_an_explicit_measured_artifact(tmp_path: Path) -> None:
    from typer.main import get_command

    from forge.cli_v2 import app

    runner = CliRunner()
    missing = runner.invoke(app, ["report"])
    assert missing.exit_code == 2
    report_command = get_command(app).commands["report"]
    results_option = next(
        parameter for parameter in report_command.params if "--results-file" in getattr(parameter, "opts", ())
    )
    assert results_option.required is True

    results = tmp_path / "results.json"
    output = tmp_path / "report.md"
    results.write_text('{"inference": {"latency_avg_ms": 4.5}}', encoding="utf-8")
    generated = runner.invoke(
        app,
        ["report", "--results-file", str(results), "--output", str(output)],
    )
    assert generated.exit_code == 0, generated.output
    assert output.is_file()
    assert "4.5 ms" in output.read_text(encoding="utf-8")

    invalid = tmp_path / "invalid.json"
    invalid.write_text("[]", encoding="utf-8")
    rejected = runner.invoke(app, ["report", "--results-file", str(invalid)])
    assert rejected.exit_code == 2
    assert "must contain a JSON object" in rejected.output


def test_packaged_export_usage_points_to_the_maintained_pipeline() -> None:
    export_modules = (
        Path("src/forge/export/onnx_export.py"),
        Path("src/forge/export/mlx_export.py"),
        Path("src/forge/export/tensorrt_export.py"),
    )
    for module in export_modules:
        source = module.read_text(encoding="utf-8")
        assert "forge export" not in source
        assert "forge pipeline --stage export" in source


def test_eval_server_has_single_maintained_surface() -> None:
    legacy_adapter = Path("src/forge/eval/forge_server.py")
    maintained_server = Path("src/forge/eval/model_server.py")
    cli_source = Path("src/forge/cli_commands/eval.py").read_text(encoding="utf-8")

    assert not legacy_adapter.exists()
    assert maintained_server.is_file()
    assert "from forge.eval.model_server import ForgeModelServer" in cli_source
