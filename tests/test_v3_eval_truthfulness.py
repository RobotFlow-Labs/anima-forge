"""Truthful checkpoint and failed-result contracts for evaluation commands."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from forge.cli_commands.eval import eval_app


@pytest.mark.parametrize(
    "args",
    [
        ["run", "libero", "--checkpoint", "{missing}", "--device", "cpu", "--json"],
        ["run-all", "--checkpoint", "{missing}", "--device", "cpu", "--json"],
        ["compare", "--a", "{missing}", "--b", "{missing2}", "--device", "cpu", "--json"],
        ["smoke", "--checkpoint", "{missing}", "--device", "cpu", "--json"],
    ],
)
def test_eval_json_rejects_missing_checkpoints(args: list[str], tmp_path: Path) -> None:
    resolved = [item.format(missing=tmp_path / "missing.pt", missing2=tmp_path / "missing-2.pt") for item in args]
    result = CliRunner().invoke(eval_app, resolved)
    assert result.exit_code == 2
    assert result.stdout == ""
    assert "Checkpoint not found" in json.loads(result.stderr)["error"]


def test_eval_run_failed_result_exits_two(tmp_path: Path, monkeypatch) -> None:
    from forge.eval.runner import EvalRunner

    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(
        EvalRunner,
        "run_benchmark",
        lambda *_args, **_kwargs: {"status": "failed", "error": "Docker unavailable"},
    )

    result = CliRunner().invoke(
        eval_app,
        ["run", "libero", "--checkpoint", str(checkpoint), "--device", "cpu", "--json"],
    )
    assert result.exit_code == 2
    assert result.stdout == ""
    assert json.loads(result.stderr) == {"error": "Docker unavailable"}


def test_eval_run_no_results_exits_two(tmp_path: Path, monkeypatch) -> None:
    from forge.eval.runner import EvalRunner

    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(
        EvalRunner,
        "run_benchmark",
        lambda *_args, **_kwargs: {"status": "no_results", "error": "No JSON result files found"},
    )

    result = CliRunner().invoke(
        eval_app,
        ["run", "libero", "--checkpoint", str(checkpoint), "--device", "cpu", "--json"],
    )
    assert result.exit_code == 2
    assert json.loads(result.stderr) == {"error": "No JSON result files found"}


def test_eval_compare_propagates_failed_side(tmp_path: Path, monkeypatch) -> None:
    from forge.eval.runner import EvalRunner

    checkpoint_a = tmp_path / "a.pt"
    checkpoint_b = tmp_path / "b.pt"
    checkpoint_a.write_bytes(b"a")
    checkpoint_b.write_bytes(b"b")

    def fake_run(self, *_args, **_kwargs):
        if self.checkpoint_path == str(checkpoint_a):
            return {"status": "failed", "error": "server failed"}
        return {"status": "completed", "success_rate": 0.5}

    monkeypatch.setattr(EvalRunner, "run_benchmark", fake_run)
    comparison = EvalRunner(
        checkpoint_path=str(checkpoint_a),
        device="cpu",
        output_dir=str(tmp_path / "eval"),
    ).compare(str(checkpoint_b))
    assert comparison["status"] == "failed"
    assert comparison["error"] == "server failed"
    assert "delta_success_rate" not in comparison
