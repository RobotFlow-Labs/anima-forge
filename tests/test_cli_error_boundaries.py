"""Regression coverage for public CLI setup and runtime error boundaries."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from forge.cli_commands.universal_distill import universal_distill_app
from forge.cli_v2 import app

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_forge(*args: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    return subprocess.run(
        [sys.executable, "-m", "forge.cli_v2", *args],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_info_yaml_parser_failure_is_one_strict_json_stderr_document(tmp_path: Path) -> None:
    malformed = tmp_path / "malformed.yaml"
    malformed.write_text("student: [\n", encoding="utf-8")

    result = _run_forge("info", "--config", str(malformed), "--json")

    assert result.returncode == 2
    assert result.stdout == ""
    payload = json.loads(result.stderr)
    assert set(payload) == {"error"}
    assert "parsing a flow node" in payload["error"]
    assert "Traceback" not in result.stderr


def test_info_non_json_parser_failure_uses_stderr_without_traceback(tmp_path: Path) -> None:
    malformed = tmp_path / "malformed.yaml"
    malformed.write_text("student: [\n", encoding="utf-8")

    result = _run_forge("info", "--config", str(malformed))

    assert result.returncode == 2
    assert result.stdout == ""
    assert "parsing a flow node" in result.stderr
    assert "Traceback" not in result.stderr


def test_eval_serve_invalid_device_is_stderr_exit_two(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "student.pt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr("forge.cli_commands.eval._verify_eval_serve_checkpoint", lambda *_args, **_kwargs: False)

    result = CliRunner().invoke(
        app,
        ["eval", "serve", "--checkpoint", str(checkpoint), "--device", "cuda:-1"],
    )

    assert result.exit_code == 2
    assert result.stdout == ""
    assert "Unsupported CUDA device" in result.stderr
    assert "Traceback" not in result.stderr


def test_eval_serve_runtime_failure_is_contained(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "student.pt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr("forge.cli_commands.eval._verify_eval_serve_checkpoint", lambda *_args, **_kwargs: False)

    class Server:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def start(self, *, blocking: bool) -> None:
            assert blocking is True
            raise RuntimeError("server bind exploded")

    monkeypatch.setattr("forge.eval.model_server.ForgeModelServer", Server)

    result = CliRunner().invoke(
        app,
        ["eval", "serve", "--checkpoint", str(checkpoint), "--device", "cpu"],
    )

    assert result.exit_code == 2
    assert result.stdout == ""
    assert "server bind exploded" in result.stderr
    assert "Traceback" not in result.stderr


def _universal_app() -> typer.Typer:
    host = typer.Typer()
    host.add_typer(universal_distill_app, name="universal-distill")
    return host


def test_universal_distill_device_failure_is_strict_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "forge.cli_commands.universal_distill.resolve_runtime_device",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("device setup exploded")),
    )

    result = CliRunner().invoke(_universal_app(), ["universal-distill", "start", "--json"])

    assert result.exit_code == 2
    assert result.stdout == ""
    assert json.loads(result.stderr) == {"error": "device setup exploded"}
    assert "Traceback" not in result.stderr


def test_universal_distill_config_failure_is_strict_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "forge.config.ForgeConfig.default",
        lambda: (_ for _ in ()).throw(ValueError("config setup exploded")),
    )

    result = CliRunner().invoke(
        _universal_app(),
        ["universal-distill", "start", "--device", "cpu", "--json"],
    )

    assert result.exit_code == 2
    assert result.stdout == ""
    assert json.loads(result.stderr) == {"error": "config setup exploded"}
    assert "Traceback" not in result.stderr
