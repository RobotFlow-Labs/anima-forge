"""PRD-39 bounded, real-label quickstart contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from forge.cli_commands.quickstart import _quickstart_next_steps, run_quickstart
from forge.cli_v2 import app


def test_quickstart_next_compression_preserves_labels_and_training_summary(tmp_path: Path) -> None:
    output_dir = tmp_path / "quick start"
    checkpoint = output_dir / "checkpoints" / "final.pt"
    data_dir = tmp_path / "real labels"

    commands = _quickstart_next_steps(
        checkpoint=checkpoint,
        output_dir=output_dir,
        data_dir=data_dir,
        device="cuda",
    )

    assert commands[0].startswith("forge pipeline --config configs/forge_nano.yaml --stage compress")
    assert f"--checkpoint '{checkpoint}'" in commands[0]
    assert f"--data-dir '{data_dir}'" in commands[0]
    assert f"--output-dir '{tmp_path / 'quick start-compressed'}'" in commands[0]
    assert commands[1].startswith("forge benchmark run --checkpoint ")
    assert str(tmp_path / "quick start-compressed" / "compressed" / "qvla_4bit.pt") in commands[1]
    assert "--device cuda" in commands[1]
    assert "--data-dir /path/to/real-lerobot-dataset" in commands[1]
    assert "--instruction 'describe the real task'" in commands[1]


def test_quickstart_json_composes_real_bounded_pipeline(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model_dir = tmp_path / "models"
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "output"
    for name in ("Qwen--Qwen3-0.6B", "google--siglip2-so400m-patch14-384"):
        (model_dir / name).mkdir(parents=True)
    (data_dir / "teacher_labels").mkdir(parents=True)
    (data_dir / "teacher_labels" / "metadata.json").write_text("{}", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_run_quickstart(**kwargs):
        captured.update(kwargs)
        checkpoint = kwargs["output_dir"] / "checkpoints" / "final.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"checkpoint")
        return {
            "status": "completed",
            "device": kwargs["device"],
            "variant": "nano",
            "steps": kwargs["max_steps"],
            "elapsed_seconds": 2.0,
            "checkpoint": str(checkpoint),
            "pipeline_summary": str(kwargs["output_dir"] / "pipeline_summary.json"),
            "labels": {"path": str(kwargs["data_dir"]), "provenance": "real"},
            "doctor": {"status": "ok", "summary": {}},
            "models": {"requested": 2, "succeeded": 2, "failed": 0},
            "next_steps": ["forge benchmark run"],
        }

    monkeypatch.setattr("forge.cli_commands.quickstart.run_quickstart", fake_run_quickstart)
    result = CliRunner().invoke(
        app,
        [
            "quickstart",
            "--yes",
            "--device",
            "cpu",
            "--model-dir",
            str(model_dir),
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(output_dir),
            "--max-steps",
            "3",
            "--batch-size",
            "2",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["status"] == "completed"
    assert payload["labels"]["provenance"] == "real"
    assert captured["max_steps"] == 3
    assert captured["batch_size"] == 2
    assert captured["progress_callback"] is not None


def test_quickstart_json_requires_explicit_download_consent(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "quickstart",
            "--device",
            "cpu",
            "--model-dir",
            str(tmp_path / "missing-models"),
            "--data-dir",
            str(tmp_path / "missing-data"),
            "--json",
        ],
    )
    assert result.exit_code == 2
    assert result.stdout == ""
    assert "forge quickstart --yes --json" in json.loads(result.stderr)["error"]


def test_quickstart_json_wraps_asset_preflight_runtime_error(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "forge.cli_commands.quickstart._quickstart_assets",
        lambda: (_ for _ in ()).throw(RuntimeError("manifest incomplete")),
    )

    result = CliRunner().invoke(
        app,
        ["quickstart", "--yes", "--device", "cpu", "--model-dir", str(tmp_path), "--json"],
    )

    assert result.exit_code == 2
    assert result.stdout == ""
    assert json.loads(result.stderr)["error"] == "manifest incomplete"


def test_quickstart_stops_when_doctor_has_blocking_errors(tmp_path: Path, monkeypatch) -> None:
    fetched = False

    monkeypatch.setattr("forge.cli_commands.quickstart._quickstart_assets", lambda: ())
    monkeypatch.setattr(
        "forge.cli_commands.doctor.run_doctor",
        lambda **_kwargs: {
            "status": "error",
            "exit_code": 2,
            "summary": {"ok": 0, "warning": 0, "error": 1},
            "checks": [{"name": "datasets", "status": "error"}],
        },
    )

    def unexpected_fetch(*_args, **_kwargs):
        nonlocal fetched
        fetched = True

    monkeypatch.setattr("forge.cli_commands.fetch.fetch_assets", unexpected_fetch)

    with pytest.raises(RuntimeError, match="doctor found blocking.*datasets"):
        run_quickstart(
            device="cpu",
            model_dir=tmp_path / "models",
            data_dir=tmp_path / "data",
            output_dir=tmp_path / "output",
            max_steps=1,
            batch_size=1,
            sample_labels_repo="example/labels",
        )

    assert fetched is False
