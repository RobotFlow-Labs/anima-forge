"""Artifact-backed benchmark CLI contracts."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from typer.testing import CliRunner

from forge.benchmark.metrics import BenchmarkReport
from forge.cli_commands.benchmark import _load_real_benchmark_input
from forge.cli_v2 import app


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))


class _RealDataset:
    provenance = {"kind": "real", "dataset": "lerobot/pusht", "samples": 1}

    def __init__(self, _path: Path, *, max_samples: int) -> None:
        assert max_samples == 1

    def __getitem__(self, _index: int) -> dict[str, torch.Tensor]:
        return {"image": torch.zeros(3, 8, 8)}


def test_benchmark_input_requires_real_instruction_when_dataset_has_no_text(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr("forge.data.lerobot_video_dataset.LeRobotVideoActionDataset", _RealDataset)

    with pytest.raises(ValueError, match="will not invent language input"):
        _load_real_benchmark_input(tmp_path, instruction=None)

    images, instruction, provenance = _load_real_benchmark_input(
        tmp_path,
        instruction="push the block to the target",
    )
    assert images.shape == (1, 3, 8, 8)
    assert instruction == "push the block to the target"
    assert provenance["kind"] == "real"
    assert provenance["instruction_source"] == "cli"


def test_benchmark_run_requires_trained_checkpoint(monkeypatch) -> None:
    monkeypatch.delenv("FORGE_ALLOW_MOCK", raising=False)
    result = CliRunner().invoke(
        app,
        ["benchmark", "run", "--device", "cpu", "--json"],
    )
    assert result.exit_code == 2
    assert result.stdout == ""
    assert "benchmark requires a trained --checkpoint" in json.loads(result.stderr)["error"]


def test_benchmark_run_rejects_explicit_missing_config_as_clean_json(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"config validation happens before checkpoint loading")
    missing = tmp_path / "missing.yaml"

    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "run",
            "--config",
            str(missing),
            "--checkpoint",
            str(checkpoint),
            "--device",
            "cpu",
            "--json",
        ],
    )

    assert result.exit_code == 2
    assert result.stdout == ""
    assert json.loads(result.stderr) == {"error": f"Config file not found: {missing}"}


def test_benchmark_run_loads_checkpoint_and_stamps_report(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checkpoint = tmp_path / "trained.pt"
    checkpoint.write_bytes(b"real checkpoint bytes")
    output = tmp_path / "report.json"
    provenance = {"vision": "real", "language": "real", "labels": "real"}
    captured: dict[str, object] = {}

    def fake_load(config_path, **kwargs):
        captured["config"] = config_path
        captured.update(kwargs)
        return SimpleNamespace(student=SimpleNamespace()), TinyModel(), provenance

    class FakeRunner:
        def __init__(self, model, config, device):
            captured["model"] = model
            captured["device"] = device

        def run(self, **kwargs):
            captured.update(kwargs)
            return BenchmarkReport(model_name="FORGE-nano", variant="nano", device="cpu")

        def export(self, report, path):
            Path(path).write_text(json.dumps(report.to_dict()), encoding="utf-8")

    monkeypatch.setattr("forge.cli_commands.quantize.load_student_for_quant", fake_load)
    monkeypatch.setattr("forge.benchmark.runner.BenchmarkRunner", FakeRunner)
    monkeypatch.setattr(
        "forge.cli_commands.benchmark._load_real_benchmark_input",
        lambda data_dir, *, instruction: (
            torch.zeros(1, 3, 8, 8),
            instruction,
            {"kind": "real", "dataset": data_dir.name},
        ),
    )

    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "run",
            "--config",
            "configs/forge_nano.yaml",
            "--checkpoint",
            str(checkpoint),
            "--device",
            "cpu",
            "--samples",
            "3",
            "--duration",
            "0.25",
            "--data-dir",
            str(tmp_path / "real-data"),
            "--instruction",
            "move the block",
            "--output",
            str(output),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["source_checkpoint"] == str(checkpoint.resolve())
    assert payload["artifact_size_mb"] == checkpoint.stat().st_size / 1e6
    assert payload["provenance"] == provenance
    assert payload["execution"]["schema"] == "forge.benchmark-execution.v1"
    assert payload["execution"]["command"] == "run"
    assert payload["execution"]["requested_device"] == "cpu"
    assert payload["execution"]["resolved_device"] == "cpu"
    assert payload["execution"]["git_sha"]
    assert json.loads(output.read_text(encoding="utf-8")) == payload
    assert captured["checkpoint"] == str(checkpoint)
    assert captured["require_trained_checkpoint"] is True
    assert captured["protected_action"] == "benchmark"
    assert captured["n_latency_samples"] == 3
    assert captured["throughput_duration"] == 0.25
    assert captured["language_text"] == "move the block"
    assert captured["input_provenance"] == {"kind": "real", "dataset": "real-data"}


@pytest.mark.parametrize("subcommand", ["matrix", "run"])
def test_benchmark_json_wraps_strict_device_runtime_error(
    subcommand: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.delenv("FORGE_ALLOW_CPU_FALLBACK", raising=False)
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    command = ["benchmark", subcommand]
    if subcommand == "matrix":
        command.append(str(manifest))
    command.extend(["--device", "cuda", "--json"])

    result = CliRunner().invoke(app, command)

    assert result.exit_code == 2
    assert result.stdout == ""
    assert "no CUDA device" in json.loads(result.stderr)["error"]
