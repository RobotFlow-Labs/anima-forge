"""PRD-39 first-run command surface."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml
from typer.testing import CliRunner

from forge import __version__
from forge.cli_commands.shared import load_forge_config
from forge.cli_v2 import app
from forge.cli_v2_root import _load_cli_config


def test_bare_forge_is_one_screen_and_mentions_five_useful_commands(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("FORGE_CONFIG_HOME", str(tmp_path / "config"))
    first = CliRunner().invoke(app, [])
    second = CliRunner().invoke(app, [])

    assert first.exit_code == 0
    assert len(first.output.splitlines()) <= 12
    for command in ("quickstart", "doctor", "pipeline", "models fetch", "config init"):
        assert command in first.output
    assert "--show-completion" in first.output
    assert "--show-completion" not in second.output


def test_config_init_emits_parseable_commented_v3_starter() -> None:
    result = CliRunner().invoke(app, ["config", "init"])

    assert result.exit_code == 0
    assert result.output.startswith("# FORGE v3 starter configuration")
    config = yaml.safe_load(result.output)
    assert config["student"]["language_model"] == "Qwen/Qwen3-0.6B"
    assert config["student"]["vision_encoder"] == "google/siglip2-so400m-patch14-384"
    assert config["quant"] == {"method": "qvla", "bits": 4}


def test_default_nano_config_loads_outside_source_checkout(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    config = _load_cli_config("configs/forge_nano.yaml")

    assert config.student.variant == "nano"
    assert config.paths.language_model == "Qwen--Qwen3-0.6B"


@pytest.mark.parametrize("config_path", ["configs/forge_nano.yaml", Path("configs/forge_nano.yaml")])
def test_shared_config_loader_resolves_packaged_default_outside_source_checkout(
    config_path: str | Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    config = load_forge_config(config_path, required=True)

    assert config.student.variant == "nano"
    assert config.paths.vision_encoder == "google--siglip2-so400m-patch14-384"


def test_info_rejects_explicit_missing_config_as_clean_json(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"

    result = CliRunner().invoke(app, ["info", "--config", str(missing), "--json"])

    assert result.exit_code == 2
    assert result.stdout == ""
    assert yaml.safe_load(result.stderr) == {"error": f"Config file not found: {missing}"}


def test_info_json_reports_runtime_version() -> None:
    result = CliRunner().invoke(app, ["info", "--json"])

    assert result.exit_code == 0, result.stderr or result.output
    payload = yaml.safe_load(result.stdout)
    assert payload["version"] == __version__


def test_pipeline_uses_packaged_default_config_outside_source_checkout(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "forge.pipeline.run_pipeline",
        lambda config, **kwargs: (
            captured.update(config=config, **kwargs)
            or {"status": "completed", "pipeline_summary_path": str(tmp_path / "summary.json")}
        ),
    )

    result = CliRunner().invoke(app, ["pipeline", "--device", "cpu"])

    assert result.exit_code == 0, result.output
    assert captured["config"].student.variant == "nano"


@pytest.mark.parametrize(
    "command",
    [
        ["pipeline", "--device", "cuda"],
        ["serve", "--device", "cuda", "--checkpoint", "missing.pt"],
    ],
)
def test_root_runtime_commands_wrap_unavailable_cuda(command: list[str], monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.delenv("FORGE_ALLOW_CPU_FALLBACK", raising=False)

    result = CliRunner().invoke(app, command)

    assert result.exit_code == 2
    assert "no CUDA device" in result.output
    assert "Traceback" not in result.output
