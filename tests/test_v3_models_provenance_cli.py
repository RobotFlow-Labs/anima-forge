"""Public model-registry provenance display contracts."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from typer.testing import CliRunner

from forge.cli_commands.models import models_app
from forge.model_registry import ModelRegistry
from forge.provenance import MOCK_WARNING, build_provenance


def test_models_list_visibly_flags_mock_artifact(tmp_path: Path) -> None:
    checkpoint = tmp_path / "mock.pt"
    provenance = build_provenance(
        vision="mock",
        language="real",
        labels="mock",
        model_dir=tmp_path / "models",
        git_sha="abc123",
        forge_version="3.0.0-test",
        torch_version=str(torch.__version__),
    )
    torch.save(
        {"student_state_dict": {"weight": torch.ones(1)}, "provenance": provenance},
        checkpoint,
    )
    registry_dir = tmp_path / "registry"
    ModelRegistry(registry_dir).register(checkpoint, "nano", name="mock-student")

    result = CliRunner().invoke(
        models_app,
        ["list", "--registry-dir", str(registry_dir)],
    )

    assert result.exit_code == 0, result.output
    assert "mock-student" in result.output
    assert MOCK_WARNING in result.output

    json_result = CliRunner().invoke(
        models_app,
        ["list", "--registry-dir", str(registry_dir), "--json"],
    )
    assert json_result.exit_code == 0
    payload = json.loads(json_result.stdout)
    assert payload[0]["is_mock"] is True
    assert payload[0]["provenance_warning"] == MOCK_WARNING
    assert payload[0]["provenance"] == provenance
