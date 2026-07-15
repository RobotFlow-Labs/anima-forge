"""PRD-20: Enhanced CLI tests."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from forge.cli import app
from forge.web.state import ServerState

runner = CliRunner()


@pytest.fixture(autouse=True)
def reset_state():
    ServerState.reset()
    yield
    ServerState.reset()


def test_cli_status_json_output():
    """forge status --json outputs valid JSON."""
    result = runner.invoke(app, ["status", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "gpu" in data
    assert "version" in data


def test_cli_teacher_list_json():
    """forge teacher list --json outputs valid JSON."""
    result = runner.invoke(app, ["teacher", "list", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert isinstance(data, list)


def test_cli_embodiment_list_json():
    """forge embodiment list --json outputs valid JSON."""
    result = runner.invoke(app, ["embodiment", "list", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert isinstance(data, list)
    if data:
        assert "name" in data[0]
        assert "dof" in data[0]


def test_cli_subcommand_groups_exist():
    """All v2 subcommand groups are registered."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    output = result.output.lower()
    assert "teacher" in output
    assert "benchmark" in output
    assert "embodiment" in output
    assert "demo" in output
    assert "web" in output
    assert "status" in output
