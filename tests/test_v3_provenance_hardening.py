"""Security and fail-closed regressions for PRD-36 checkpoint handling."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml
from torch import nn
from typer.testing import CliRunner

from forge.checkpoint_compat import (
    load_checkpoint_payload,
    load_model_weights_with_compatibility,
)
from forge.config import ForgeConfig
from forge.model_registry import ModelRegistry
from forge.pipeline import run_pipeline
from forge.provenance import build_provenance


def _provenance(*, mock: bool, model_dir: Path) -> dict[str, str]:
    status = "mock" if mock else "real"
    return build_provenance(
        vision=status,
        language=status,
        labels=status,
        model_dir=model_dir,
        git_sha="abc123",
        forge_version="3.0.0-test",
        torch_version=str(torch.__version__),
    )


@pytest.mark.parametrize("operation", [None, "serve"])
def test_checkpoint_load_never_falls_back_to_unsafe_pickle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str | None,
) -> None:
    checkpoint = tmp_path / "unsafe.pt"
    checkpoint.write_bytes(b"not a safe tensor checkpoint")
    attempts: list[bool] = []

    def fail_load(*_args, **kwargs):
        attempts.append(bool(kwargs["weights_only"]))
        raise RuntimeError("safe load failed")

    monkeypatch.setattr(torch, "load", fail_load)

    with pytest.raises(ValueError, match="Refusing unsafe legacy checkpoint load"):
        load_checkpoint_payload(str(checkpoint), verify_provenance_for=operation)

    assert attempts == [True]


def test_checkpoint_loader_rejects_zero_matching_keys() -> None:
    model = nn.Linear(2, 2)

    with pytest.raises(RuntimeError, match="No compatible tensor keys"):
        load_model_weights_with_compatibility(
            model,
            {"completely_unrelated": torch.ones(1)},
            context="serve:test",
        )


def test_protected_checkpoint_loader_rejects_low_coverage() -> None:
    model = nn.Linear(2, 2)

    with pytest.raises(RuntimeError, match="mostly random weights"):
        load_model_weights_with_compatibility(
            model,
            {"bias": torch.ones(2)},
            context="eval:test",
            minimum_coverage=0.8,
        )


@pytest.mark.parametrize("stage", ["compress", "export", "validate"])
def test_strict_standalone_artifact_stages_require_checkpoint(
    stage: str,
    tmp_path: Path,
) -> None:
    config = ForgeConfig.default()
    config.student.allow_mock = False
    config.paths.output_dir = str(tmp_path / "outputs")

    result = run_pipeline(config, device="cpu", stage=stage)

    assert result["status"] == "failed"
    result_key = {"compress": "compression", "export": "export", "validate": "validation"}[stage]
    assert "requires a trained checkpoint" in result[result_key]["error"]
    assert not Path(config.paths.output_dir).exists()


def test_root_serve_refusal_is_clean_exit_two(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from forge.cli_v2 import app

    monkeypatch.delenv("FORGE_ALLOW_MOCK", raising=False)
    checkpoint = tmp_path / "mock.pt"
    torch.save(
        {
            "model_state_dict": {},
            "provenance": _provenance(mock=True, model_dir=tmp_path / "models"),
        },
        checkpoint,
    )

    result = CliRunner().invoke(
        app,
        ["serve", "--checkpoint", str(checkpoint), "--device", "cpu"],
    )

    assert result.exit_code == 2
    assert "Refusing to serve a mock-derived checkpoint" in result.output
    assert "Traceback" not in result.output


def test_allow_mock_env_has_precedence_over_yaml_false(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"student": {"allow_mock": False}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("FORGE_ALLOW_MOCK", "1")

    assert ForgeConfig.from_yaml(config_path).student.allow_mock is True


def test_quoted_false_yaml_is_not_truthy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text('student:\n  allow_mock: "false"\n', encoding="utf-8")
    monkeypatch.delenv("FORGE_ALLOW_MOCK", raising=False)

    assert ForgeConfig.from_yaml(config_path).student.allow_mock is False


def test_registry_refuses_provenance_override_conflict(tmp_path: Path) -> None:
    checkpoint = tmp_path / "mock.pt"
    torch.save(
        {
            "student_state_dict": {"weight": torch.ones(1)},
            "provenance": _provenance(mock=True, model_dir=tmp_path / "models"),
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="conflicts with the checkpoint provenance"):
        ModelRegistry(tmp_path / "registry").register(
            checkpoint,
            "nano",
            provenance=_provenance(mock=False, model_dir=tmp_path / "models"),
        )
