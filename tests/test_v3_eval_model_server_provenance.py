"""Checkpoint and provenance contracts for the maintained eval model server."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from forge.eval.model_server import ForgeModelServer
from forge.provenance import MockArtifactError, build_provenance


class _TinyStudent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)
        self.total_params = sum(parameter.numel() for parameter in self.parameters())
        self.trainable_params = self.total_params


def _provenance(tmp_path: Path, *, mock: bool) -> dict[str, str]:
    return build_provenance(
        vision="mock" if mock else "real",
        language="real",
        labels="real",
        model_dir=tmp_path / "models",
        git_sha="a" * 40,
        forge_version="3.0.0-test",
        torch_version=str(torch.__version__),
    )


def _checkpoint(
    tmp_path: Path,
    *,
    mock: bool,
    sparse: bool = False,
    student_config: dict[str, object] | None = None,
) -> Path:
    state_dict = _TinyStudent().state_dict()
    if sparse:
        state_dict = {"linear.weight": state_dict["linear.weight"]}
    path = tmp_path / "student.pt"
    torch.save(
        {
            "model_state_dict": state_dict,
            "provenance": _provenance(tmp_path, mock=mock),
            **({"student_config": student_config} if student_config is not None else {}),
        },
        path,
    )
    return path


def _replace_student(monkeypatch: pytest.MonkeyPatch, captured: list[bool]) -> None:
    import forge.student

    def build_student(config, *, model_dir=None):
        captured.append(config.allow_mock)
        return _TinyStudent()

    monkeypatch.setattr(forge.student, "FORGEStudent", build_student)


def test_missing_checkpoint_fails_before_student_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[bool] = []
    _replace_student(monkeypatch, captured)
    server = ForgeModelServer(str(tmp_path / "missing.pt"), device="cpu")

    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        server._ensure_model_loaded()

    assert captured == []


def test_unsupported_checkpoint_fails_before_student_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "unsupported.pt"
    torch.save(["not", "a", "mapping"], checkpoint)
    captured: list[bool] = []
    _replace_student(monkeypatch, captured)
    server = ForgeModelServer(str(checkpoint), device="cpu")

    with pytest.raises(ValueError, match="Unsupported or unreadable checkpoint payload"):
        server._ensure_model_loaded()

    assert captured == []


def test_unreadable_checkpoint_fails_before_student_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "corrupt.pt"
    checkpoint.write_bytes(b"not-a-torch-checkpoint")
    captured: list[bool] = []
    _replace_student(monkeypatch, captured)
    server = ForgeModelServer(str(checkpoint), device="cpu")

    with pytest.raises(ValueError, match="Refusing unsafe legacy checkpoint load for eval"):
        server._ensure_model_loaded()

    assert captured == []


def test_mock_checkpoint_is_refused_before_student_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FORGE_ALLOW_MOCK", raising=False)
    checkpoint = _checkpoint(tmp_path, mock=True)
    captured: list[bool] = []
    _replace_student(monkeypatch, captured)
    server = ForgeModelServer(str(checkpoint), device="cpu")

    with pytest.raises(MockArtifactError, match="Refusing to eval"):
        server._ensure_model_loaded()

    assert captured == []


@pytest.mark.parametrize("opt_in", ["api", "environment"])
def test_explicit_mock_opt_in_loads_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    opt_in: str,
) -> None:
    monkeypatch.delenv("FORGE_ALLOW_MOCK", raising=False)
    if opt_in == "environment":
        monkeypatch.setenv("FORGE_ALLOW_MOCK", "1")
    checkpoint = _checkpoint(tmp_path, mock=True)
    captured: list[bool] = []
    _replace_student(monkeypatch, captured)
    server = ForgeModelServer(
        str(checkpoint),
        device="cpu",
        allow_mock=opt_in == "api",
    )

    server._ensure_model_loaded()

    assert captured == [True]
    assert server._model is not None


def test_sparse_checkpoint_is_refused_as_mostly_random(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = _checkpoint(tmp_path, mock=False, sparse=True)
    captured: list[bool] = []
    _replace_student(monkeypatch, captured)
    server = ForgeModelServer(str(checkpoint), device="cpu")

    with pytest.raises(RuntimeError, match="at least 80.0%.*mostly random weights"):
        server._ensure_model_loaded()

    assert len(captured) == 1


def test_checkpoint_binds_protocol_action_shape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoint = _checkpoint(
        tmp_path,
        mock=False,
        student_config={"action_dim": 2, "action_horizon": 4},
    )
    captured: list[bool] = []
    _replace_student(monkeypatch, captured)
    server = ForgeModelServer(str(checkpoint), device="cpu")

    server._ensure_model_loaded()

    assert server.config.action_dim == 2
    assert server.config.chunk_size == 4
