"""PRD-39 user-facing error hierarchy and console boundary."""

from __future__ import annotations

import pytest

from forge.errors import ForgeDataNotFoundError, ForgeError, ForgeModelNotFoundError


def test_domain_errors_expose_exact_recovery_hints() -> None:
    model = ForgeModelNotFoundError(
        component="language",
        model_id="Qwen/Qwen3-0.6B",
        path="models/Qwen--Qwen3-0.6B",
    )
    assert isinstance(model, ForgeError)
    assert "models/Qwen--Qwen3-0.6B" in model.message
    assert "forge models fetch Qwen/Qwen3-0.6B" in model.hint

    data = ForgeDataNotFoundError("Teacher labels missing at ./data/teacher_labels")
    assert "./data/teacher_labels" in data.message
    assert "forge pipeline --stage labels" in data.hint


def test_console_entrypoint_formats_forge_errors_without_traceback(monkeypatch, capsys) -> None:
    from forge import cli_v2

    def fail() -> None:
        raise ForgeError("Bad checkpoint at /tmp/missing.pt.", hint="Run `forge doctor`.")

    monkeypatch.setattr(cli_v2, "app", fail)
    with pytest.raises(SystemExit) as exited:
        cli_v2.main()

    assert exited.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "Error: Bad checkpoint at /tmp/missing.pt.\nHint: Run `forge doctor`.\n"
    assert "Traceback" not in captured.err
