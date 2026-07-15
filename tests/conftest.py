"""Cross-module test isolation for CLI state."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def isolate_doctor_cli_logging(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep legacy CPU tests explicit about their synthetic model dependency."""
    monkeypatch.setenv("FORGE_ALLOW_MOCK", "1")
    if request.node.path.name in {"test_doctor.py", "test_doctor_hardening.py"}:
        monkeypatch.setattr("forge.cli_v2_root.setup_cli_logging", lambda **kwargs: None)
