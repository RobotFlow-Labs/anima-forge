"""Launch-time provenance shared by public benchmark commands."""

from __future__ import annotations

import platform
from importlib.metadata import PackageNotFoundError, version

from forge.provenance import current_git_sha


def _package_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "unknown"


def benchmark_execution(
    *,
    command: str,
    requested_device: str,
    resolved_device: str | None = None,
    suite: str | None = None,
    suite_number: str | None = None,
) -> dict[str, str]:
    """Capture the code and runtime identity before benchmark work starts."""
    execution = {
        "schema": "forge.benchmark-execution.v1",
        "command": command,
        "requested_device": requested_device,
        "git_sha": current_git_sha(),
        "forge_version": _package_version("anima-forge"),
        "torch_version": _package_version("torch"),
        "python_version": platform.python_version(),
    }
    if resolved_device is not None:
        execution["resolved_device"] = resolved_device
    if suite is not None:
        execution["suite"] = suite
    if suite_number is not None:
        execution["suite_number"] = suite_number
    return execution
