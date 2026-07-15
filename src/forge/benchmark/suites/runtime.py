"""Shared paths for packaged benchmark suites."""

from __future__ import annotations

import os
from pathlib import Path


def results_dir() -> Path:
    """Return the caller-selected benchmark artifact directory."""
    path = Path(os.environ.get("FORGE_BENCHMARK_RESULTS_DIR", "benchmarks"))
    path = path.expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def export_dir() -> Path:
    """Return the caller-selected benchmark export scratch directory."""
    path = Path(os.environ.get("FORGE_BENCHMARK_EXPORT_DIR", "outputs/export"))
    path = path.expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path
