"""Strict JSON serialization for every automation-facing CLI response."""

from __future__ import annotations

from typing import Any, NoReturn

import typer

from forge.json_utils import (
    SANITIZED_NOTE as SANITIZED_NOTE,
)
from forge.json_utils import (
    json_payload as json_payload,
)
from forge.json_utils import (
    json_ready as json_ready,
)
from forge.json_utils import (
    sanitize_json as sanitize_json,
)


def emit_json(value: Any, *, err: bool = False) -> None:
    """Emit strict JSON to exactly one CLI stream."""
    typer.echo(json_payload(value), err=err)


def emit_cli_error(message: str, *, output_json: bool, exit_code: int = 2) -> NoReturn:
    """Emit a stable error contract and terminate with the requested code."""
    if output_json:
        emit_json({"error": message}, err=True)
    else:
        typer.echo(message, err=True)
    raise typer.Exit(exit_code)


__all__ = [
    "SANITIZED_NOTE",
    "emit_cli_error",
    "emit_json",
    "json_payload",
    "json_ready",
    "sanitize_json",
]
