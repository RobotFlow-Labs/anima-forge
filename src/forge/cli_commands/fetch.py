"""Upstream model asset downloads for ``forge models fetch``."""

from __future__ import annotations

import hashlib
import os
import sys
import urllib.request
from collections.abc import Sequence
from contextlib import contextmanager, nullcontext, redirect_stdout
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from forge.cli_commands._doctor_core import (
    _default_hf_cache_dir,
    _latest_cache_snapshot,
    _project_root,
    _snapshot_is_valid,
    _validate_model,
)
from forge.cli_commands.models import models_app
from forge.cli_commands.shared import emit_cli_error, emit_json
from forge.model_assets import (
    ALL_MODEL_ASSETS,
    CORE_MODEL_ASSETS,
    OPTIONAL_MODEL_ASSETS,
    ModelAsset,
    find_model_asset,
)

console = Console()


class ModelFetchError(RuntimeError):
    """A model asset could not be safely installed."""


def _dotenv_value(path: Path, names: set[str]) -> str | None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, value = line.partition("=")
        if separator and key.strip() in names:
            resolved = value.strip().strip("\"'")
            return resolved or None
    return None


def _resolve_hf_token(project_root: Path) -> str | None:
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or _dotenv_value(project_root / ".env", {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"})
    )


@contextmanager
def _disable_xet() -> Any:
    overrides = {
        "HF_HUB_DISABLE_XET": "1",
        "HF_HUB_DOWNLOAD_TIMEOUT": "600",
        "HF_HUB_ETAG_TIMEOUT": "60",
    }
    previous = {name: os.environ.get(name) for name in overrides}
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    for name in ("HF_HUB_DOWNLOAD_TIMEOUT", "HF_HUB_ETAG_TIMEOUT"):
        os.environ.setdefault(name, overrides[name])
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _snapshot_download(repo_id: str, *, cache_dir: Path, token: str | None) -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=repo_id,
            cache_dir=str(cache_dir),
            token=token,
        )
    )


def _atomic_model_link(destination: Path, snapshot: Path) -> None:
    """Atomically install a model symlink without replacing real directories."""
    if destination.exists() and destination.is_dir() and not destination.is_symlink():
        raise ModelFetchError(f"refusing to replace existing model directory: {destination}")
    if os.path.lexists(destination) and not destination.is_symlink():
        raise ModelFetchError(f"refusing to replace existing path: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.forge-fetch")
    try:
        if os.path.lexists(temporary):
            if not temporary.is_symlink():
                raise ModelFetchError(f"temporary install path is occupied: {temporary}")
            temporary.unlink()
        temporary.symlink_to(snapshot.resolve(), target_is_directory=True)
        os.replace(temporary, destination)
    except OSError as exc:
        try:
            if temporary.is_symlink():
                temporary.unlink()
        except OSError:
            pass
        raise ModelFetchError(f"could not install model link {destination}: {exc}") from exc


def _ensure_sidecar(asset: ModelAsset, model_dir: Path) -> Path | None:
    """Fetch and checksum a small official non-Hugging-Face companion file."""
    if not asset.sidecar_url:
        return None
    if not asset.sidecar_filename or not asset.sidecar_sha256:
        raise ModelFetchError(f"incomplete sidecar manifest for {asset.repo_id}")
    destination = model_dir / asset.sidecar_filename
    if destination.is_file() and hashlib.sha256(destination.read_bytes()).hexdigest() == asset.sidecar_sha256:
        return destination

    model_dir.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.forge-fetch")
    try:
        with urllib.request.urlopen(asset.sidecar_url, timeout=60) as response, temporary.open("wb") as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
        digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
        if digest != asset.sidecar_sha256:
            raise ModelFetchError(
                f"sidecar checksum mismatch for {asset.repo_id}: expected {asset.sidecar_sha256}, got {digest}"
            )
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def _fetch_one(
    asset: ModelAsset,
    *,
    model_dir: Path,
    cache_dir: Path,
    token: str | None,
) -> dict[str, Any]:
    destination = model_dir / asset.local_name
    existing = _validate_model(asset, model_dir)
    if existing.status == "ok":
        sidecar = _ensure_sidecar(asset, model_dir)
        return {
            "repo_id": asset.repo_id,
            "status": "already_present",
            "path": str(destination),
            "weight_bytes": existing.details["weight_bytes"],
            **({"sidecar": str(sidecar)} if sidecar else {}),
        }

    if os.path.lexists(destination) and not destination.is_symlink():
        raise ModelFetchError(f"incomplete model directory already exists at {destination}; move it aside and retry")

    cached = _latest_cache_snapshot(asset, cache_dir)
    if cached is not None:
        _atomic_model_link(destination, cached)
        sidecar = _ensure_sidecar(asset, model_dir)
        return {
            "repo_id": asset.repo_id,
            "status": "linked_from_cache",
            "path": str(destination),
            "source": str(cached.resolve()),
            **({"sidecar": str(sidecar)} if sidecar else {}),
        }

    with _disable_xet():
        snapshot = _snapshot_download(asset.repo_id, cache_dir=cache_dir, token=token)
    if not _snapshot_is_valid(snapshot, asset):
        raise ModelFetchError(f"downloaded snapshot failed integrity checks: {asset.repo_id}")
    _atomic_model_link(destination, snapshot)
    sidecar = _ensure_sidecar(asset, model_dir)
    return {
        "repo_id": asset.repo_id,
        "status": "downloaded",
        "path": str(destination),
        "source": str(snapshot.resolve()),
        **({"sidecar": str(sidecar)} if sidecar else {}),
    }


def fetch_assets(
    assets: Sequence[ModelAsset],
    *,
    model_dir: str | Path,
    cache_dir: str | Path,
    token: str | None,
) -> dict[str, Any]:
    """Fetch/link each asset and return a complete non-throwing report."""
    resolved_model_dir = Path(model_dir).expanduser()
    resolved_cache_dir = Path(cache_dir).expanduser()
    results: list[dict[str, Any]] = []
    failures = 0
    for asset in assets:
        try:
            result = _fetch_one(
                asset,
                model_dir=resolved_model_dir,
                cache_dir=resolved_cache_dir,
                token=token,
            )
        except Exception as exc:
            failures += 1
            message = str(exc)
            if token:
                message = message.replace(token, "***")
            result = {
                "repo_id": asset.repo_id,
                "status": "error",
                "path": str(resolved_model_dir / asset.local_name),
                "error": message,
            }
        results.append(result)

    return {
        "status": "error" if failures else "ok",
        "exit_code": 2 if failures else 0,
        "model_dir": str(resolved_model_dir),
        "cache_dir": str(resolved_cache_dir),
        "authenticated": token is not None,
        "summary": {"requested": len(assets), "succeeded": len(assets) - failures, "failed": failures},
        "results": results,
    }


def _unique_assets(assets: Sequence[ModelAsset]) -> tuple[ModelAsset, ...]:
    return tuple({asset.repo_id: asset for asset in assets}.values())


def select_fetch_assets(
    *,
    name: str | None,
    all_students: bool,
    teachers: bool,
    all_assets: bool,
) -> tuple[ModelAsset, ...]:
    """Resolve the exactly-one selector accepted by the CLI."""
    selector_count = int(name is not None) + int(all_students) + int(teachers) + int(all_assets)
    if selector_count != 1:
        raise ValueError("choose exactly one: MODEL, --all-students, --teachers, or --all")

    if name is not None:
        known = find_model_asset(name)
        if known is not None:
            return (known,)
        normalized = name.strip().strip("/")
        if normalized.count("/") != 1 or any(part in {"", ".", ".."} for part in normalized.split("/")):
            raise ValueError("MODEL must be a Hugging Face repo id in org/name form")
        return (ModelAsset(normalized, "custom", required=False),)

    if teachers:
        return tuple(asset for asset in CORE_MODEL_ASSETS if asset.role == "teacher")
    if all_students:
        active = tuple(
            asset for asset in CORE_MODEL_ASSETS if asset.role.startswith("student:") or asset.role == "vision"
        )
        stretch = tuple(asset for asset in OPTIONAL_MODEL_ASSETS if asset.role == "student:stretch")
        return _unique_assets(active + stretch)
    return ALL_MODEL_ASSETS


def _print_fetch_report(report: dict[str, Any]) -> None:
    table = Table(title="FORGE Model Assets")
    table.add_column("Model", style="cyan")
    table.add_column("Status")
    table.add_column("Path / Error")
    colors = {
        "already_present": "green",
        "linked_from_cache": "green",
        "downloaded": "green",
        "error": "red",
    }
    for result in report["results"]:
        status = result["status"]
        color = colors[status]
        detail = result.get("error") or result.get("path", "")
        model = result.get("repo_id", "selection")
        table.add_row(model, f"[{color}]{status}[/{color}]", detail)
    console.print(table)
    summary = report["summary"]
    console.print(f"Fetched {summary['succeeded']}/{summary['requested']} model assets; {summary['failed']} failed.")


@models_app.command("fetch")
def models_fetch(
    name: str | None = typer.Argument(None, help="Hugging Face repo id (org/name)"),
    all_students: bool = typer.Option(False, "--all-students", help="Fetch V3 students and vision"),
    teachers: bool = typer.Option(False, "--teachers", help="Fetch current teacher fleet"),
    all_assets: bool = typer.Option(False, "--all", help="Fetch every known model asset"),
    model_dir: str | None = typer.Option(None, "--model-dir", help="Override FORGE model directory"),
    cache_dir: str | None = typer.Option(None, "--cache-dir", help="Override Hugging Face cache"),
    output_json: bool = typer.Option(False, "--json", help="Emit JSON only"),
) -> None:
    """Download upstream model weights and install canonical local links."""
    root = _project_root()
    resolved_model_dir = Path(model_dir or os.environ.get("FORGE_MODEL_DIR") or root / "models")
    resolved_cache_dir = Path(cache_dir).expanduser() if cache_dir else _default_hf_cache_dir()
    token = _resolve_hf_token(root)
    try:
        assets = select_fetch_assets(
            name=name,
            all_students=all_students,
            teachers=teachers,
            all_assets=all_assets,
        )
    except ValueError as exc:
        emit_cli_error(
            str(exc),
            output_json=output_json,
            exit_code=2,
        )
    else:
        output_context = redirect_stdout(sys.stderr) if output_json else nullcontext()
        with output_context:
            report = fetch_assets(
                assets,
                model_dir=resolved_model_dir,
                cache_dir=resolved_cache_dir,
                token=token,
            )

    if output_json:
        emit_json(report)
    else:
        _print_fetch_report(report)
    if report["exit_code"]:
        raise typer.Exit(report["exit_code"])


__all__ = [
    "ModelFetchError",
    "fetch_assets",
    "models_fetch",
    "select_fetch_assets",
]
