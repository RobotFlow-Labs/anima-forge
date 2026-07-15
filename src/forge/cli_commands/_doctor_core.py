"""Environment diagnostics and local model-cache repair for FORGE v3."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console

from forge.model_assets import ALL_MODEL_ASSETS, ModelAsset

console = Console()

MIN_MODEL_WEIGHT_BYTES = 50 * 1024 * 1024
MIN_DISK_FREE_BYTES = 50 * 1024**3
WEIGHT_SUFFIXES = {".bin", ".ckpt", ".pt", ".pth", ".safetensors"}
EXPECTED_DATASETS = (
    "lerobot--pusht",
    "lerobot--aloha_sim_insertion_human",
    "lerobot--aloha_sim_transfer_cube_human",
)


@dataclass(frozen=True)
class DoctorCheck:
    """One stable, JSON-serializable doctor result."""

    name: str
    status: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _project_root() -> Path:
    """Find the checkout root when running from source, otherwise use cwd."""
    cwd = Path.cwd().resolve()
    if (cwd / "pyproject.toml").is_file():
        return cwd

    source_root = Path(__file__).resolve().parents[3]
    if (source_root / "pyproject.toml").is_file():
        return source_root
    return cwd


def _default_dataset_dir(project_root: Path) -> Path:
    """Resolve the dataset root without baking a host-specific absolute path."""
    configured = os.environ.get("FORGE_DATASET_DIR") or os.environ.get("FORGE_DATA_DIR")
    if configured:
        return Path(configured).expanduser()

    checkout_data = project_root / "data"
    workspace_data = project_root.parent.parent / "datasets"
    if workspace_data.exists():
        return workspace_data
    if checkout_data.exists():
        return checkout_data
    return checkout_data


def _default_hf_cache_dir() -> Path:
    explicit = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if explicit:
        return Path(explicit).expanduser()
    if hf_home := os.environ.get("HF_HOME"):
        return Path(hf_home).expanduser() / "hub"
    workspace_cache = _project_root().parent.parent / ".hf-cache" / "hub"
    if workspace_cache.is_dir():
        return workspace_cache
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        return Path(HF_HUB_CACHE).expanduser()
    except Exception:
        return Path.home() / ".cache" / "huggingface" / "hub"


def _nearest_existing_path(path: Path) -> Path | None:
    current = path.expanduser().absolute()
    while not current.exists() and current != current.parent:
        current = current.parent
    return current if current.exists() else None


def _weight_bytes(model_path: Path) -> int:
    total = 0
    for candidate in model_path.rglob("*"):
        try:
            if candidate.is_file() and candidate.suffix.lower() in WEIGHT_SUFFIXES:
                total += candidate.stat().st_size
        except OSError:
            continue
    return total


def _validate_model(asset: ModelAsset, model_dir: Path) -> DoctorCheck:
    path = model_dir / asset.local_name
    details: dict[str, Any] = {
        "repo_id": asset.repo_id,
        "role": asset.role,
        "path": str(path),
    }
    if path.is_symlink() and not path.exists():
        details["symlink_target"] = os.readlink(path)
        return DoctorCheck(
            f"model:{asset.repo_id}",
            "error",
            "broken model symlink",
            details,
        )
    if not path.is_dir():
        return DoctorCheck(
            f"model:{asset.repo_id}",
            "error",
            "required model is missing",
            details,
        )

    config_path = path / asset.config_filename
    if not config_path.is_file():
        return DoctorCheck(
            f"model:{asset.repo_id}",
            "error",
            f"{asset.config_filename} is missing",
            details,
        )
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("config root is not an object")
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        details["config_error"] = str(exc)
        return DoctorCheck(
            f"model:{asset.repo_id}",
            "error",
            f"{asset.config_filename} is unreadable",
            details,
        )

    missing_files = [filename for filename in asset.required_files if not (path / filename).is_file()]
    if missing_files:
        details["missing_files"] = missing_files
        return DoctorCheck(f"model:{asset.repo_id}", "error", "required asset files are missing", details)

    weight_bytes = _weight_bytes(path)
    details["weight_bytes"] = weight_bytes
    details["minimum_weight_bytes"] = MIN_MODEL_WEIGHT_BYTES
    if asset.weights_required and weight_bytes <= MIN_MODEL_WEIGHT_BYTES:
        return DoctorCheck(
            f"model:{asset.repo_id}",
            "error",
            "real model weights are missing or too small",
            details,
        )

    return DoctorCheck(
        f"model:{asset.repo_id}",
        "ok",
        f"real weights found ({weight_bytes / 1024**3:.2f} GiB)" if asset.weights_required else "asset files verified",
        details,
    )


def _snapshot_is_valid(path: Path, asset: ModelAsset) -> bool:
    try:
        config = json.loads((path / asset.config_filename).read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            return False
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
        return False
    if any(not (path / filename).is_file() for filename in asset.required_files):
        return False
    return not asset.weights_required or _weight_bytes(path) > MIN_MODEL_WEIGHT_BYTES


def _latest_cache_snapshot(asset: ModelAsset, hf_cache_dir: Path) -> Path | None:
    snapshots_dir = hf_cache_dir / f"models--{asset.local_name}" / "snapshots"
    if not snapshots_dir.is_dir():
        return None
    try:
        snapshots = [path for path in snapshots_dir.iterdir() if path.is_dir() and _snapshot_is_valid(path, asset)]
        return max(snapshots, key=lambda path: path.stat().st_mtime) if snapshots else None
    except OSError:
        return None


def repair_model_links(
    model_dir: Path,
    hf_cache_dir: Path,
    assets: Sequence[ModelAsset] = ALL_MODEL_ASSETS,
) -> list[dict[str, str]]:
    """Repair missing/broken model links from already-downloaded HF snapshots.

    This function is deliberately local-only: it never contacts the Hub and it
    never replaces a real directory.
    """
    repairs: list[dict[str, str]] = []
    try:
        model_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return repairs
    for asset in assets:
        destination = model_dir / asset.local_name
        if destination.exists() and destination.is_dir():
            continue
        if os.path.lexists(destination) and not destination.is_symlink():
            continue

        snapshot = _latest_cache_snapshot(asset, hf_cache_dir)
        if snapshot is None:
            continue
        temporary = destination.with_name(f".{destination.name}.forge-repair")
        try:
            snapshot_target = snapshot.resolve()
            if os.path.lexists(temporary):
                if not temporary.is_symlink():
                    continue
                temporary.unlink()
            temporary.symlink_to(snapshot_target, target_is_directory=True)
            os.replace(temporary, destination)
        except OSError:
            try:
                if temporary.is_symlink():
                    temporary.unlink()
            except OSError:
                pass
            continue
        repairs.append(
            {
                "repo_id": asset.repo_id,
                "path": str(destination),
                "target": str(snapshot_target),
            }
        )
    return repairs


def _python_checks(project_root: Path) -> list[DoctorCheck]:
    checks: list[DoctorCheck] = []
    executable = Path(sys.executable)
    checks.append(
        DoctorCheck(
            "python",
            "ok" if executable.is_file() else "error",
            f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            {"executable": str(executable), "prefix": sys.prefix},
        )
    )

    forge_spec = importlib.util.find_spec("forge")
    checks.append(
        DoctorCheck(
            "forge_import",
            "ok" if forge_spec is not None else "error",
            "forge is importable" if forge_spec is not None else "forge cannot be imported",
            {"origin": str(forge_spec.origin) if forge_spec and forge_spec.origin else None},
        )
    )

    in_venv = sys.prefix != sys.base_prefix
    project_venv = (project_root / ".venv").resolve()
    active_prefix = Path(sys.prefix).resolve()
    forge_origin = Path(forge_spec.origin).resolve() if forge_spec and forge_spec.origin else None
    source_root = (project_root / "src").resolve()
    running_from_checkout = forge_origin is not None and forge_origin.is_relative_to(source_root)
    if not in_venv:
        venv_status = "warning"
        venv_message = "no virtual environment detected"
    elif running_from_checkout and project_venv.is_dir() and active_prefix != project_venv:
        venv_status = "warning"
        venv_message = "active environment is not this checkout's supported .venv"
    elif not running_from_checkout:
        venv_status = "ok"
        venv_message = "isolated installed environment active"
    else:
        venv_status = "ok"
        venv_message = "project virtual environment active"
    checks.append(
        DoctorCheck(
            "venv",
            venv_status,
            venv_message,
            {
                "prefix": sys.prefix,
                "base_prefix": sys.base_prefix,
                "expected_prefix": str(project_venv),
                "source_checkout": running_from_checkout,
            },
        )
    )

    suffix = "Scripts/forge.exe" if os.name == "nt" else "bin/forge"
    candidates = [Path(sys.prefix) / suffix]
    if found := shutil.which("forge"):
        candidates.append(Path(found))
    entrypoint = next((path for path in candidates if path.is_file()), candidates[0])
    if not entrypoint.is_file():
        checks.append(
            DoctorCheck(
                "forge_entrypoint",
                "warning",
                "forge launcher was not found in the active environment",
                {"expected": str(entrypoint)},
            )
        )
        return checks

    details: dict[str, Any] = {"path": str(entrypoint)}
    if os.name != "nt":
        try:
            with entrypoint.open("rb") as stream:
                first_line = stream.readline(4096).decode("utf-8").strip()
            if first_line.startswith("#!"):
                target = Path(first_line[2:].split()[0])
                details["shebang_target"] = str(target)
                if not target.exists():
                    checks.append(
                        DoctorCheck(
                            "forge_entrypoint",
                            "error",
                            "forge launcher has a stale shebang",
                            details,
                        )
                    )
                    return checks
        except (OSError, UnicodeError) as exc:
            details["read_error"] = str(exc)
            checks.append(
                DoctorCheck(
                    "forge_entrypoint",
                    "error",
                    "forge launcher could not be inspected",
                    details,
                )
            )
            return checks
    checks.append(DoctorCheck("forge_entrypoint", "ok", "forge launcher is healthy", details))
    return checks
