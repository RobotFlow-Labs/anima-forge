"""``forge doctor`` command and complete environment health report."""

from __future__ import annotations

import os
import shutil
import subprocess
import warnings
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from forge.cli_commands._doctor_core import (
    DoctorCheck,
    _default_dataset_dir,
    _default_hf_cache_dir,
    _nearest_existing_path,
    _project_root,
    _python_checks,
    _validate_model,
    repair_model_links,
)
from forge.cli_commands.shared import emit_json
from forge.model_assets import ALL_MODEL_ASSETS, CORE_MODEL_ASSETS, ModelAsset

console = Console()

MIN_DISK_FREE_BYTES = 50 * 1024**3
EXPECTED_DATASETS = (
    "lerobot--pusht",
    "lerobot--aloha_sim_insertion_human",
    "lerobot--aloha_sim_transfer_cube_human",
)


def _dataset_check(dataset_dir: Path) -> DoctorCheck:
    details = {"path": str(dataset_dir), "expected": list(EXPECTED_DATASETS)}
    if not dataset_dir.is_dir():
        return DoctorCheck("datasets", "error", "dataset directory is missing", details)
    missing = [name for name in EXPECTED_DATASETS if not (dataset_dir / name).exists()]
    details["missing"] = missing
    if missing:
        return DoctorCheck("datasets", "error", "required datasets are missing", details)
    return DoctorCheck("datasets", "ok", "required datasets found", details)


def _capture_probe_warnings(probe: Any) -> Any:
    def wrapped() -> DoctorCheck:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            result = probe()
        if captured:
            warning_messages = list(dict.fromkeys(str(item.message)[:500] for item in captured))
            details = {**result.details, "warnings": warning_messages}
            if result.status == "ok":
                return DoctorCheck(
                    result.name,
                    "warning",
                    f"{result.message}; CUDA runtime warnings detected",
                    details,
                )
            return DoctorCheck(result.name, result.status, result.message, details)
        return result

    return wrapped


def _nvidia_management_probe() -> dict[str, Any]:
    """Verify the NVML-backed management path required by NCCL and containers."""
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {
            "healthy": False,
            "error": "nvidia-smi executable not found",
        }
    details: dict[str, Any] = {"healthy": False, "path": executable}
    try:
        result = subprocess.run(
            [
                executable,
                "--query-gpu=index,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        details["error"] = str(exc)[:500]
        return details

    details["return_code"] = result.returncode
    if result.returncode != 0:
        error = result.stderr.strip() or result.stdout.strip() or "nvidia-smi failed without diagnostic output"
        details["error"] = error[:500]
        return details

    devices: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        index, separator, driver_version = line.partition(",")
        if not separator or not index.strip() or not driver_version.strip():
            details["error"] = f"unexpected nvidia-smi output: {line[:300]}"
            return details
        devices.append({"index": index.strip(), "driver_version": driver_version.strip()})
    if not devices:
        details["error"] = "nvidia-smi reported no GPUs"
        return details
    details.update({"healthy": True, "devices": devices})
    return details


@_capture_probe_warnings
def _gpu_check() -> DoctorCheck:
    try:
        import torch
    except Exception as exc:
        return DoctorCheck("gpu", "warning", "PyTorch GPU probe unavailable", {"error": str(exc)})

    try:
        if not torch.cuda.is_available():
            return DoctorCheck("gpu", "warning", "CUDA is not available", {"device_count": 0})
        devices: list[dict[str, Any]] = []
        for index in range(torch.cuda.device_count()):
            with torch.cuda.device(index):
                free_bytes, total_bytes = torch.cuda.mem_get_info()
            devices.append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "free_vram_bytes": int(free_bytes),
                    "total_vram_bytes": int(total_bytes),
                }
            )
        if not devices:
            return DoctorCheck("gpu", "warning", "CUDA reported no visible devices", {"device_count": 0})
        management = _nvidia_management_probe()
        details: dict[str, Any] = {
            "device_count": len(devices),
            "devices": devices,
            "nvidia_management": management,
        }
        if not management["healthy"]:
            return DoctorCheck(
                "gpu",
                "warning",
                f"{len(devices)} CUDA device(s) visible, but NVIDIA management/NVML is unhealthy",
                details,
            )
        return DoctorCheck("gpu", "ok", f"{len(devices)} CUDA device(s) visible", details)
    except Exception as exc:
        return DoctorCheck("gpu", "warning", "CUDA probe failed", {"error": str(exc)})


def _disk_check(name: str, path: Path) -> DoctorCheck:
    mount_path = _nearest_existing_path(path)
    details: dict[str, Any] = {"path": str(path)}
    if mount_path is None:
        return DoctorCheck(name, "error", "no existing parent mount found", details)
    try:
        usage = shutil.disk_usage(mount_path)
    except OSError as exc:
        details["error"] = str(exc)
        return DoctorCheck(name, "error", "disk usage probe failed", details)
    details.update(
        {
            "mount_path": str(mount_path),
            "free_bytes": usage.free,
            "total_bytes": usage.total,
            "warning_below_bytes": MIN_DISK_FREE_BYTES,
        }
    )
    if usage.free < MIN_DISK_FREE_BYTES:
        return DoctorCheck(name, "warning", "less than 50 GiB disk space is free", details)
    return DoctorCheck(name, "ok", f"{usage.free / 1024**3:.1f} GiB free", details)


def _docker_checks() -> list[DoctorCheck]:
    executable = shutil.which("docker")
    if executable is None:
        return [DoctorCheck("docker", "warning", "Docker is not installed")]

    try:
        from forge.eval.runner import BENCHMARK_IMAGES
    except Exception as exc:
        return [
            DoctorCheck("docker", "ok", "Docker executable found", {"path": executable}),
            DoctorCheck("docker_images", "warning", "eval image manifest unavailable", {"error": str(exc)}),
        ]

    missing: list[str] = []
    errors: dict[str, str] = {}
    for image in BENCHMARK_IMAGES.values():
        try:
            result = subprocess.run(
                [executable, "image", "inspect", image],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            missing.append(image)
            errors[image] = str(exc)
            continue
        if result.returncode != 0:
            missing.append(image)
            if result.stderr.strip():
                errors[image] = result.stderr.strip()[:500]

    docker = DoctorCheck("docker", "ok", "Docker executable found", {"path": executable})
    details: dict[str, Any] = {"expected": sorted(BENCHMARK_IMAGES.values()), "missing": sorted(missing)}
    if errors:
        details["errors"] = errors
    images = DoctorCheck(
        "docker_images",
        "warning" if missing else "ok",
        "one or more eval images are missing" if missing else "all eval images found",
        details,
    )
    return [docker, images]


def _env_file_has_token(path: Path) -> bool:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return False
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, value = line.partition("=")
        if separator and key.strip() in {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"}:
            return bool(value.strip().strip("\"'"))
    return False


def _token_check(project_root: Path) -> DoctorCheck:
    source = None
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        source = "environment"
    elif _env_file_has_token(project_root / ".env"):
        source = str(project_root / ".env")
    if source:
        return DoctorCheck("hf_token", "ok", "Hugging Face token configured", {"source": source})
    return DoctorCheck(
        "hf_token",
        "warning",
        "HF_TOKEN is not configured; gated/private downloads may fail",
    )


def _summarize(checks: Iterable[DoctorCheck]) -> tuple[str, int, dict[str, int]]:
    counts = {"ok": 0, "warning": 0, "error": 0}
    for check in checks:
        counts[check.status] += 1
    if counts["error"]:
        return "error", 2, counts
    if counts["warning"]:
        return "warning", 1, counts
    return "ok", 0, counts


def run_doctor(
    *,
    fix: bool = False,
    model_dir: str | Path | None = None,
    dataset_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    hf_cache_dir: str | Path | None = None,
    project_root: str | Path | None = None,
    expected_assets: Sequence[ModelAsset] = CORE_MODEL_ASSETS,
) -> dict[str, Any]:
    """Collect a complete doctor report without printing or exiting."""
    root = Path(project_root).expanduser().resolve() if project_root else _project_root()
    resolved_model_dir = Path(model_dir or os.environ.get("FORGE_MODEL_DIR") or root / "models").expanduser()
    resolved_dataset_dir = Path(dataset_dir).expanduser() if dataset_dir else _default_dataset_dir(root)
    resolved_output_dir = Path(output_dir or os.environ.get("FORGE_OUTPUT_DIR") or root / "outputs").expanduser()
    resolved_hf_cache = Path(hf_cache_dir).expanduser() if hf_cache_dir else _default_hf_cache_dir()

    repairs: list[dict[str, str]] = []
    if fix:
        repair_assets = tuple(expected_assets) + tuple(
            asset for asset in ALL_MODEL_ASSETS if asset not in expected_assets
        )
        repairs = repair_model_links(resolved_model_dir, resolved_hf_cache, repair_assets)

    checks = _python_checks(root)
    checks.append(
        DoctorCheck(
            "model_dir",
            "ok" if resolved_model_dir.is_dir() else "error",
            "model directory found" if resolved_model_dir.is_dir() else "model directory is missing",
            {"path": str(resolved_model_dir)},
        )
    )
    checks.extend(_validate_model(asset, resolved_model_dir) for asset in expected_assets)
    checks.append(_dataset_check(resolved_dataset_dir))
    checks.append(_gpu_check())
    checks.append(_disk_check("disk:model", resolved_model_dir))
    checks.append(_disk_check("disk:output", resolved_output_dir))
    checks.extend(_docker_checks())
    checks.append(_token_check(root))

    status, exit_code, summary = _summarize(checks)
    return {
        "status": status,
        "exit_code": exit_code,
        "timestamp": datetime.now(UTC).isoformat(),
        "summary": summary,
        "paths": {
            "project_root": str(root),
            "model_dir": str(resolved_model_dir),
            "dataset_dir": str(resolved_dataset_dir),
            "output_dir": str(resolved_output_dir),
            "hf_cache_dir": str(resolved_hf_cache),
        },
        "repairs": repairs,
        "checks": [check.to_dict() for check in checks],
    }


def _print_human_report(report: dict[str, Any]) -> None:
    colors = {"ok": "green", "warning": "yellow", "error": "red"}
    table = Table(title="FORGE Doctor")
    table.add_column("Check", style="cyan", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Result")
    for check in report["checks"]:
        color = colors[check["status"]]
        table.add_row(check["name"], f"[{color}]{check['status'].upper()}[/{color}]", check["message"])
    console.print(table)
    if report["repairs"]:
        console.print(f"[green]Repaired {len(report['repairs'])} model link(s).[/green]")
    summary = report["summary"]
    color = colors[report["status"]]
    console.print(
        f"[{color}]Status: {report['status']} "
        f"({summary['ok']} ok, {summary['warning']} warning, {summary['error']} error)[/{color}]"
    )


def doctor_command(
    output_json: bool = typer.Option(False, "--json", help="Emit JSON only"),
    fix: bool = typer.Option(False, "--fix", help="Repair model links from the local HF cache"),
) -> None:
    """Check the FORGE environment and optionally repair local model links."""
    report = run_doctor(fix=fix)
    if output_json:
        emit_json(report)
    else:
        _print_human_report(report)
    if report["exit_code"]:
        raise typer.Exit(report["exit_code"])


__all__ = [
    "DoctorCheck",
    "doctor_command",
    "repair_model_links",
    "run_doctor",
]
