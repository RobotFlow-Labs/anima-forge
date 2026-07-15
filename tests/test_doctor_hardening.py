"""Failure-path and integration coverage for ``forge doctor`` hardening."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

import forge.cli_commands._doctor_core as doctor_core
import forge.cli_commands.doctor as doctor_module
from forge.cli_commands._doctor_core import (
    MIN_MODEL_WEIGHT_BYTES,
    DoctorCheck,
    _latest_cache_snapshot,
    repair_model_links,
)
from forge.cli_v2 import app
from forge.model_assets import ModelAsset

TEST_ASSET = ModelAsset("example/hardened-model", "student:test")


def _sparse_file(path: Path, size: int = MIN_MODEL_WEIGHT_BYTES + 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.truncate(size)


def _model_tree(path: Path, *, valid: bool = True) -> Path:
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}" if valid else "{broken", encoding="utf-8")
    _sparse_file(path / "model.safetensors")
    return path


def _cache_snapshot(cache: Path, name: str, *, valid: bool = True) -> Path:
    return _model_tree(
        cache / f"models--{TEST_ASSET.local_name}" / "snapshots" / name,
        valid=valid,
    )


def test_cache_selects_newest_valid_snapshot(tmp_path: Path) -> None:
    cache = tmp_path / "hub"
    valid = _cache_snapshot(cache, "older-valid")
    invalid = _cache_snapshot(cache, "newer-invalid", valid=False)
    os.utime(valid, (1, 1))
    os.utime(invalid, (2, 2))

    assert _latest_cache_snapshot(TEST_ASSET, cache) == valid


def test_cache_rejects_all_invalid_snapshots(tmp_path: Path) -> None:
    cache = tmp_path / "hub"
    _cache_snapshot(cache, "invalid", valid=False)

    assert _latest_cache_snapshot(TEST_ASSET, cache) is None


def test_repair_preserves_existing_real_directory(tmp_path: Path) -> None:
    model_dir = tmp_path / "models"
    destination = _model_tree(model_dir / TEST_ASSET.local_name)
    cache = tmp_path / "hub"
    _cache_snapshot(cache, "cached")

    repairs = repair_model_links(model_dir, cache, assets=(TEST_ASSET,))

    assert repairs == []
    assert destination.is_dir()
    assert not destination.is_symlink()


def test_repair_is_idempotent(tmp_path: Path) -> None:
    model_dir = tmp_path / "models"
    cache = tmp_path / "hub"
    snapshot = _cache_snapshot(cache, "cached")

    first = repair_model_links(model_dir, cache, assets=(TEST_ASSET,))
    second = repair_model_links(model_dir, cache, assets=(TEST_ASSET,))

    assert len(first) == 1
    assert second == []
    assert (model_dir / TEST_ASSET.local_name).resolve() == snapshot.resolve()


def test_repair_filesystem_error_does_not_escape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    destination = model_dir / TEST_ASSET.local_name
    destination.symlink_to(tmp_path / "missing", target_is_directory=True)
    cache = tmp_path / "hub"
    _cache_snapshot(cache, "cached")
    original_symlink_to = Path.symlink_to

    def deny_temporary_link(
        path: Path,
        target: str | os.PathLike[str],
        target_is_directory: bool = False,
    ) -> None:
        if path.name.endswith(".forge-repair"):
            raise PermissionError("read-only model directory")
        original_symlink_to(path, target, target_is_directory=target_is_directory)

    monkeypatch.setattr(Path, "symlink_to", deny_temporary_link)

    assert repair_model_links(model_dir, cache, assets=(TEST_ASSET,)) == []
    assert destination.is_symlink()


def test_workspace_dataset_and_cache_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "forge" / "repo"
    root.mkdir(parents=True)
    workspace = root.parent.parent
    datasets = workspace / "datasets"
    datasets.mkdir()
    cache = workspace / ".hf-cache" / "hub"
    cache.mkdir(parents=True)
    (root / "data").mkdir()
    monkeypatch.delenv("FORGE_DATASET_DIR", raising=False)
    monkeypatch.delenv("FORGE_DATA_DIR", raising=False)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.setattr(doctor_core, "_project_root", lambda: root)

    assert doctor_core._default_dataset_dir(root) == datasets
    assert doctor_core._default_hf_cache_dir() == cache


def _stub_non_model_checks(
    monkeypatch: pytest.MonkeyPatch,
    *,
    gpu_status: str = "ok",
) -> None:
    monkeypatch.setattr(doctor_module, "_python_checks", lambda root: [DoctorCheck("python", "ok", "ok")])
    monkeypatch.setattr(doctor_module, "_dataset_check", lambda path: DoctorCheck("datasets", "ok", "ok"))
    monkeypatch.setattr(doctor_module, "_gpu_check", lambda: DoctorCheck("gpu", gpu_status, gpu_status))
    monkeypatch.setattr(
        doctor_module,
        "_disk_check",
        lambda name, path: DoctorCheck(name, "ok", "ok"),
    )
    monkeypatch.setattr(
        doctor_module,
        "_docker_checks",
        lambda: [DoctorCheck("docker", "ok", "ok")],
    )
    monkeypatch.setattr(doctor_module, "_token_check", lambda root: DoctorCheck("hf_token", "ok", "ok"))


@pytest.mark.parametrize(
    ("model_present", "gpu_status", "expected_status", "expected_exit"),
    [
        (True, "ok", "ok", 0),
        (True, "warning", "warning", 1),
        (False, "ok", "error", 2),
    ],
)
def test_run_doctor_aggregates_full_report_exit_codes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_present: bool,
    gpu_status: str,
    expected_status: str,
    expected_exit: int,
) -> None:
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    if model_present:
        _model_tree(model_dir / TEST_ASSET.local_name)
    _stub_non_model_checks(monkeypatch, gpu_status=gpu_status)

    report = doctor_module.run_doctor(
        project_root=tmp_path,
        model_dir=model_dir,
        dataset_dir=tmp_path / "datasets",
        output_dir=tmp_path / "outputs",
        hf_cache_dir=tmp_path / "hub",
        expected_assets=(TEST_ASSET,),
    )

    assert report["status"] == expected_status
    assert report["exit_code"] == expected_exit
    assert report["summary"]["error"] == (0 if model_present else 1)


@pytest.mark.parametrize(("status", "exit_code"), [("warning", 1), ("error", 2)])
def test_doctor_cli_preserves_json_for_nonzero_exits(
    monkeypatch: pytest.MonkeyPatch,
    status: str,
    exit_code: int,
) -> None:
    report = {
        "status": status,
        "exit_code": exit_code,
        "timestamp": "2026-07-12T00:00:00+00:00",
        "summary": {"ok": 0, "warning": int(status == "warning"), "error": int(status == "error")},
        "paths": {},
        "repairs": [],
        "checks": [DoctorCheck("probe", status, status).to_dict()],
    }
    monkeypatch.setattr(doctor_module, "run_doctor", lambda fix=False: report)

    result = CliRunner().invoke(app, ["doctor", "--json"])

    assert result.exit_code == exit_code
    assert json.loads(result.stdout) == report
    assert result.stderr == ""


def test_python_check_warns_for_non_project_venv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / ".venv").mkdir()
    source = tmp_path / "src" / "forge" / "__init__.py"
    source.parent.mkdir(parents=True)
    source.touch()
    monkeypatch.setattr(
        doctor_core.importlib.util,
        "find_spec",
        lambda _name: type("Spec", (), {"origin": str(source)})(),
    )
    monkeypatch.setattr(doctor_core.sys, "prefix", str(tmp_path / "shared-venv"))
    monkeypatch.setattr(doctor_core.sys, "base_prefix", str(tmp_path / "python"))
    monkeypatch.setattr(doctor_core.shutil, "which", lambda name: None)

    checks = doctor_core._python_checks(tmp_path)
    venv = next(check for check in checks if check.name == "venv")

    assert venv.status == "warning"
    assert "not this checkout's supported .venv" in venv.message


def test_python_check_accepts_isolated_installed_tool_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / ".venv").mkdir()
    installed = tmp_path / "tools" / "anima-forge" / "site-packages" / "forge" / "__init__.py"
    installed.parent.mkdir(parents=True)
    installed.touch()
    monkeypatch.setattr(doctor_core.sys, "prefix", str(tmp_path / "tools" / "anima-forge"))
    monkeypatch.setattr(doctor_core.sys, "base_prefix", str(tmp_path / "python"))
    monkeypatch.setattr(
        doctor_core.importlib.util,
        "find_spec",
        lambda _name: type("Spec", (), {"origin": str(installed)})(),
    )
    monkeypatch.setattr(doctor_core.shutil, "which", lambda name: None)

    checks = doctor_core._python_checks(tmp_path)
    venv = next(check for check in checks if check.name == "venv")

    assert venv.status == "ok"
    assert venv.message == "isolated installed environment active"
