"""Focused tests for the PRD-33 ``forge doctor`` contract."""

from __future__ import annotations

import json
import warnings
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
from typer.testing import CliRunner

import forge.cli_commands.doctor as doctor_module
from forge.cli_commands._doctor_core import (
    MIN_MODEL_WEIGHT_BYTES,
    DoctorCheck,
    _validate_model,
    repair_model_links,
)
from forge.cli_v2 import app
from forge.model_assets import ModelAsset

TEST_ASSET = ModelAsset("example/model", "student:test")


def _sparse_file(path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.truncate(size)


def _create_model(
    model_dir: Path,
    asset: ModelAsset = TEST_ASSET,
    *,
    config_text: str = "{}",
    weight_bytes: int = MIN_MODEL_WEIGHT_BYTES + 1,
) -> Path:
    model_path = model_dir / asset.local_name
    model_path.mkdir(parents=True)
    (model_path / "config.json").write_text(config_text, encoding="utf-8")
    _sparse_file(model_path / "model.safetensors", weight_bytes)
    return model_path


def _successful_report() -> dict:
    return {
        "status": "ok",
        "exit_code": 0,
        "timestamp": "2026-07-12T00:00:00+00:00",
        "summary": {"ok": 1, "warning": 0, "error": 0},
        "paths": {},
        "repairs": [],
        "checks": [DoctorCheck("python", "ok", "healthy").to_dict()],
    }


def test_doctor_cli_json_is_clean_and_forwards_fix(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[bool] = []

    def fake_run_doctor(*, fix: bool = False) -> dict:
        seen.append(fix)
        return _successful_report()

    monkeypatch.setattr(doctor_module, "run_doctor", fake_run_doctor)

    result = CliRunner().invoke(app, ["doctor", "--json", "--fix"])

    assert result.exit_code == 0
    assert json.loads(result.stdout) == _successful_report()
    assert seen == [True]


@pytest.mark.parametrize(
    ("statuses", "expected_status", "expected_exit", "expected_counts"),
    [
        (["ok", "ok"], "ok", 0, {"ok": 2, "warning": 0, "error": 0}),
        (["ok", "warning"], "warning", 1, {"ok": 1, "warning": 1, "error": 0}),
        (["warning", "error"], "error", 2, {"ok": 0, "warning": 1, "error": 1}),
    ],
)
def test_summarize_maps_severity_to_exit_code(
    statuses: list[str],
    expected_status: str,
    expected_exit: int,
    expected_counts: dict[str, int],
) -> None:
    checks = [DoctorCheck(f"check-{index}", status, "result") for index, status in enumerate(statuses)]

    status, exit_code, counts = doctor_module._summarize(checks)

    assert status == expected_status
    assert exit_code == expected_exit
    assert counts == expected_counts


def test_validate_model_reports_missing_model(tmp_path: Path) -> None:
    check = _validate_model(TEST_ASSET, tmp_path / "models")

    assert check.status == "error"
    assert check.message == "required model is missing"
    assert check.details["repo_id"] == TEST_ASSET.repo_id


def test_validate_model_reports_broken_symlink(tmp_path: Path) -> None:
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    destination = model_dir / TEST_ASSET.local_name
    destination.symlink_to(tmp_path / "missing-target", target_is_directory=True)

    check = _validate_model(TEST_ASSET, model_dir)

    assert check.status == "error"
    assert check.message == "broken model symlink"
    assert check.details["symlink_target"] == str(tmp_path / "missing-target")


def test_validate_model_reports_invalid_config(tmp_path: Path) -> None:
    model_dir = tmp_path / "models"
    _create_model(model_dir, config_text="{not-json")

    check = _validate_model(TEST_ASSET, model_dir)

    assert check.status == "error"
    assert check.message == "config.json is unreadable"
    assert check.details["config_error"]


@pytest.mark.parametrize("weight_bytes", [0, MIN_MODEL_WEIGHT_BYTES])
def test_validate_model_rejects_weights_at_or_below_50_mib(tmp_path: Path, weight_bytes: int) -> None:
    model_dir = tmp_path / "models"
    _create_model(model_dir, weight_bytes=weight_bytes)

    check = _validate_model(TEST_ASSET, model_dir)

    assert check.status == "error"
    assert check.message == "real model weights are missing or too small"
    assert check.details["weight_bytes"] == weight_bytes
    assert check.details["minimum_weight_bytes"] == MIN_MODEL_WEIGHT_BYTES


def test_validate_model_accepts_real_weights_above_50_mib(tmp_path: Path) -> None:
    model_dir = tmp_path / "models"
    _create_model(model_dir, weight_bytes=MIN_MODEL_WEIGHT_BYTES + 1)

    check = _validate_model(TEST_ASSET, model_dir)

    assert check.status == "ok"
    assert check.details["weight_bytes"] == MIN_MODEL_WEIGHT_BYTES + 1


def test_validate_model_accepts_weightless_tokenizer_manifest(tmp_path: Path) -> None:
    asset = ModelAsset(
        "example/tokenizer",
        "teacher",
        config_filename="processor_config.json",
        weights_required=False,
        required_files=("tokenizer.json",),
    )
    path = tmp_path / asset.local_name
    path.mkdir()
    (path / "processor_config.json").write_text("{}", encoding="utf-8")
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")

    assert _validate_model(asset, tmp_path).status == "ok"


def test_fix_repairs_broken_link_from_local_hf_cache(tmp_path: Path) -> None:
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    destination = model_dir / TEST_ASSET.local_name
    destination.symlink_to(tmp_path / "removed-snapshot", target_is_directory=True)

    hf_cache = tmp_path / "hub"
    snapshot = hf_cache / f"models--{TEST_ASSET.local_name}" / "snapshots" / "snapshot-sha"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    _sparse_file(snapshot / "model.safetensors", MIN_MODEL_WEIGHT_BYTES + 1)

    repairs = repair_model_links(model_dir, hf_cache, assets=(TEST_ASSET,))

    assert repairs == [
        {
            "repo_id": TEST_ASSET.repo_id,
            "path": str(destination),
            "target": str(snapshot.resolve()),
        }
    ]
    assert destination.is_symlink()
    assert destination.resolve() == snapshot.resolve()
    assert _validate_model(TEST_ASSET, model_dir).status == "ok"


def test_token_check_detects_project_env_without_exposing_secret(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    secret = "hf_private_test_value"
    env_path = tmp_path / ".env"
    env_path.write_text(f"# local credentials\nexport HF_TOKEN='{secret}'\n", encoding="utf-8")

    check = doctor_module._token_check(tmp_path)

    assert check.status == "ok"
    assert check.details["source"] == str(env_path)
    assert secret not in json.dumps(check.to_dict())


def test_gpu_check_reports_each_devices_free_vram(monkeypatch: pytest.MonkeyPatch) -> None:
    gib = 1024**3
    entered_devices: list[int] = []

    @contextmanager
    def fake_device(index: int):
        entered_devices.append(index)
        yield

    mem_get_info = Mock(side_effect=[(10 * gib, 24 * gib), (12 * gib, 24 * gib)])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "device", fake_device)
    monkeypatch.setattr(torch.cuda, "mem_get_info", mem_get_info)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: f"NVIDIA L4 #{index}")
    management = {
        "healthy": True,
        "path": "/usr/bin/nvidia-smi",
        "return_code": 0,
        "devices": [
            {"index": "0", "driver_version": "580.159.03"},
            {"index": "1", "driver_version": "580.159.03"},
        ],
    }
    monkeypatch.setattr(doctor_module, "_nvidia_management_probe", lambda: management)

    check = doctor_module._gpu_check()

    assert check.status == "ok"
    assert check.details == {
        "device_count": 2,
        "devices": [
            {
                "index": 0,
                "name": "NVIDIA L4 #0",
                "free_vram_bytes": 10 * gib,
                "total_vram_bytes": 24 * gib,
            },
            {
                "index": 1,
                "name": "NVIDIA L4 #1",
                "free_vram_bytes": 12 * gib,
                "total_vram_bytes": 24 * gib,
            },
        ],
        "nvidia_management": management,
    }
    assert entered_devices == [0, 1]
    assert mem_get_info.call_count == 2


def test_gpu_check_warns_when_nvml_management_is_broken(monkeypatch: pytest.MonkeyPatch) -> None:
    gib = 1024**3

    @contextmanager
    def fake_device(_index: int):
        yield

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "device", fake_device)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (20 * gib, 24 * gib))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _index: "NVIDIA L4")
    monkeypatch.setattr(
        doctor_module,
        "_nvidia_management_probe",
        lambda: {
            "healthy": False,
            "path": "/usr/bin/nvidia-smi",
            "return_code": 255,
            "error": "Failed to initialize NVML: Driver/library version mismatch",
        },
    )

    check = doctor_module._gpu_check()

    assert check.status == "warning"
    assert "NVML is unhealthy" in check.message
    assert check.details["device_count"] == 1
    assert check.details["nvidia_management"]["return_code"] == 255


def test_nvidia_management_probe_parses_healthy_driver_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    run = Mock(
        return_value=Mock(
            returncode=0,
            stdout="0, 580.159.03\n1, 580.159.03\n",
            stderr="",
        )
    )
    monkeypatch.setattr(doctor_module.shutil, "which", lambda _name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(doctor_module.subprocess, "run", run)

    probe = doctor_module._nvidia_management_probe()

    assert probe == {
        "healthy": True,
        "path": "/usr/bin/nvidia-smi",
        "return_code": 0,
        "devices": [
            {"index": "0", "driver_version": "580.159.03"},
            {"index": "1", "driver_version": "580.159.03"},
        ],
    }
    assert run.call_args.args[0] == [
        "/usr/bin/nvidia-smi",
        "--query-gpu=index,driver_version",
        "--format=csv,noheader,nounits",
    ]
    assert run.call_args.kwargs["timeout"] == 10


def test_gpu_warning_capture_promotes_ok_probe_and_deduplicates() -> None:
    @doctor_module._capture_probe_warnings
    def probe() -> DoctorCheck:
        warnings.warn("Can't initialize NVML", UserWarning)
        warnings.warn("Can't initialize NVML", UserWarning)
        return DoctorCheck("gpu", "ok", "one CUDA device visible")

    check = probe()

    assert check.status == "warning"
    assert check.details["warnings"] == ["Can't initialize NVML"]
