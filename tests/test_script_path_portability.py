"""Regression tests for portable defaults in standalone maintenance scripts."""

from __future__ import annotations

import runpy
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PORTABLE_SCRIPTS = (
    "scripts/compress_and_push.py",
    "scripts/verify_teachers.py",
)
RETIRED_STANDALONE_SCRIPTS = (
    "scripts/baseline_all_modules.py",
    "scripts/benchmark_cuda.py",
    "scripts/demo.py",
    "scripts/download_hf_repo.py",
    "scripts/download_models.py",
    "scripts/eval_turboquant_hf.py",
    "scripts/gpu_validation.py",
    "scripts/run_pipeline.sh",
    "scripts/train_all_modules.py",
    "scripts/train_all_variants.py",
    "scripts/train_real_data.py",
)
HOST_PATHS = (
    "/" + "mnt/forge-data",
    "/" + "mnt/development",
    "/" + "home/datai",
    "datai_" + "srv",
)


@pytest.mark.parametrize("relative_path", PORTABLE_SCRIPTS)
def test_maintenance_script_has_no_host_specific_path(relative_path: str) -> None:
    source = (ROOT / relative_path).read_text(encoding="utf-8")
    assert not [path for path in HOST_PATHS if path in source]


def test_release_workflows_are_owned_by_the_forge_cli() -> None:
    assert not [path for path in RETIRED_STANDALONE_SCRIPTS if (ROOT / path).exists()]


@pytest.mark.parametrize("relative_path", ("scripts/compress_and_push.py",))
def test_script_model_default_is_repo_relative(
    relative_path: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FORGE_MODEL_DIR", raising=False)
    namespace = runpy.run_path(str(ROOT / relative_path))
    assert namespace["MODEL_DIR"] == Path("models")


def test_teacher_verifier_honors_portable_dataset_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.verify_teachers import parse_args

    monkeypatch.setenv("FORGE_DATASET_DIR", "custom-datasets")
    monkeypatch.setattr(sys, "argv", ["verify_teachers.py"])
    assert parse_args().dataset_dir == "custom-datasets"


def test_gpu_fit_profiler_parses_gpu_and_process_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import gpu_fit_profiler

    def fake_run(command: list[str], *, text: bool = True) -> str:
        assert text is True
        query = next(value for value in command if value.startswith("--query-"))
        if query.startswith("--query-gpu=index,name"):
            return "0, NVIDIA L4, 23034, 1234, 21800, 91, 50, 68.5, 61"
        if query.startswith("--query-gpu=index,uuid"):
            return "0, GPU-abcd"
        if query.startswith("--query-compute-apps"):
            return "GPU-abcd, 42, 1024\nGPU-abcd, 99, 2048"
        raise AssertionError(command)

    monkeypatch.setattr(gpu_fit_profiler.shutil, "which", lambda _name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(gpu_fit_profiler, "_run", fake_run)

    assert gpu_fit_profiler.query_gpus() == [
        {
            "index": 0,
            "name": "NVIDIA L4",
            "memory_total_mb": 23034.0,
            "memory_used_mb": 1234.0,
            "memory_free_mb": 21800.0,
            "utilization_gpu": 91.0,
            "utilization_mem": 50.0,
            "power_watts": 68.5,
            "temp_c": 61.0,
        }
    ]
    assert gpu_fit_profiler.map_gpu_uuid_to_index() == {"GPU-abcd": 0}
    assert gpu_fit_profiler.query_proc_gpu_memory([42]) == {"GPU-abcd": 1024}


def test_gpu_fit_profiler_falls_back_when_nvml_is_broken(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import gpu_fit_profiler

    from forge import gpu_utils

    monkeypatch.setattr(gpu_fit_profiler.shutil, "which", lambda _name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        gpu_fit_profiler,
        "_run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(subprocess.CalledProcessError(1, "nvidia-smi")),
    )
    monkeypatch.setattr(
        gpu_utils,
        "_torch_gpu_samples",
        lambda: [
            {
                "index": 2,
                "name": "NVIDIA L4",
                "memory_total_mib": 23034,
                "memory_used_mib": 1234,
                "memory_free_mib": 21800,
                "utilization_gpu": -1,
                "utilization_memory": -1,
                "metrics_source": "torch.cuda.mem_get_info",
            }
        ],
    )

    assert gpu_fit_profiler.query_gpus() == [
        {
            "index": 2,
            "name": "NVIDIA L4",
            "memory_total_mb": 23034.0,
            "memory_used_mb": 1234.0,
            "memory_free_mb": 21800.0,
            "utilization_gpu": -1.0,
            "utilization_mem": -1.0,
            "power_watts": -1.0,
            "temp_c": -1.0,
        }
    ]
    assert gpu_fit_profiler.map_gpu_uuid_to_index() == {}


def test_matrix_profile_validator_requires_both_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from scripts import validate_matrix_profiles

    matrix = tmp_path / "matrix_results.csv"
    matrix.write_text("step,status\ntier3_pipeline_short,passed\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_matrix_profiles.py",
            "--matrix-results",
            str(matrix),
            "--matrix-dir",
            str(tmp_path),
        ],
    )

    assert validate_matrix_profiles.main() == 1
    (tmp_path / "tier3_pipeline_short_prof.csv").touch()
    (tmp_path / "tier3_pipeline_short_prof.csv.json").touch()
    assert validate_matrix_profiles.main() == 0


def test_gpu_fit_profiler_accepts_bare_output_filename(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from scripts import gpu_fit_profiler

    observed: dict[str, object] = {}

    def fake_launch(command: str, interval: float, csv_path: str, out_json: bool) -> int:
        observed.update(command=command, interval=interval, csv_path=csv_path, out_json=out_json)
        return 0

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(gpu_fit_profiler, "launch_and_monitor", fake_launch)
    monkeypatch.setattr(
        sys,
        "argv",
        ["gpu_fit_profiler.py", "--command", "true", "--out", "profile.csv", "--json"],
    )

    assert gpu_fit_profiler.main() == 0
    assert observed == {
        "command": "true",
        "interval": 5.0,
        "csv_path": "profile.csv",
        "out_json": True,
    }
