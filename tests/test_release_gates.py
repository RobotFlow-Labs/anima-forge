"""Static contracts for release workflows and maintained utility scripts."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import tomllib
from pathlib import Path

import pytest
import torch
import yaml


def _text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_ci_workflow_covers_the_full_maintained_python_surface() -> None:
    workflow = _text(".github/workflows/ci.yaml")
    yaml.compose(workflow)

    assert "uv run ruff check src/ scripts/ tests/" in workflow
    assert "uv run ruff format --check src/ scripts/ tests/" in workflow
    assert "uv run mypy src/forge/ scripts/" in workflow
    assert "--ignore-missing-imports" not in workflow
    assert "continue-on-error" not in workflow


def test_ci_and_release_gate_the_actual_public_snapshot() -> None:
    ci = _text(".github/workflows/ci.yaml")
    release = _text(".github/workflows/release.yml")

    assert "python scripts/check_public_snapshot.py --json" in ci
    assert "python scripts/check_public_snapshot.py --json" in release


def test_readme_tracks_live_ci_and_the_enforced_local_gate() -> None:
    readme = _text("README.md")

    assert "actions/workflows/ci.yaml/badge.svg?branch=develop" in readme
    assert "shields.io/badge/tests-" not in readme
    assert "uv run ruff check src/ scripts/ tests/" in readme
    assert "uv run ruff format --check src/ scripts/ tests/" in readme
    assert "uv run mypy src/forge/ scripts/" in readme
    assert 'uv run pytest tests/ -m "not gpu"' in readme


def test_readme_scopes_json_support_to_automation_commands() -> None:
    readme = _text("README.md")

    assert "Every command speaks `--json`" not in readme
    assert "Automation-facing status and artifact commands provide strict `--json`" in readme


def test_public_architecture_docs_are_fail_closed_and_match_canonical_configs() -> None:
    architecture = _text("docs/ARCHITECTURE.md")
    profiler = _text("docs/PROFILER.md")

    for stale_claim in (
        "graceful fallback to mock",
        "80% of student parameters",
        "rank-32 LoRA",
        "~2M parameters",
    ):
        assert stale_claim not in architecture
    assert "Fail-closed runtime assets" in architecture

    for variant in ("micro", "nano", "small", "medium"):
        config = yaml.safe_load(_text(f"configs/forge_{variant}.yaml"))
        expected_cells = (
            variant,
            str(config["distill"]["learning_rate"]),
            str(config["student"]["lora_rank"]),
            str(config["student"]["bridge_n_layers"]),
            str(config["distill"]["batch_size"]),
        )
        row = next(line for line in profiler.splitlines() if line.startswith(f"| {variant}"))
        actual_cells = tuple(cell.strip() for cell in row.strip("|").split("|"))
        assert actual_cells == expected_cells


def test_workflows_use_node24_native_action_majors() -> None:
    workflows = _text(".github/workflows/ci.yaml") + _text(".github/workflows/release.yml")

    assert "actions/checkout@v4" not in workflows
    assert "astral-sh/setup-uv@v3" not in workflows
    assert "actions/checkout@9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0 # v7.0.0" in workflows
    assert "astral-sh/setup-uv@11f9893b081a58869d3b5fccaea48c9e9e46f990 # v8.3.2" in workflows


def test_workflow_actions_are_pinned_to_immutable_commits() -> None:
    for workflow_path in sorted(Path(".github/workflows").glob("*.y*ml")):
        workflow = _text(str(workflow_path))
        for line_number, line in enumerate(workflow.splitlines(), start=1):
            match = re.search(r"\buses:\s*([^\s#]+)", line)
            if match is None or match.group(1).startswith("./"):
                continue
            reference = match.group(1)
            assert re.fullmatch(r"[^@\s]+@[0-9a-f]{40}", reference), (
                f"{workflow_path}:{line_number} action is not pinned to a full commit SHA: {reference}"
            )
            assert "#" in line, f"{workflow_path}:{line_number} pinned action needs a readable version comment"


def test_gpu_ci_requires_an_explicit_manual_input() -> None:
    workflow = _text(".github/workflows/ci.yaml")

    assert "workflow_dispatch:" in workflow
    assert "run_gpu_tests:" in workflow
    assert "if: github.event_name == 'workflow_dispatch' && inputs.run_gpu_tests" in workflow
    assert "github.ref == 'refs/heads/main'" not in workflow


def test_gpu_ci_produces_and_preserves_real_profile_evidence() -> None:
    workflow = _text(".github/workflows/ci.yaml")
    gpu_job = workflow.split("  gpu-test:", 1)[1].split("\n  wheel-smoke:", 1)[0]

    assert "scripts/gpu_fit_profiler.py" in gpu_job
    assert 'payload["status"] == "finished"' in gpu_job
    assert 'assert payload["rows"]' in gpu_job
    assert "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02 # v4" in gpu_job
    assert "if-no-files-found: error" in gpu_job
    assert "No matrix_results.csv found" not in gpu_job


def test_wheel_smoke_imports_every_mandatory_teacher_runtime() -> None:
    workflow = _text(".github/workflows/ci.yaml")
    release_workflow = _text(".github/workflows/release.yml")
    verifier = _text("scripts/verify_installed_runtime.py")

    for runtime_import in (
        "forge.teachers.molmoact2_adapter import MolmoAct2Adapter",
        "forge.teachers.rdt2_adapter import RDT2Adapter",
        "forge.teachers.smolvla_adapter import SmolVLAAdapter",
        "forge.teachers.vla_jepa_adapter import VLAJEPAAdapter",
        "forge.teachers.registry import get_registry",
        "forge.vendor.rdt2 import RDTRunner",
        "lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy",
        "lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy",
        "lerobot.policies.vla_jepa.modeling_vla_jepa import VLAJEPAPolicy",
    ):
        assert runtime_import in verifier
    assert workflow.count("scripts/verify_installed_runtime.py") == 3
    assert "scripts\\verify_installed_runtime.py" in workflow
    assert release_workflow.count("scripts/verify_installed_runtime.py") == 3
    assert "registry.create(name)" in verifier


def test_wheel_smoke_exercises_the_public_installer() -> None:
    workflow = _text(".github/workflows/ci.yaml")

    assert "shellcheck install.sh" in workflow
    assert './install.sh --cpu --from-wheel "$(echo dist/*.whl)" --no-modify-path' in workflow
    assert "/tmp/forge-tool-bin/forge --version" in workflow
    assert "/tmp/forge-tool-envs/anima-forge/bin/python" in workflow
    assert "Smoke pipx installer backend" in workflow
    assert "--cpu --backend pipx --from-wheel" in workflow
    assert "/tmp/forge-pipx-home/venvs/anima-forge/bin/python" in workflow
    assert "scripts/build_release_kit.py --check" in workflow


def test_testpypi_acceptance_mechanically_gates_production() -> None:
    workflow = _text(".github/workflows/release.yml")
    yaml.compose(workflow)

    testpypi = workflow.split("  test-pypi:", 1)[1].split("\n  test-pypi-acceptance:", 1)[0]
    tag_acceptance = workflow.split("  test-pypi-acceptance:", 1)[1].split("\n  pypi:", 1)[0]
    production = workflow.split("\n  pypi:", 1)[1]
    assert "if: github.event_name == 'workflow_dispatch'" in testpypi
    assert "skip-existing: true" in testpypi
    assert "contents: read" in testpypi
    assert "actions/checkout@9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0 # v7.0.0" in testpypi
    assert "scripts/verify_testpypi_artifact.py" in testpypi
    assert "testpypi-dist/*.whl" in testpypi
    assert "--extra-index-url" not in testpypi
    assert "/tmp/forge-testpypi/bin/forge info --json" in testpypi
    assert "scripts/verify_installed_runtime.py" in testpypi
    assert "if: github.event_name == 'push'" in tag_acceptance
    assert "scripts/verify_testpypi_artifact.py" in tag_acceptance
    assert "gh-action-pypi-publish" not in tag_acceptance
    assert "testpypi-dist/*.whl" in tag_acceptance
    assert "/tmp/forge-testpypi-tag/bin/forge info --json" in tag_acceptance
    assert "scripts/verify_installed_runtime.py" in tag_acceptance
    assert "needs: [wheel-smoke, test-pypi-acceptance]" in production


def test_release_distribution_build_uses_commit_epoch_and_sha() -> None:
    workflow = _text(".github/workflows/release.yml")
    build = workflow.split("  build:", 1)[1].split("\n  wheel-smoke:", 1)[0]

    assert 'SOURCE_DATE_EPOCH="$(git show -s --format=%ct HEAD)"' in build
    assert 'FORGE_GIT_SHA="$(git rev-parse HEAD)"' in build
    assert "uv build" in build


def test_static_analysis_targets_the_supported_python_version() -> None:
    metadata = tomllib.loads(_text("pyproject.toml"))

    assert metadata["project"]["requires-python"] == ">=3.12,<3.13"
    assert metadata["tool"]["ruff"]["target-version"] == "py312"
    assert metadata["tool"]["mypy"]["check_untyped_defs"] is True


def test_compression_script_requires_modern_onnx_export() -> None:
    script = _text("scripts/compress_and_push.py")

    assert "dynamic_axes" not in script
    assert 'dynamic_shapes={"pixel_values": {0: batch}}' in script
    assert "Required ONNX export failed" in script
    assert "ONNX export failed (non-critical)" not in script


def test_required_onnx_export_removes_partial_artifact_and_raises(monkeypatch, tmp_path) -> None:
    spec = importlib.util.spec_from_file_location("forge_compress_and_push", "scripts/compress_and_push.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    output = tmp_path / "model.onnx"

    def fail_export(*_args, **_kwargs):
        output.write_bytes(b"partial")
        raise RuntimeError("exporter exploded")

    monkeypatch.setattr(torch.onnx, "export", fail_export)
    with pytest.raises(RuntimeError, match="Required ONNX export failed for test-model"):
        module._export_required_onnx(torch.nn.Identity(), torch.zeros(1, 3), output, "test-model")

    assert not output.exists()


def test_compression_script_returns_failure_when_a_model_fails(monkeypatch, tmp_path) -> None:
    spec = importlib.util.spec_from_file_location("forge_compress_and_push_exit", "scripts/compress_and_push.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    model_dir = tmp_path / "models"
    (model_dir / "test-model").mkdir(parents=True)
    monkeypatch.setattr(module, "MODEL_DIR", model_dir)
    monkeypatch.setattr(
        module,
        "MODELS",
        {
            "test-model": {
                "category": "small",
                "hf_name": "test-model-export",
                "size_gb": 0.1,
            }
        },
    )

    def fail_export(*_args, **_kwargs):
        raise RuntimeError("required export failed")

    monkeypatch.setattr(module, "export_small_model", fail_export)
    output = tmp_path / "output"
    exit_code = module.main(["--models", "test-model", "--output-dir", str(output)])

    assert exit_code == 1
    results = json.loads((output / "compress_results.json").read_text(encoding="utf-8"))
    assert results[0]["status"] == "failed"
    assert results[0]["error"] == "required export failed"


def test_compression_script_returns_failure_when_requested_weights_are_missing(monkeypatch, tmp_path) -> None:
    spec = importlib.util.spec_from_file_location("forge_compress_and_push_missing", "scripts/compress_and_push.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    monkeypatch.setattr(module, "MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr(
        module,
        "MODELS",
        {
            "missing-model": {
                "category": "small",
                "hf_name": "missing-model-export",
                "size_gb": 0.1,
            }
        },
    )

    assert module.main(["--models", "missing-model", "--dry-run"]) == 1
