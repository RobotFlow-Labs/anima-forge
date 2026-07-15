"""Truthfulness contracts for the trained-student listing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from typer.testing import CliRunner

from forge.config import ForgeConfig, StudentConfig, apply_student_variant
from forge.model_registry import ModelRegistry


def test_students_list_reads_trained_model_registry(tmp_path: Path) -> None:
    from forge.cli_v2 import app

    checkpoint = tmp_path / "student.pt"
    checkpoint.write_bytes(b"real checkpoint fixture")
    registry_dir = tmp_path / "registry"
    entry = ModelRegistry(registry_dir).register(
        checkpoint,
        variant="nano",
        config=ForgeConfig.default(),
        name="forge-nano-trained",
        tags=["tested"],
    )

    result = CliRunner().invoke(
        app,
        ["students", "list", "--registry-dir", str(registry_dir), "--json"],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert [item["model_id"] for item in payload] == [entry.model_id]
    assert payload[0]["name"] == "forge-nano-trained"
    assert all(item["name"] not in {"openvla", "rdt2", "smolvla"} for item in payload)


def test_students_list_empty_registry_is_empty_json(tmp_path: Path) -> None:
    from forge.cli_v2 import app

    result = CliRunner().invoke(
        app,
        ["students", "list", "--registry-dir", str(tmp_path / "empty"), "--json"],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == []


def _real_provenance(model_dir: str) -> dict[str, str]:
    return {
        "vision": "real",
        "language": "real",
        "labels": "real",
        "model_dir": model_dir,
        "git_sha": "a" * 40,
        "forge_version": "3.0.0",
        "torch_version": "2.10.0+cu128",
    }


def _write_hub_package_inputs(tmp_path: Path, *, memory_gate: bool = True) -> tuple[Path, Path]:
    provenance = _real_provenance(str(tmp_path / "private" / "models"))
    checkpoint = tmp_path / "final.pt"
    student_config = asdict(apply_student_variant(StudentConfig(allow_mock=False), "nano"))
    student_config["action_head_type"] = "flow"
    torch.save(
        {
            "step": 2000,
            "model_state_dict": {"weight": torch.arange(4, dtype=torch.float32)},
            "optimizer_state_dict": {"state": {0: {"step": torch.tensor(2)}}},
            "scheduler_state_dict": {"last_epoch": 2},
            "provenance": provenance,
            "student_config": student_config,
        },
        checkpoint,
    )
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    summary = tmp_path / "pipeline_summary.json"
    summary.write_text(
        json.dumps(
            {
                "status": "completed",
                "config": "nano",
                "distill": {
                    "total_steps": 2000,
                    "checkpoint_sha256": checkpoint_sha256,
                    "initial_loss": 1.5,
                    "final_loss": 0.25,
                    "loss_reduction_percent": 83.333,
                    "best_loss": 0.2,
                    "steps_per_second": 0.5,
                    "elapsed_seconds": 4000.0,
                    "cuda_memory": {
                        "peak_reserved_gib": 16.0,
                        "total_gib": 22.278,
                        "peak_reserved_utilization": 0.7182,
                        "target_60_80_percent_met": memory_gate,
                    },
                    "provenance": provenance,
                },
            }
        ),
        encoding="utf-8",
    )
    return checkpoint, summary


def test_students_package_builds_private_path_free_hub_payload(tmp_path: Path) -> None:
    from forge.cli_v2 import app

    checkpoint, summary = _write_hub_package_inputs(tmp_path)
    output = tmp_path / "hub-package"
    result = CliRunner().invoke(
        app,
        [
            "students",
            "package",
            str(checkpoint),
            "--training-summary",
            str(summary),
            "--output-dir",
            str(output),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["status"] == "completed"
    assert result.stderr == ""
    packaged = torch.load(output / "forge-nano.pt", map_location="cpu", weights_only=True)
    assert set(packaged) == {"format", "step", "model_state_dict", "provenance", "student_config"}
    assert packaged["format"] == "forge.hub-checkpoint.v1"
    assert torch.equal(packaged["model_state_dict"]["weight"], torch.arange(4, dtype=torch.float32))
    assert packaged["provenance"]["model_dir"] == "external Hub model IDs in student_config"
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    artifact_hash = hashlib.sha256((output / "forge-nano.pt").read_bytes()).hexdigest()
    assert manifest["artifact_sha256"] == artifact_hash == payload["artifact_sha256"]
    assert manifest["source_checkpoint"] == "final.pt"
    assert manifest["stripped_training_state"] == ["optimizer_state_dict", "scheduler_state_dict"]
    assert manifest["training"]["cuda_memory"]["target_60_80_percent_met"] is True
    public_text = (output / "README.md").read_text(encoding="utf-8") + json.dumps(manifest)
    assert str(tmp_path) not in public_text
    assert "/" + "mnt/" not in public_text


def test_students_package_rejects_checkpoint_that_missed_memory_gate(tmp_path: Path) -> None:
    from forge.cli_v2 import app

    checkpoint, summary = _write_hub_package_inputs(tmp_path, memory_gate=False)
    output = tmp_path / "rejected-package"
    result = CliRunner().invoke(
        app,
        [
            "students",
            "package",
            str(checkpoint),
            "--training-summary",
            str(summary),
            "--output-dir",
            str(output),
            "--json",
        ],
    )

    assert result.exit_code == 2
    assert result.stdout == ""
    assert "did not pass the required 60–80% VRAM gate" in json.loads(result.stderr)["error"]
    assert not output.exists()


def test_students_package_rejects_summary_for_different_checkpoint(tmp_path: Path) -> None:
    from forge.cli_v2 import app

    checkpoint, summary = _write_hub_package_inputs(tmp_path)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["model_state_dict"]["weight"] = torch.arange(4, dtype=torch.float32) + 1
    torch.save(payload, checkpoint)

    result = CliRunner().invoke(
        app,
        [
            "students",
            "package",
            str(checkpoint),
            "--training-summary",
            str(summary),
            "--output-dir",
            str(tmp_path / "rejected-package"),
            "--json",
        ],
    )

    assert result.exit_code == 2
    assert result.stdout == ""
    assert "not bound to this exact checkpoint" in json.loads(result.stderr)["error"]


def test_students_package_missing_paths_preserve_json_error_contract(tmp_path: Path) -> None:
    from forge.cli_v2 import app

    result = CliRunner().invoke(
        app,
        [
            "students",
            "package",
            str(tmp_path / "missing.pt"),
            "--training-summary",
            str(tmp_path / "missing.json"),
            "--output-dir",
            str(tmp_path / "output"),
            "--json",
        ],
    )

    assert result.exit_code == 2
    assert result.stdout == ""
    assert set(json.loads(result.stderr)) == {"error"}


def test_students_package_rejects_private_mapping_keys_and_non_tensor_state(tmp_path: Path) -> None:
    from forge.hub_package import package_hub_checkpoint

    checkpoint, summary = _write_hub_package_inputs(tmp_path)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    private_key = "C:" + "\\" + "Users" + "\\" + "operator" + "\\" + "model"
    payload["student_config"][private_key] = "private"
    torch.save(payload, checkpoint)
    summary_payload = json.loads(summary.read_text(encoding="utf-8"))
    summary_payload["distill"]["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    summary.write_text(json.dumps(summary_payload), encoding="utf-8")

    with pytest.raises(ValueError, match="private values"):
        package_hub_checkpoint(checkpoint, summary, tmp_path / "private-output", repo_id="robotflowlabs/forge-nano")

    payload["student_config"].pop(private_key)
    payload["model_state_dict"]["not-a-tensor"] = "invalid"
    torch.save(payload, checkpoint)
    summary_payload["distill"]["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    summary.write_text(json.dumps(summary_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="only named tensors"):
        package_hub_checkpoint(checkpoint, summary, tmp_path / "invalid-output", repo_id="robotflowlabs/forge-nano")


def test_students_package_rejects_model_card_control_character_injection(tmp_path: Path) -> None:
    from forge.hub_package import package_hub_checkpoint

    checkpoint, summary = _write_hub_package_inputs(tmp_path)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["student_config"]["language_model"] = "Qwen/Qwen3-0.6B\nprivate-card-section"
    torch.save(payload, checkpoint)
    summary_payload = json.loads(summary.read_text(encoding="utf-8"))
    summary_payload["distill"]["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    summary.write_text(json.dumps(summary_payload), encoding="utf-8")

    with pytest.raises(ValueError, match="private values"):
        package_hub_checkpoint(checkpoint, summary, tmp_path / "invalid-card", repo_id="robotflowlabs/forge-nano")


@pytest.mark.parametrize(
    "private_value",
    [
        "/" + "tmp/operator/checkpoint.pt",
        "/" + "root/.cache/model",
        "/" + "opt/forge/private",
        "/" + "srv/weights/student.pt",
        "C:" + "\\" + "Windows" + "\\" + "System32",
        "\\" * 2 + "internal-host" + "\\" + "weights",
        "ghp" + "_" + "a" * 36,
        "github" + "_pat_" + "a" * 40,
        "sk" + "-proj-" + "a" * 32,
        "AKIA" + "A" * 16,
        "aws_secret_access_key=" + "a" * 40,
        "-----BEGIN " + "OPENSSH PRIVATE KEY-----",
    ],
)
def test_hub_privacy_matcher_rejects_sensitive_nested_values_and_keys(private_value: str) -> None:
    from forge.hub_package import _private_strings

    assert _private_strings({"metadata": [{"value": private_value}]}) == ["metadata[0].value"]
    assert _private_strings({private_value: "value"}) == [f"{private_value}.<key>"]


@pytest.mark.parametrize(
    "public_value",
    [
        "https://github.com/openai/example",
        "/optimization/results",
        "sk-short-example",
        "AKIA_EXAMPLE_PLACEHOLDER",
        "-----BEGIN PUBLIC KEY-----",
        "version 10.16.0.72",
    ],
)
def test_hub_privacy_matcher_avoids_public_false_positives(public_value: str) -> None:
    from forge.hub_package import _private_strings

    assert _private_strings({"metadata": public_value}) == []


def test_students_package_requires_complete_non_mock_student_contract(tmp_path: Path) -> None:
    from forge.hub_package import package_hub_checkpoint

    checkpoint, summary = _write_hub_package_inputs(tmp_path)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["student_config"].pop("bridge_n_queries")
    torch.save(payload, checkpoint)
    summary_payload = json.loads(summary.read_text(encoding="utf-8"))
    summary_payload["distill"]["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    summary.write_text(json.dumps(summary_payload), encoding="utf-8")

    with pytest.raises(ValueError, match="missing required fields: bridge_n_queries"):
        package_hub_checkpoint(checkpoint, summary, tmp_path / "incomplete", repo_id="robotflowlabs/forge-nano")

    payload["student_config"]["bridge_n_queries"] = 64
    payload["student_config"]["allow_mock"] = True
    torch.save(payload, checkpoint)
    summary_payload["distill"]["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    summary.write_text(json.dumps(summary_payload), encoding="utf-8")

    with pytest.raises(ValueError, match="allow_mock must be false"):
        package_hub_checkpoint(checkpoint, summary, tmp_path / "mock-enabled", repo_id="robotflowlabs/forge-nano")
