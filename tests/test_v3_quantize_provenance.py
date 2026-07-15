"""PRD-36 provenance and trained-input contracts for public quantization."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from torch import nn
from typer.testing import CliRunner

from forge.cli_commands.quantize import load_student_for_quant
from forge.cli_v2 import app
from forge.provenance import build_provenance
from forge.quantize.serialization import PACKED_STATE_KEY, unpack_state_dict


class TinyStudent(nn.Module):
    """Small state-compatible stand-in; provenance still uses real modules."""

    def __init__(self, _config, model_dir=None):
        super().__init__()
        self._model_dir = Path(model_dir or "./models")
        self.vision_encoder = nn.Linear(2, 2)
        self.language = nn.Linear(2, 2)
        self.action_head = nn.Linear(2, 2)


def _provenance(tmp_path: Path, *, mock: bool) -> dict[str, str]:
    status = "mock" if mock else "real"
    return build_provenance(
        vision=status,
        language=status,
        labels=status,
        model_dir=tmp_path / "models",
        git_sha="quant-test-sha",
        forge_version="3.0.0-test",
        torch_version=str(torch.__version__),
    )


@pytest.fixture
def tiny_quant_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("forge.student.FORGEStudent", TinyStudent)
    monkeypatch.setattr(
        "forge.quantize.quantize_model_with_config",
        lambda model, _config, **_kwargs: model,
    )
    monkeypatch.setattr(
        "forge.quantize.create_quant_profile",
        lambda *_args, **_kwargs: SimpleNamespace(
            avg_bits=3.0,
            compressed_size_mb=0.001,
        ),
    )
    monkeypatch.setattr(
        "forge.quantize.benchmark_quantization",
        lambda *_args, **_kwargs: {"method": "qvla", "mse": 0.0},
    )


def _invoke_json(*args: str):
    return CliRunner().invoke(app, ["quantize", *args, "--json"])


def test_quantize_run_requires_checkpoint_without_mock_opt_in(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FORGE_ALLOW_MOCK", raising=False)
    output = tmp_path / "must-not-exist.pt"

    result = _invoke_json(
        "run",
        "--method",
        "qvla",
        "--device",
        "cpu",
        "--output",
        str(output),
    )

    assert result.exit_code == 2
    assert result.stdout == ""
    assert "requires a trained --checkpoint" in json.loads(result.stderr)["error"]
    assert not output.exists()


def test_quantize_run_rejects_non_packable_bit_width(tmp_path: Path) -> None:
    output = tmp_path / "must-not-exist.pt"
    result = _invoke_json(
        "run",
        "--method",
        "qvla",
        "--bits",
        "3",
        "--device",
        "cpu",
        "--allow-mock",
        "--output",
        str(output),
    )

    assert result.exit_code == 2
    assert "supports --bits 4 or 8" in json.loads(result.stderr)["error"]
    assert not output.exists()


def test_quantize_run_rejects_unknown_method_as_clean_json() -> None:
    result = _invoke_json(
        "run",
        "--method",
        "not-a-backend",
        "--device",
        "cpu",
        "--allow-mock",
    )

    assert result.exit_code == 2
    assert result.stdout == ""
    error = json.loads(result.stderr)["error"]
    assert "Unknown quantization method" in error
    assert "Traceback" not in result.stderr


def test_quantize_run_rejects_explicit_missing_config_as_clean_json(tmp_path: Path) -> None:
    result = _invoke_json(
        "run",
        "--config",
        str(tmp_path / "missing.yaml"),
        "--method",
        "qvla",
        "--device",
        "cpu",
        "--allow-mock",
    )

    assert result.exit_code == 2
    assert result.stdout == ""
    assert json.loads(result.stderr) == {"error": f"Config file not found: {tmp_path / 'missing.yaml'}"}


@pytest.mark.parametrize(
    ("command", "expected"),
    [("run", "Refusing to quantize"), ("bench", "Refusing to benchmark")],
)
def test_quantize_commands_refuse_mock_checkpoint_by_default(
    command: str,
    expected: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FORGE_ALLOW_MOCK", raising=False)
    checkpoint = tmp_path / "mock.pt"
    torch.save(
        {
            "model_state_dict": {},
            "provenance": _provenance(tmp_path, mock=True),
        },
        checkpoint,
    )

    result = _invoke_json(
        command,
        "--method",
        "qvla",
        "--device",
        "cpu",
        "--checkpoint",
        str(checkpoint),
    )

    assert result.exit_code == 2
    assert result.stdout == ""
    error = json.loads(result.stderr)["error"]
    assert expected in error
    assert "--allow-mock" in error
    assert "Traceback" not in result.stderr


def test_quantize_run_allow_mock_writes_wrapped_provenance_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tiny_quant_runtime: None,
) -> None:
    monkeypatch.delenv("FORGE_ALLOW_MOCK", raising=False)
    output = tmp_path / "nested" / "quantized.pt"

    result = _invoke_json(
        "run",
        "--method",
        "qvla",
        "--device",
        "cpu",
        "--allow-mock",
        "--output",
        str(output),
    )

    assert result.exit_code == 0, result.output
    response = json.loads(result.stdout)
    artifact = torch.load(output, map_location="cpu", weights_only=True)
    assert set(artifact) == {PACKED_STATE_KEY, "config_sha256", "quantization", "provenance"}
    assert artifact["provenance"] == response["provenance"]
    assert artifact["config_sha256"] == response["config_sha256"]
    assert artifact["provenance"]["labels"] == "mock"
    assert artifact[PACKED_STATE_KEY]
    assert response["serialization_schema"] == "forge.packed-state.v1"


def test_quantize_run_profiles_the_requested_uniform_width(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tiny_quant_runtime: None,
) -> None:
    observed: dict[str, object] = {}

    def record_profile(*_args, **kwargs):
        observed.update(kwargs)
        return SimpleNamespace(avg_bits=8.0, compressed_size_mb=0.001)

    monkeypatch.setattr("forge.quantize.create_quant_profile", record_profile)
    result = _invoke_json(
        "run",
        "--method",
        "qvla",
        "--bits",
        "8",
        "--device",
        "cpu",
        "--allow-mock",
        "--output",
        str(tmp_path / "q8.pt"),
    )

    assert result.exit_code == 0, result.output
    assert observed["uniform_bits"] == 8


def test_quantize_runtime_failure_is_one_stderr_json_document(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tiny_quant_runtime: None,
) -> None:
    def fail_quantization(*_args, **_kwargs):
        print("quantizer chatter that must not corrupt the error")
        raise ValueError("quantizer runtime unavailable")

    monkeypatch.setattr("forge.quantize.quantize_model_with_config", fail_quantization)

    result = _invoke_json(
        "run",
        "--method",
        "qvla",
        "--device",
        "cpu",
        "--allow-mock",
        "--output",
        str(tmp_path / "must-not-exist.pt"),
    )

    assert result.exit_code == 2
    assert result.stdout == ""
    assert json.loads(result.stderr) == {"error": "quantizer runtime unavailable"}


def test_quantize_run_preserves_real_checkpoint_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tiny_quant_runtime: None,
) -> None:
    monkeypatch.delenv("FORGE_ALLOW_MOCK", raising=False)
    checkpoint = tmp_path / "trained.pt"
    checkpoint_alias = tmp_path / "trained-link.pt"
    output = tmp_path / "quantized.pt"
    provenance = _provenance(tmp_path, mock=False)
    torch.save(
        {
            "model_state_dict": TinyStudent(None).state_dict(),
            "provenance": provenance,
        },
        checkpoint,
    )
    checkpoint_alias.symlink_to(checkpoint)
    observed: dict[str, object] = {}

    def record_resolved_checkpoint(config_path, checkpoint=None, **kwargs):
        observed["checkpoint"] = checkpoint
        return load_student_for_quant(config_path, checkpoint=checkpoint, **kwargs)

    monkeypatch.setattr("forge.cli_commands.quantize.load_student_for_quant", record_resolved_checkpoint)

    result = _invoke_json(
        "run",
        "--method",
        "qvla",
        "--device",
        "cpu",
        "--checkpoint",
        str(checkpoint_alias),
        "--output",
        str(output),
    )

    assert result.exit_code == 0, result.output
    response = json.loads(result.stdout)
    artifact = torch.load(output, map_location="cpu", weights_only=True)
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    artifact_sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
    assert artifact["provenance"] == provenance
    assert response["provenance"] == provenance
    assert artifact["source_checkpoint_sha256"] == checkpoint_sha256
    assert response["source_checkpoint_sha256"] == checkpoint_sha256
    assert artifact["config_sha256"] == response["config_sha256"]
    assert len(response["config_sha256"]) == 64
    assert response["artifact_sha256"] == artifact_sha256
    assert observed["checkpoint"] == str(checkpoint.resolve())
    restored = unpack_state_dict(artifact[PACKED_STATE_KEY])
    assert set(restored) == set(TinyStudent(None).state_dict())


@pytest.mark.parametrize("failure_stage", ["serialize", "fsync", "replace"])
def test_quantize_run_failure_preserves_existing_artifact_and_cleans_temp(
    failure_stage: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tiny_quant_runtime: None,
) -> None:
    output = tmp_path / "quantized.pt"
    accepted_bytes = b"previously accepted artifact"
    output.write_bytes(accepted_bytes)
    failure_message = "save failed" if failure_stage == "serialize" else f"{failure_stage} failed"

    def fail(*_args, **_kwargs):
        raise OSError(failure_message)

    if failure_stage == "serialize":

        def fail_after_partial_write(_payload, handle, **_kwargs):
            handle.write(b"partial temporary artifact")
            raise OSError(failure_message)

        monkeypatch.setattr(torch, "save", fail_after_partial_write)
    elif failure_stage == "fsync":
        monkeypatch.setattr("forge.cli_commands.quantize.os.fsync", fail)
    else:
        monkeypatch.setattr("forge.cli_commands.quantize.os.replace", fail)

    result = _invoke_json(
        "run",
        "--method",
        "qvla",
        "--device",
        "cpu",
        "--allow-mock",
        "--output",
        str(output),
    )

    assert result.exit_code == 2
    assert result.stdout == ""
    assert json.loads(result.stderr) == {"error": failure_message}
    assert output.read_bytes() == accepted_bytes
    assert list(tmp_path.glob(f".{output.name}.*.tmp")) == []


def test_quantize_loader_restores_saved_student_architecture_before_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tiny_quant_runtime: None,
) -> None:
    monkeypatch.delenv("FORGE_ALLOW_MOCK", raising=False)
    checkpoint = tmp_path / "flagship.pt"
    torch.save(
        {
            "model_state_dict": TinyStudent(None).state_dict(),
            "student_config": {
                "action_head_type": "flow",
                "action_head_layers": 7,
                "lora_rank": 64,
                "lora_alpha": 128,
            },
            "provenance": _provenance(tmp_path, mock=False),
        },
        checkpoint,
    )

    config, _model, _provenance_data = load_student_for_quant(
        "configs/forge_nano.yaml",
        checkpoint=str(checkpoint),
        require_trained_checkpoint=True,
    )

    assert config.student.action_head_type == "flow"
    assert config.student.action_head_layers == 7
    assert config.student.lora_rank == 64
    assert config.student.lora_alpha == 128


def test_quantize_run_preserves_pruning_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tiny_quant_runtime: None,
) -> None:
    checkpoint = tmp_path / "trained.pt"
    output = tmp_path / "quantized.pt"
    provenance = _provenance(tmp_path, mock=False)
    pruning = {"removed_layers": [], "target_layers": 4, "calibration_provenance": "real"}
    torch.save(
        {
            "model_state_dict": TinyStudent(None).state_dict(),
            "pruning": pruning,
            "provenance": provenance,
        },
        checkpoint,
    )

    result = _invoke_json(
        "run",
        "--method",
        "qvla",
        "--bits",
        "4",
        "--device",
        "cpu",
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output),
    )

    assert result.exit_code == 0, result.output
    artifact = torch.load(output, map_location="cpu", weights_only=True)
    assert artifact["pruning"] == pruning


def test_quantize_run_rejects_checkpoint_below_eighty_percent_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tiny_quant_runtime: None,
) -> None:
    monkeypatch.delenv("FORGE_ALLOW_MOCK", raising=False)
    checkpoint = tmp_path / "sparse.pt"
    full_state = TinyStudent(None).state_dict()
    torch.save(
        {
            "model_state_dict": {
                "vision_encoder.weight": full_state["vision_encoder.weight"],
                "vision_encoder.bias": full_state["vision_encoder.bias"],
            },
            "provenance": _provenance(tmp_path, mock=False),
        },
        checkpoint,
    )

    result = _invoke_json(
        "run",
        "--method",
        "qvla",
        "--device",
        "cpu",
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(tmp_path / "must-not-exist.pt"),
    )

    assert result.exit_code == 2
    assert result.stdout == ""
    assert "at least 80.0% is required" in json.loads(result.stderr)["error"]
    assert not (tmp_path / "must-not-exist.pt").exists()


@pytest.mark.parametrize("opt_in", ["environment", "config"])
def test_quantize_run_combines_environment_and_config_mock_opt_in(
    opt_in: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tiny_quant_runtime: None,
) -> None:
    config_path = tmp_path / "forge.yaml"
    config_path.write_text(
        yaml.safe_dump({"student": {"allow_mock": opt_in == "config"}}),
        encoding="utf-8",
    )
    if opt_in == "environment":
        monkeypatch.setenv("FORGE_ALLOW_MOCK", "1")
    else:
        monkeypatch.delenv("FORGE_ALLOW_MOCK", raising=False)

    config, _, provenance = load_student_for_quant(
        str(config_path),
        require_trained_checkpoint=True,
    )

    assert config.student.allow_mock is True
    assert provenance["labels"] == "mock"


def test_quantize_bench_allows_untrained_model_without_writing_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tiny_quant_runtime: None,
) -> None:
    monkeypatch.delenv("FORGE_ALLOW_MOCK", raising=False)

    result = _invoke_json(
        "bench",
        "--method",
        "qvla",
        "--device",
        "cpu",
    )

    assert result.exit_code == 0, result.output
    response = json.loads(result.stdout)
    assert response["mse"] == 0.0
    assert response["provenance"]["labels"] == "mock"


def test_quantize_bench_allow_mock_propagates_to_checkpoint_verifier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tiny_quant_runtime: None,
) -> None:
    monkeypatch.delenv("FORGE_ALLOW_MOCK", raising=False)
    checkpoint = tmp_path / "mock.pt"
    provenance = _provenance(tmp_path, mock=True)
    torch.save(
        {
            "model_state_dict": TinyStudent(None).state_dict(),
            "provenance": provenance,
        },
        checkpoint,
    )

    result = _invoke_json(
        "bench",
        "--method",
        "qvla",
        "--device",
        "cpu",
        "--checkpoint",
        str(checkpoint),
        "--allow-mock",
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["provenance"] == provenance


def test_quantize_bench_invalid_device_is_strict_json_error() -> None:
    result = _invoke_json(
        "bench",
        "--method",
        "qvla",
        "--device",
        "cuda:-1",
    )

    assert result.exit_code == 2
    assert result.stdout == ""
    assert set(json.loads(result.stderr)) == {"error"}
    assert "Unsupported CUDA device" in json.loads(result.stderr)["error"]


def test_quantize_bench_runtime_failure_is_strict_json_error(
    monkeypatch: pytest.MonkeyPatch,
    tiny_quant_runtime: None,
) -> None:
    monkeypatch.setattr(
        "forge.quantize.benchmark_quantization",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("benchmark exploded")),
    )

    result = _invoke_json(
        "bench",
        "--method",
        "qvla",
        "--device",
        "cpu",
    )

    assert result.exit_code == 2
    assert result.stdout == ""
    assert json.loads(result.stderr) == {"error": "benchmark exploded"}


def test_quantize_bench_indexed_cuda_oom_falls_back_with_strict_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from contextlib import nullcontext

    moves: list[str] = []
    attempts = 0

    class Model:
        def to(self, device: str):
            moves.append(device)
            return self

    def benchmark(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("CUDA out of memory")
        return {"method": "qvla", "mse": 0.0}

    monkeypatch.setattr("forge.cli_commands.quantize.resolve_runtime_device", lambda **_kwargs: "cuda:1")
    monkeypatch.setattr(
        "forge.cli_commands.quantize.load_student_for_quant",
        lambda *_args, **_kwargs: (None, Model(), {"labels": "real"}),
    )
    monkeypatch.setattr("forge.quantize.benchmark_quantization", benchmark)
    monkeypatch.setattr(torch.cuda, "device", lambda _device: nullcontext())
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    result = _invoke_json(
        "bench",
        "--method",
        "qvla",
        "--device",
        "cuda:1",
        "--allow-cpu-fallback",
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert result.stderr == ""
    assert moves == ["cuda:1", "cpu"]
    assert payload["device"] == "cpu"
    assert payload["fallbacks"] == [
        {
            "stage": "bench",
            "from": "cuda:1",
            "to": "cpu",
            "reason": "RuntimeError: out-of-memory",
        }
    ]
