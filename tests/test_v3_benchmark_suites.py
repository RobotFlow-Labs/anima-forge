"""Packaged benchmark-suite and CLI contracts."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from forge.benchmark.suite_runner import (
    SUITES,
    _contains_failure,
    resolve_suite,
    run_all_suites,
    run_suite,
    suite_catalog,
    summarize_existing_suites,
    verify_suite_summary_artifacts,
)
from forge.cli_v2 import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def stable_benchmark_git_sha(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("forge.benchmark.execution.current_git_sha", lambda: "benchmark-test-sha")


class _FakeProcess:
    def __init__(self, lines: str, return_code: int = 0) -> None:
        self.stdout = io.StringIO(lines)
        self._return_code = return_code

    def wait(self) -> int:
        return self._return_code


def test_suite_catalog_and_aliases() -> None:
    catalog = suite_catalog()
    assert len(catalog) == 15
    assert catalog[0]["slug"] == "vision-encoder"
    assert resolve_suite("1") is SUITES[0]
    assert resolve_suite("bench_15_auto_hp_400") is SUITES[-1]


@pytest.mark.parametrize("field", ["coverage_passed", "quality_passed"])
def test_benchmark_failure_detection_consumes_explicit_acceptance_fields(field: str) -> None:
    assert _contains_failure({field: False}) is True
    assert _contains_failure({field: True}) is False


@pytest.mark.parametrize("status", ["blocked", "cancelled", "incomplete", "partial", "timeout"])
def test_benchmark_failure_detection_rejects_nonterminal_success_statuses(status: str) -> None:
    assert _contains_failure({"status": status}) is True


def test_benchmark_directory_is_results_only() -> None:
    benchmark_dir = Path(__file__).parents[1] / "benchmarks"
    assert (benchmark_dir / ".gitkeep").is_file()
    unexpected = [
        path
        for path in benchmark_dir.rglob("*")
        if path.is_file() and path.name != ".gitkeep" and path.suffix != ".json"
    ]
    assert unexpected == []

    gitignore = (Path(__file__).parents[1] / ".gitignore").read_text(encoding="utf-8")
    assert "/benchmarks/**" in gitignore
    assert "!/benchmarks/.gitkeep" in gitignore


def test_packaged_benchmarks_do_not_generate_random_observations_or_actions() -> None:
    suite_dir = Path(__file__).parents[1] / "src/forge/benchmark/suites"
    violations = []
    for path in sorted(suite_dir.glob("bench_*.py")):
        source = path.read_text(encoding="utf-8")
        for forbidden in ("torch.randn", "np.random"):
            if forbidden in source:
                violations.append(f"{path.name}: {forbidden}")

    assert violations == []


def test_export_suite_uses_production_runtime_paths() -> None:
    source = (Path(__file__).parents[1] / "src/forge/benchmark/suites/bench_14_export_tensorrt.py").read_text(
        encoding="utf-8"
    )

    assert "dynamic_axes" not in source
    assert "pycuda" not in source
    assert "from forge.export.onnx_export import _onnx_artifact_files, export_onnx" in source
    assert "from forge.export.tensorrt_export import benchmark_tensorrt_runtime" in source


def test_export_suite_clears_stale_runtime_artifacts(tmp_path: Path) -> None:
    from forge.benchmark.suites.bench_14_export_tensorrt import clear_previous_export_artifacts

    onnx_path = tmp_path / "forge_nano_flow.onnx"
    external_data_path = tmp_path / "forge_nano_flow.onnx.data"
    trt_path = tmp_path / "forge_nano_flow.engine"
    unrelated_path = tmp_path / "keep.json"
    for path in (onnx_path, external_data_path, trt_path, unrelated_path):
        path.write_text("stale", encoding="utf-8")

    clear_previous_export_artifacts(onnx_path, trt_path)

    assert not onnx_path.exists()
    assert not external_data_path.exists()
    assert not trt_path.exists()
    assert unrelated_path.is_file()


def test_export_suite_reports_decimal_artifact_family_sizes(tmp_path: Path) -> None:
    from forge.benchmark.suites.bench_14_export_tensorrt import artifact_size_metrics

    graph = tmp_path / "forge.onnx"
    sidecar = tmp_path / "forge.onnx.data"
    graph.write_bytes(b"g" * 10)
    sidecar.write_bytes(b"d" * 20)

    metrics = artifact_size_metrics(graph, [graph, sidecar, graph])

    assert metrics == {
        "graph_size_mb": 10 / 1e6,
        "artifact_size_mb": 30 / 1e6,
        "artifact_files": ["forge.onnx", "forge.onnx.data"],
        "artifacts_sha256": {
            "forge.onnx": hashlib.sha256(b"g" * 10).hexdigest(),
            "forge.onnx.data": hashlib.sha256(b"d" * 20).hexdigest(),
        },
    }


def test_export_suite_refuses_tensorrt_after_failed_onnx(tmp_path: Path, monkeypatch) -> None:
    from forge.benchmark.suites import bench_14_export_tensorrt as export_suite

    monkeypatch.setattr(
        export_suite,
        "try_tensorrt_export",
        lambda *_args, **_kwargs: pytest.fail("must not convert a stale ONNX graph"),
    )

    result = export_suite.export_tensorrt_after_onnx(
        {"status": "failed", "error": "fresh export broke"},
        tmp_path / "stale.onnx",
        tmp_path / "stale.engine",
    )

    assert result == {
        "status": "failed",
        "reason": "TensorRT export requires a successful fresh ONNX export",
    }


def test_400_trial_suite_requires_four_visible_gpus() -> None:
    source = (Path(__file__).parents[1] / "src/forge/benchmark/suites/bench_15_auto_hp_400.py").read_text(
        encoding="utf-8"
    )

    assert "if n_gpus < 4:" in source
    assert "gpu_assignments = list(range(4))" in source


def test_multi_teacher_suite_uses_process_isolated_fleet() -> None:
    source = (Path(__file__).parents[1] / "src/forge/benchmark/suites/bench_10_multi_teacher.py").read_text(
        encoding="utf-8"
    )

    assert "build_isolated_fleet_report" in source
    assert "build_fleet_report(" not in source


def test_real_data_benchmark_scores_free_running_actions() -> None:
    import torch

    from forge.benchmark.suites.bench_13_real_data_training import inference_action_mse

    predicted = torch.tensor([[[1.0, 4.0], [99.0, 99.0]]])
    target = torch.tensor([[1.0, 2.0]])

    assert inference_action_mse(predicted, target) == 2.0


def test_real_data_peak_memory_is_isolated_on_selected_cuda_device(monkeypatch) -> None:
    from forge.benchmark.suites import bench_13_real_data_training as suite

    calls: list[tuple[str, str]] = []
    peaks = iter((3 * 1024**3, 2 * 1024**3))
    monkeypatch.setattr(suite.torch.cuda, "synchronize", lambda device: calls.append(("synchronize", str(device))))
    monkeypatch.setattr(
        suite.torch.cuda,
        "reset_peak_memory_stats",
        lambda device: calls.append(("reset", str(device))),
    )
    monkeypatch.setattr(suite.torch.cuda, "max_memory_allocated", lambda device: next(peaks))

    suite.reset_configuration_peak_memory("cuda:2")
    first_peak = suite.configuration_peak_memory_gb("cuda:2")
    suite.reset_configuration_peak_memory("cuda:2")
    second_peak = suite.configuration_peak_memory_gb("cuda:2")

    assert first_peak == 3.0
    assert second_peak == 2.0
    assert calls == [
        ("synchronize", "cuda:2"),
        ("reset", "cuda:2"),
        ("synchronize", "cuda:2"),
        ("synchronize", "cuda:2"),
        ("reset", "cuda:2"),
        ("synchronize", "cuda:2"),
    ]


def test_fixed_action_loss_forks_rng_on_selected_cuda_device(monkeypatch) -> None:
    import contextlib

    import torch

    from forge.benchmark.suites import real_data

    observed_devices: list[list[int]] = []

    class Student:
        training = True

        def eval(self):
            self.training = False

        def train(self, mode):
            self.training = mode

        def __call__(self, _images, *, gt_actions):
            return {"loss": gt_actions.sum()}

    monkeypatch.setattr(
        real_data.torch.random,
        "fork_rng",
        lambda *, devices: observed_devices.append(devices) or contextlib.nullcontext(),
    )
    monkeypatch.setattr(
        real_data,
        "real_batch",
        lambda *_args, **_kwargs: (torch.ones(1), torch.ones(1)),
    )

    loss = real_data.fixed_action_loss(Student(), object(), "cuda:2", n_batches=1)

    assert loss == 1.0
    assert observed_devices == [[2]]


def test_export_suite_preserves_export_hashes_when_validation_fails(tmp_path: Path, monkeypatch) -> None:
    import torch

    from forge.benchmark.suites import bench_14_export_tensorrt as suite

    class Student:
        def __init__(self) -> None:
            self.moves: list[str] = []

        def cpu(self):
            self.moves.append("cpu")
            return self

        def to(self, device):
            self.moves.append(str(device))
            return self

    artifact_hashes = {"model.onnx": "a" * 64, "model.onnx.data": "b" * 64}

    def fail_validation(*_args, **_kwargs):
        raise RuntimeError("bad graph")

    monkeypatch.setattr(
        suite,
        "export_onnx_model",
        lambda *_args, **_kwargs: {
            "graph_size_mb": 1.0,
            "artifact_size_mb": 2.0,
            "artifact_files": list(artifact_hashes),
            "artifacts_sha256": artifact_hashes,
        },
    )
    monkeypatch.setattr(suite, "validate_onnx", fail_validation)
    monkeypatch.setattr(
        suite,
        "benchmark_onnx",
        lambda *_args, **_kwargs: {"status": "success", "provider": "CUDA", "fps": 20.0, "mean_ms": 50.0},
    )
    student = Student()

    stages = suite.run_onnx_stages(student, tmp_path / "model.onnx", torch.zeros(1), device="cuda:2")
    aggregate = suite.required_stage_summary(
        {
            **stages,
            "tensorrt_export": {"status": "success"},
            "tensorrt_runtime": {"status": "success"},
        }
    )

    assert stages["onnx_export"]["status"] == "success"
    assert stages["onnx_export"]["artifacts_sha256"] == artifact_hashes
    assert stages["onnx_validation"] == {"status": "failed", "error": "bad graph"}
    assert stages["onnx_runtime"]["status"] == "success"
    assert aggregate["status"] == "failed"
    assert aggregate["failed_stages"] == ["onnx_validation"]
    assert student.moves == ["cpu", "cuda:2"]


def test_export_suite_attributes_runtime_failure_and_fails_required_aggregate(tmp_path: Path, monkeypatch) -> None:
    import torch

    from forge.benchmark.suites import bench_14_export_tensorrt as suite

    class Student:
        def cpu(self):
            return self

        def to(self, _device):
            return self

    artifact_hashes = {"model.onnx": "c" * 64}

    def fail_runtime(*_args, **_kwargs):
        raise RuntimeError("ORT failed")

    monkeypatch.setattr(
        suite,
        "export_onnx_model",
        lambda *_args, **_kwargs: {
            "graph_size_mb": 1.0,
            "artifact_size_mb": 1.0,
            "artifact_files": list(artifact_hashes),
            "artifacts_sha256": artifact_hashes,
        },
    )
    monkeypatch.setattr(suite, "validate_onnx", lambda *_args, **_kwargs: {"status": "passed", "max_diff": 0.0})
    monkeypatch.setattr(suite, "benchmark_onnx", fail_runtime)

    stages = suite.run_onnx_stages(Student(), tmp_path / "model.onnx", torch.zeros(1), device="cuda")
    result = {
        **stages,
        "tensorrt_export": {"status": "success"},
        "tensorrt_runtime": {"status": "success"},
    }
    aggregate = suite.required_stage_summary(result)

    assert stages["onnx_export"]["artifacts_sha256"] == artifact_hashes
    assert stages["onnx_validation"]["status"] == "passed"
    assert stages["onnx_runtime"] == {"status": "failed", "error": "ORT failed"}
    assert aggregate["status"] == "failed"
    assert aggregate["failed_stages"] == ["onnx_runtime"]


def test_training_benchmarks_evaluate_a_fixed_real_sample_set(monkeypatch) -> None:
    import torch

    from forge.benchmark.suites import real_data

    class Student(torch.nn.Module):
        def forward(self, images, *, gt_actions):
            del images
            return {"loss": gt_actions.mean()}

    monkeypatch.setattr(
        real_data,
        "real_batch",
        lambda _dataset, _batch_size, _device, *, start=0, **_kwargs: (
            torch.zeros(1, 3, 2, 2),
            torch.tensor([[float(start + 1)]]),
        ),
    )
    student = Student().train()

    loss = real_data.fixed_action_loss(student, object(), "cpu", n_batches=3)

    assert loss == 2.0
    assert student.training is True


def test_benchmark_rng_reset_is_repeatable() -> None:
    import random

    import numpy as np
    import torch

    from forge.benchmark.suites.real_data import reset_benchmark_rng

    reset_benchmark_rng(123)
    first = (random.random(), float(np.random.random()), float(torch.rand(())))
    reset_benchmark_rng(123)
    second = (random.random(), float(np.random.random()), float(torch.rand(())))

    assert first == second


def test_training_benchmark_suites_reset_and_report_rng_seed() -> None:
    suite_dir = Path(__file__).parents[1] / "src/forge/benchmark/suites"
    names = (
        "bench_03_training.py",
        "bench_08_e2e_pipeline.py",
        "bench_09_multi_gpu.py",
        "bench_10_multi_teacher.py",
        "bench_11_student_variants.py",
        "bench_12_full_pipeline_combos.py",
        "bench_13_real_data_training.py",
        "bench_15_auto_hp_400.py",
    )
    for name in names:
        source = (suite_dir / name).read_text(encoding="utf-8")
        assert "reset_benchmark_rng(" in source, name
        assert "BENCHMARK_SEED" in source, name
        assert '"random_seed' in source, name


def test_training_benchmark_raw_loss_fields_are_explicit() -> None:
    suite_dir = Path(__file__).parents[1] / "src/forge/benchmark/suites"
    for name in ("bench_03_training.py", "bench_10_multi_teacher.py", "bench_11_student_variants.py"):
        source = (suite_dir / name).read_text(encoding="utf-8")
        assert '"loss_curve":' not in source
        assert '"loss_curves":' not in source


def test_fixed_real_evaluation_reuses_stochastic_noise(monkeypatch) -> None:
    import torch

    from forge.benchmark.suites import real_data

    class StochasticStudent(torch.nn.Module):
        def forward(self, images, *, gt_actions):
            del images
            return {"loss": gt_actions.mean() + torch.rand(())}

    monkeypatch.setattr(
        real_data,
        "real_batch",
        lambda *_args, **_kwargs: (torch.zeros(1, 3, 2, 2), torch.ones(1, 1)),
    )
    student = StochasticStudent().train()

    first = real_data.fixed_action_loss(student, object(), "cpu", n_batches=3)
    second = real_data.fixed_action_loss(student, object(), "cpu", n_batches=3)

    assert first == second
    assert student.training is True


def test_real_data_training_reports_fixed_evaluation_separately() -> None:
    from forge.benchmark.suites.bench_13_real_data_training import training_metrics

    metrics = training_metrics(
        [9.0, 1.0],
        total_steps=2,
        train_time_s=1.0,
        evaluation_loss_before=4.0,
        evaluation_loss_after=3.0,
    )

    assert metrics["loss_metric"] == "fixed-real-evaluation-mean"
    assert metrics["loss_reduction_pct"] == 25.0
    assert metrics["training_loss_start"] == 9.0
    assert metrics["training_loss_end"] == 1.0


def test_multi_teacher_benchmark_evaluates_fixed_router_inputs(monkeypatch) -> None:
    import torch

    from forge.benchmark.suites import bench_10_multi_teacher as suite

    class Student(torch.nn.Module):
        def forward(self, images, *, gt_actions):
            batch = images.shape[0]
            return {
                "actions": torch.rand(batch, 7),
                "vision_features": torch.zeros(batch, 2, 3),
                "loss": gt_actions.mean(),
            }

    class RouterLoss(torch.nn.Module):
        def forward(self, actions, _teacher_actions, ground_truth, _features, *_confidence):
            return {"total": ground_truth.mean() + actions.mean()}

    monkeypatch.setattr(
        suite,
        "real_batch",
        lambda _dataset, batch_size, _device, *, start=0, **_kwargs: (
            torch.zeros(batch_size, 3, 2, 2),
            torch.full((batch_size, 7), float(start + 1)),
        ),
    )
    monkeypatch.setattr(
        suite,
        "_teacher_tensors",
        lambda _records, *, batch_size, **_kwargs: (
            [torch.zeros(batch_size, 7)],
            torch.zeros(batch_size, 1, 7),
        ),
    )
    student = Student().train()
    router_loss = RouterLoss().train()

    standard = suite.evaluate_fixed_router_loss(student, router_loss, object(), [{}], n_batches=3)
    standard_repeat = suite.evaluate_fixed_router_loss(student, router_loss, object(), [{}], n_batches=3)
    universal = suite.evaluate_fixed_router_loss(
        student,
        router_loss,
        object(),
        [{}],
        universal=True,
        n_batches=3,
    )

    assert standard > 5.0
    assert standard_repeat == standard
    assert universal == standard
    assert student.training is True
    assert router_loss.training is True


def test_run_suite_records_fresh_json_artifact(tmp_path: Path, monkeypatch) -> None:
    def fake_popen(_command, *, env, **_kwargs):
        artifact = Path(env["FORGE_BENCHMARK_RESULTS_DIR"]) / SUITES[0].artifact
        artifact.write_text(
            json.dumps({"benchmark": "vision_encoder", "data_provenance": {"kind": "real"}}),
            encoding="utf-8",
        )
        return _FakeProcess("BENCH 01: DONE\n")

    monkeypatch.setattr("forge.benchmark.suite_runner.subprocess.Popen", fake_popen)
    progress = io.StringIO()
    result = run_suite("01", results_dir=tmp_path, progress=progress)

    assert result["status"] == "completed"
    assert result["artifact"] == SUITES[0].artifact
    assert result["artifact_sha256"] == hashlib.sha256((tmp_path / SUITES[0].artifact).read_bytes()).hexdigest()
    assert progress.getvalue() == "BENCH 01: DONE\n"
    payload = json.loads((tmp_path / SUITES[0].artifact).read_text(encoding="utf-8"))
    assert payload["execution"]["schema"] == "forge.benchmark-execution.v1"
    assert payload["execution"]["suite"] == "vision-encoder"
    assert payload["execution"]["requested_device"] == "auto"
    assert payload["execution"]["git_sha"]
    assert payload["execution"]["forge_version"]
    assert payload["execution"]["torch_version"]


def test_suite_summary_verifier_rejects_missing_or_tampered_completed_artifact(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fake_popen(_command, *, env, **_kwargs):
        artifact = Path(env["FORGE_BENCHMARK_RESULTS_DIR"]) / SUITES[0].artifact
        artifact.write_text(
            json.dumps({"benchmark": "vision_encoder", "data_provenance": {"kind": "real"}}),
            encoding="utf-8",
        )
        return _FakeProcess("BENCH 01: DONE\n")

    monkeypatch.setattr("forge.benchmark.suite_runner.subprocess.Popen", fake_popen)
    record = run_suite("01", results_dir=tmp_path)
    summary = {"suites": [record]}
    artifact = tmp_path / SUITES[0].artifact

    verify_suite_summary_artifacts(summary, results_dir=tmp_path)

    artifact.write_text('{"tampered": true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_suite_summary_artifacts(summary, results_dir=tmp_path)

    artifact.unlink()
    with pytest.raises(ValueError, match="artifact is missing"):
        verify_suite_summary_artifacts(summary, results_dir=tmp_path)


def test_suite_summary_verifier_rejects_cross_suite_artifact_lineage(tmp_path: Path, monkeypatch) -> None:
    def fake_popen(_command, *, env, **_kwargs):
        artifact = Path(env["FORGE_BENCHMARK_RESULTS_DIR"]) / SUITES[0].artifact
        artifact.write_text(
            json.dumps({"benchmark": "vision_encoder", "data_provenance": {"kind": "real"}}),
            encoding="utf-8",
        )
        return _FakeProcess("BENCH 01: DONE\n")

    monkeypatch.setattr("forge.benchmark.suite_runner.subprocess.Popen", fake_popen)
    record = run_suite("01", results_dir=tmp_path)
    source = tmp_path / SUITES[0].artifact
    target = tmp_path / SUITES[1].artifact
    target.write_bytes(source.read_bytes())
    record.update(
        suite=SUITES[1].slug,
        number=SUITES[1].number,
        artifact=SUITES[1].artifact,
        artifact_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
    )

    with pytest.raises(ValueError, match="execution lineage does not match"):
        verify_suite_summary_artifacts({"suites": [record]}, results_dir=tmp_path)


@pytest.mark.parametrize("artifact_path", ["../outside.json", "/absolute.json", "nested/../artifact.json"])
def test_suite_summary_verifier_rejects_noncanonical_artifact_path(
    tmp_path: Path,
    artifact_path: str,
) -> None:
    outside = tmp_path.parent / "outside.json"
    outside.write_text("{}\n", encoding="utf-8")
    summary = {
        "suites": [
            {
                "suite": "vision-encoder",
                "status": "completed",
                "artifact": artifact_path,
                "artifact_sha256": hashlib.sha256(outside.read_bytes()).hexdigest(),
            }
        ]
    }

    with pytest.raises(ValueError, match="not canonical"):
        verify_suite_summary_artifacts(summary, results_dir=tmp_path)


def test_run_all_suites_persists_and_returns_identical_content_bound_summary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from forge.benchmark import suite_runner as suite_runner_module
    from forge.json_artifacts import write_json_artifact as durable_write_json_artifact

    published: list[str] = []

    def recording_writer(path, payload) -> None:
        published.append(Path(path).name)
        durable_write_json_artifact(path, payload)

    def fake_popen(command, *, env, **_kwargs):
        module = command[-1].rsplit(".", 1)[-1]
        spec = next(item for item in SUITES if item.module == module)
        artifact = Path(env["FORGE_BENCHMARK_RESULTS_DIR"]) / spec.artifact
        artifact.write_text(
            json.dumps({"benchmark": spec.slug, "data_provenance": {"kind": "real"}}),
            encoding="utf-8",
        )
        return _FakeProcess(f"BENCH {spec.number}: DONE\n")

    monkeypatch.setattr(suite_runner_module, "write_json_artifact", recording_writer)
    monkeypatch.setattr(suite_runner_module.subprocess, "Popen", fake_popen)

    summary = run_all_suites(results_dir=tmp_path)
    persisted = json.loads((tmp_path / "suite_summary.json").read_text(encoding="utf-8"))

    assert summary == persisted
    assert summary["artifact"] == "suite_summary.json"
    assert summary["status"] == "completed"
    assert len(summary["suites"]) == len(SUITES)
    assert all(record["artifact"] == SUITES[index].artifact for index, record in enumerate(summary["suites"]))
    assert all(record["artifact_sha256"] for record in summary["suites"])
    verify_suite_summary_artifacts(summary, results_dir=tmp_path)
    assert published[-1] == "suite_summary.json"


def test_run_all_suites_rejects_changed_artifact_before_publishing_summary(tmp_path: Path, monkeypatch) -> None:
    from forge.benchmark import suite_runner as suite_runner_module

    spec = SUITES[0]

    def fake_run_suite(_spec, *, results_dir, **_kwargs):
        artifact = Path(results_dir) / spec.artifact
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text("{}\n", encoding="utf-8")
        return {
            "suite": spec.slug,
            "number": spec.number,
            "status": "completed",
            "artifact": spec.artifact,
            "artifact_sha256": "0" * 64,
        }

    monkeypatch.setattr(suite_runner_module, "SUITES", (spec,))
    monkeypatch.setattr(suite_runner_module, "run_suite", fake_run_suite)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        run_all_suites(results_dir=tmp_path)

    assert not (tmp_path / "suite_summary.json").exists()


def _write_completed_suite_artifact(root: Path, spec, *, sha_suffix: str = "") -> None:
    payload = {
        "benchmark": spec.slug,
        "data_provenance": {"kind": "real"},
        "execution": {
            "schema": "forge.benchmark-execution.v1",
            "command": "suite",
            "requested_device": "cuda",
            "git_sha": f"benchmark-test-sha{sha_suffix}",
            "forge_version": "3.0.0",
            "torch_version": "2.10.0",
            "python_version": "3.12.12",
            "suite": spec.slug,
            "suite_number": spec.number,
        },
    }
    (root / spec.artifact).write_text(json.dumps(payload), encoding="utf-8")


def test_summarize_existing_suites_publishes_content_bound_summary(tmp_path: Path) -> None:
    for spec in SUITES:
        _write_completed_suite_artifact(tmp_path, spec)

    summary = summarize_existing_suites(results_dir=tmp_path)
    persisted = json.loads((tmp_path / "suite_summary.json").read_text(encoding="utf-8"))

    assert summary == persisted
    assert summary["source"] == "existing-artifacts"
    assert summary["status"] == "completed"
    assert summary["completed"] == len(SUITES)
    assert summary["failed"] == 0
    assert all(record["artifact_sha256"] for record in summary["suites"])
    verify_suite_summary_artifacts(summary, results_dir=tmp_path)


def test_summarize_existing_suites_rejects_cross_suite_lineage(tmp_path: Path) -> None:
    for spec in SUITES:
        _write_completed_suite_artifact(tmp_path, spec)
    payload = json.loads((tmp_path / SUITES[1].artifact).read_text(encoding="utf-8"))
    payload["execution"]["suite_number"] = SUITES[0].number
    (tmp_path / SUITES[1].artifact).write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="execution lineage"):
        summarize_existing_suites(results_dir=tmp_path)

    assert not (tmp_path / "suite_summary.json").exists()


def test_benchmark_aggregate_cli_emits_summary_json(tmp_path: Path) -> None:
    for spec in SUITES:
        _write_completed_suite_artifact(tmp_path, spec)

    result = runner.invoke(app, ["benchmark", "aggregate", "--results-dir", str(tmp_path), "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "completed"
    assert payload["artifact"] == "suite_summary.json"
    assert (tmp_path / "suite_summary.json").is_file()


def test_run_suite_captures_git_sha_before_process_launch(tmp_path: Path, monkeypatch) -> None:
    observed: list[str] = []

    def changing_git_sha() -> str:
        value = "sha-at-launch" if not observed else "sha-after-launch"
        observed.append(value)
        return value

    def fake_popen(_command, *, env, **_kwargs):
        artifact = Path(env["FORGE_BENCHMARK_RESULTS_DIR"]) / SUITES[0].artifact
        artifact.write_text(
            json.dumps({"benchmark": "vision_encoder", "data_provenance": {"kind": "real"}}),
            encoding="utf-8",
        )
        return _FakeProcess("BENCH 01: DONE\n")

    monkeypatch.setattr("forge.benchmark.execution.current_git_sha", changing_git_sha)
    monkeypatch.setattr("forge.benchmark.suite_runner.subprocess.Popen", fake_popen)

    result = run_suite("01", results_dir=tmp_path)
    payload = json.loads((tmp_path / SUITES[0].artifact).read_text(encoding="utf-8"))

    assert result["status"] == "completed"
    assert payload["execution"]["git_sha"] == "sha-at-launch"
    assert observed == ["sha-at-launch"]


def test_run_suite_propagates_real_data_directory(tmp_path: Path, monkeypatch) -> None:
    data_dir = tmp_path / "real-data"

    def fake_popen(_command, *, env, **_kwargs):
        assert env["FORGE_BENCHMARK_DATA_DIR"] == str(data_dir)
        artifact = Path(env["FORGE_BENCHMARK_RESULTS_DIR"]) / SUITES[12].artifact
        artifact.write_text(json.dumps({"data_provenance": {"kind": "real"}}), encoding="utf-8")
        return _FakeProcess("BENCH 13: DONE\n")

    monkeypatch.setattr("forge.benchmark.suite_runner.subprocess.Popen", fake_popen)
    result = run_suite("13", results_dir=tmp_path, data_dir=data_dir)

    assert result["status"] == "completed"


def test_run_suite_reports_skip_without_reusing_stale_artifact(tmp_path: Path, monkeypatch) -> None:
    artifact = tmp_path / SUITES[0].artifact
    artifact.write_text(json.dumps({"old": True}), encoding="utf-8")
    monkeypatch.setattr(
        "forge.benchmark.suite_runner.subprocess.Popen",
        lambda *_args, **_kwargs: _FakeProcess("SKIP: No CUDA device\n"),
    )

    result = run_suite("vision-encoder", results_dir=tmp_path)

    assert result["status"] == "skipped"
    assert result["artifact"] is None
    assert result["reason"] == "SKIP: No CUDA device"
    assert json.loads(artifact.read_text(encoding="utf-8")) == {"old": True}


def test_run_suite_setup_failure_does_not_quarantine_existing_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / SUITES[0].artifact
    original = '{"accepted": true}\n'
    artifact.write_text(original, encoding="utf-8")

    with pytest.raises(ValueError, match="device must be one of"):
        run_suite("vision-encoder", results_dir=tmp_path, device="cuda:invalid")

    assert artifact.read_text(encoding="utf-8") == original
    assert sorted(tmp_path.iterdir()) == [artifact]


def test_run_suite_cannot_accept_a_touched_stale_artifact(tmp_path: Path, monkeypatch) -> None:
    artifact = tmp_path / SUITES[0].artifact
    original = json.dumps({"data_provenance": {"kind": "real"}})
    artifact.write_text(original, encoding="utf-8")

    def fake_popen(_command, *, env, **_kwargs):
        assert not (Path(env["FORGE_BENCHMARK_RESULTS_DIR"]) / SUITES[0].artifact).exists()
        return _FakeProcess("BENCH 01: DONE\n")

    monkeypatch.setattr("forge.benchmark.suite_runner.subprocess.Popen", fake_popen)

    result = run_suite("vision-encoder", results_dir=tmp_path)

    assert result["status"] == "failed"
    assert result["artifact"] is None
    assert "without producing" in result["error"]
    assert artifact.read_text(encoding="utf-8") == original


def test_run_suite_rejects_non_object_json_without_crashing(tmp_path: Path, monkeypatch) -> None:
    def fake_popen(_command, *, env, **_kwargs):
        artifact = Path(env["FORGE_BENCHMARK_RESULTS_DIR"]) / SUITES[0].artifact
        artifact.write_text("[]\n", encoding="utf-8")
        return _FakeProcess("BENCH 01: DONE\n")

    monkeypatch.setattr("forge.benchmark.suite_runner.subprocess.Popen", fake_popen)

    result = run_suite("vision-encoder", results_dir=tmp_path)

    assert result["status"] == "failed"
    assert "must contain a JSON object" in result["error"]


def test_run_suite_does_not_treat_incidental_skip_text_as_skip(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "forge.benchmark.suite_runner.subprocess.Popen",
        lambda *_args, **_kwargs: _FakeProcess("diagnostic mentioned SKIP: but the suite produced nothing\n"),
    )

    result = run_suite("vision-encoder", results_dir=tmp_path)

    assert result["status"] == "failed"
    assert "without producing" in result["error"]


def test_every_benchmark_suite_uses_atomic_strict_json_writer() -> None:
    suites_dir = Path("src/forge/benchmark/suites")
    for source_path in suites_dir.glob("bench_*.py"):
        source = source_path.read_text(encoding="utf-8")
        assert "write_json_artifact(" in source, source_path
        assert "json.dump(" not in source, source_path


def test_every_benchmark_training_step_clips_gradients() -> None:
    suites_dir = Path("src/forge/benchmark/suites")
    for source_path in suites_dir.glob("bench_*.py"):
        source = source_path.read_text(encoding="utf-8")
        assert ".backward(" not in source, source_path
    assert ".backward(" not in Path("src/forge/auto_hyperparam.py").read_text(encoding="utf-8")


def test_atomic_json_writer_preserves_existing_artifact_on_non_finite_value(tmp_path: Path) -> None:
    from forge.benchmark.artifacts import write_json_artifact

    artifact = tmp_path / "benchmark.json"
    original = '{"status": "accepted"}\n'
    artifact.write_text(original, encoding="utf-8")

    with pytest.raises(ValueError, match="Out of range float values"):
        write_json_artifact(artifact, {"latency_ms": float("nan")})

    assert artifact.read_text(encoding="utf-8") == original
    assert list(tmp_path.iterdir()) == [artifact]


def test_atomic_json_writer_publishes_nothing_on_non_finite_value(tmp_path: Path) -> None:
    from forge.benchmark.artifacts import write_json_artifact

    artifact = tmp_path / "benchmark.json"

    with pytest.raises(ValueError, match="Out of range float values"):
        write_json_artifact(artifact, {"loss": float("inf")})

    assert not artifact.exists()
    assert list(tmp_path.iterdir()) == []


def test_multi_teacher_evidence_uses_benchmark_data_parent(tmp_path: Path, monkeypatch) -> None:
    from forge.benchmark.suites import bench_10_multi_teacher

    dataset_root = tmp_path / "datasets"
    benchmark_data = dataset_root / "lerobot--pusht"
    benchmark_data.mkdir(parents=True)
    (dataset_root / "lerobot--aloha_sim_transfer_cube_human").mkdir()
    stale_teacher_root = tmp_path / "stale-teacher-root"
    captured: dict[str, object] = {}
    monkeypatch.setenv("FORGE_BENCHMARK_DATA_DIR", str(benchmark_data))
    monkeypatch.setenv("FORGE_TEACHER_DATASET_ROOT", str(stale_teacher_root))

    def fake_report(**kwargs):
        captured.update(kwargs)
        return {
            "all_real": True,
            "teachers_verified": len(bench_10_multi_teacher.TEACHERS),
            "results": [],
        }

    monkeypatch.setattr("forge.teacher_fleet.build_isolated_fleet_report", fake_report)

    bench_10_multi_teacher.collect_real_teacher_evidence()

    assert captured["dataset_dir"] == dataset_root


def test_multi_teacher_evidence_falls_back_to_teacher_dataset_root(tmp_path: Path, monkeypatch) -> None:
    from forge.benchmark.suites import bench_10_multi_teacher

    dataset_root = tmp_path / "teacher-datasets"
    (dataset_root / "lerobot--pusht").mkdir(parents=True)
    (dataset_root / "lerobot--aloha_sim_transfer_cube_human").mkdir()
    captured: dict[str, object] = {}
    monkeypatch.delenv("FORGE_BENCHMARK_DATA_DIR", raising=False)
    monkeypatch.setenv("FORGE_TEACHER_DATASET_ROOT", str(dataset_root))

    def fake_report(**kwargs):
        captured.update(kwargs)
        return {
            "all_real": True,
            "teachers_verified": len(bench_10_multi_teacher.TEACHERS),
            "results": [],
        }

    monkeypatch.setattr("forge.teacher_fleet.build_isolated_fleet_report", fake_report)

    bench_10_multi_teacher.collect_real_teacher_evidence()

    assert captured["dataset_dir"] == dataset_root.resolve()


def test_run_suite_propagates_nested_benchmark_failure(tmp_path: Path, monkeypatch) -> None:
    def fake_popen(_command, *, env, **_kwargs):
        artifact = Path(env["FORGE_BENCHMARK_RESULTS_DIR"]) / SUITES[13].artifact
        artifact.write_text(
            json.dumps(
                {
                    "data_provenance": {"kind": "real"},
                    "onnx_export": {"status": "failed", "error": "export broke"},
                }
            ),
            encoding="utf-8",
        )
        return _FakeProcess("BENCH 14: DONE\n")

    monkeypatch.setattr("forge.benchmark.suite_runner.subprocess.Popen", fake_popen)
    result = run_suite("14", results_dir=tmp_path)

    assert result["status"] == "failed"
    assert "failed checks" in result["error"]


def test_run_suite_rejects_nested_optional_skip(tmp_path: Path, monkeypatch) -> None:
    def fake_popen(_command, *, env, **_kwargs):
        artifact = Path(env["FORGE_BENCHMARK_RESULTS_DIR"]) / SUITES[13].artifact
        artifact.write_text(
            json.dumps(
                {
                    "data_provenance": {"kind": "real"},
                    "tensorrt_export": {"status": "skipped", "reason": "missing runtime"},
                }
            ),
            encoding="utf-8",
        )
        return _FakeProcess("BENCH 14: DONE\n")

    monkeypatch.setattr("forge.benchmark.suite_runner.subprocess.Popen", fake_popen)
    result = run_suite("14", results_dir=tmp_path)

    assert result["status"] == "failed"
    assert "failed checks" in result["error"]


def test_run_suite_rejects_false_real_provenance(tmp_path: Path, monkeypatch) -> None:
    def fake_popen(_command, *, env, **_kwargs):
        artifact = Path(env["FORGE_BENCHMARK_RESULTS_DIR"]) / SUITES[9].artifact
        artifact.write_text(json.dumps({"all_teachers_real": False}), encoding="utf-8")
        return _FakeProcess("BENCH 10: DONE\n")

    monkeypatch.setattr("forge.benchmark.suite_runner.subprocess.Popen", fake_popen)
    result = run_suite("10", results_dir=tmp_path)

    assert result["status"] == "failed"


def test_run_suite_requires_real_data_provenance(tmp_path: Path, monkeypatch) -> None:
    def fake_popen(_command, *, env, **_kwargs):
        artifact = Path(env["FORGE_BENCHMARK_RESULTS_DIR"]) / SUITES[12].artifact
        artifact.write_text(json.dumps({"data_provenance": {"kind": "mock"}}), encoding="utf-8")
        return _FakeProcess("BENCH 13: DONE\n")

    monkeypatch.setattr("forge.benchmark.suite_runner.subprocess.Popen", fake_popen)
    result = run_suite("13", results_dir=tmp_path)

    assert result["status"] == "failed"
    assert "real input-data provenance" in result["error"]


def test_run_suite_rejects_non_finite_metrics(tmp_path: Path, monkeypatch) -> None:
    def fake_popen(_command, *, env, **_kwargs):
        artifact = Path(env["FORGE_BENCHMARK_RESULTS_DIR"]) / SUITES[0].artifact
        artifact.write_text(
            json.dumps({"data_provenance": {"kind": "real"}, "latency_ms": float("nan")}),
            encoding="utf-8",
        )
        return _FakeProcess("BENCH 01: DONE\n")

    monkeypatch.setattr("forge.benchmark.suite_runner.subprocess.Popen", fake_popen)
    result = run_suite("01", results_dir=tmp_path)

    assert result["status"] == "failed"
    assert "strict JSON" in result["error"]


def test_benchmark_list_cli_emits_clean_json() -> None:
    result = runner.invoke(app, ["benchmark", "list", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert len(payload) == 15
    assert result.stderr == ""


def test_benchmark_suite_cli_returns_nonzero_for_failed_suite(monkeypatch) -> None:
    monkeypatch.setattr(
        "forge.benchmark.suite_runner.run_suite",
        lambda *_args, **_kwargs: {
            "number": "01",
            "suite": "vision-encoder",
            "status": "failed",
            "artifact": None,
        },
    )

    result = runner.invoke(app, ["benchmark", "suite", "01", "--json"])

    assert result.exit_code == 2
    assert json.loads(result.stdout)["status"] == "failed"
    assert result.stderr == ""


def test_benchmark_suite_cli_returns_nonzero_for_required_skip(monkeypatch) -> None:
    monkeypatch.setattr(
        "forge.benchmark.suite_runner.run_suite",
        lambda *_args, **_kwargs: {
            "number": "15",
            "suite": "auto-hp-400",
            "status": "skipped",
            "artifact": None,
        },
    )

    result = runner.invoke(app, ["benchmark", "suite", "15", "--json"])

    assert result.exit_code == 2
    assert json.loads(result.stdout)["status"] == "skipped"
    assert result.stderr == ""
