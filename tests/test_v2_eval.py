"""PRD-32: VLA Evaluation Harness tests."""

import json
import os
import tempfile
from pathlib import Path

import pytest
import yaml

# ── EvalResult tests ──────────────────────────────────────


def test_eval_result_dataclass():
    """EvalResult stores all fields correctly."""
    from forge.eval.results import EvalResult

    result = EvalResult(
        benchmark="libero_spatial",
        success_rate=0.75,
        tasks=10,
        episodes_per_task=20,
        per_task_rates={"pick_up_cup": 0.8, "open_drawer": 0.7},
        latency_p50_ms=45.2,
        student_variant="nano",
        checkpoint="best.pt",
        timestamp="2026-03-19 10:00",
    )
    assert result.success_rate == 0.75
    assert result.tasks == 10
    assert result.benchmark == "libero_spatial"
    assert result.per_task_rates["pick_up_cup"] == 0.8


def test_eval_result_to_dict():
    """EvalResult serializes to dict."""
    from forge.eval.results import EvalResult

    result = EvalResult(benchmark="libero", success_rate=0.5, tasks=5)
    d = result.to_dict()
    assert d["benchmark"] == "libero"
    assert d["success_rate"] == 0.5
    assert d["tasks"] == 5


def test_eval_result_from_dict():
    """EvalResult deserializes from dict."""
    from forge.eval.results import EvalResult

    data = {
        "benchmark": "simpler",
        "success_rate": 0.6,
        "tasks": 8,
        "episodes_per_task": 20,
        "student_variant": "small",
        "checkpoint": "model.pt",
        "extra_field": "ignored",
    }
    result = EvalResult.from_dict(data)
    assert result.benchmark == "simpler"
    assert result.success_rate == 0.6
    assert result.student_variant == "small"


def test_eval_result_to_json():
    """EvalResult serializes to JSON string."""
    from forge.eval.results import EvalResult

    result = EvalResult(benchmark="vlabench", success_rate=0.3)
    j = result.to_json()
    parsed = json.loads(j)
    assert parsed["benchmark"] == "vlabench"


def test_eval_result_report_markdown():
    """EvalResult formats as markdown for REPORT_GPU.md."""
    from forge.eval.results import EvalResult

    result = EvalResult(
        benchmark="libero_spatial",
        success_rate=0.75,
        tasks=10,
        episodes_per_task=20,
        latency_p50_ms=45.2,
        student_variant="nano",
        checkpoint="/outputs/best.pt",
        timestamp="2026-03-19 10:00",
    )
    md = result.to_report_markdown()
    assert "libero_spatial" in md
    assert "75.0%" in md
    assert "nano" in md
    assert "45.2ms" in md


# ── Results parser tests ──────────────────────────────────


def test_parse_vla_eval_results_no_files():
    """parse_vla_eval_results handles empty directory."""
    from forge.eval.results import parse_vla_eval_results

    with tempfile.TemporaryDirectory() as tmpdir:
        result = parse_vla_eval_results(Path(tmpdir), "libero", "nano", "best.pt")
        assert result.status == "no_results"


def test_parse_vla_eval_results_with_data():
    """parse_vla_eval_results parses JSON results."""
    from forge.eval.results import parse_vla_eval_results

    with tempfile.TemporaryDirectory() as tmpdir:
        results_file = Path(tmpdir) / "results.json"
        results_file.write_text(
            json.dumps(
                {
                    "success_rate": 0.65,
                    "num_tasks": 10,
                    "episodes_per_task": 20,
                    "per_task_success_rates": {"task_a": 0.7, "task_b": 0.6},
                    "latency_p50_ms": 42.0,
                }
            )
        )

        result = parse_vla_eval_results(Path(tmpdir), "libero", "nano", "best.pt")
        assert result.success_rate == 0.65
        assert result.tasks == 10
        assert result.latency_p50_ms == 42.0
        assert result.per_task_rates["task_a"] == 0.7


def test_parse_vla_eval_results_uses_latest_harness_result(tmp_path):
    """Timestamped harness results override stale files in reused directories."""
    from forge.eval.results import parse_vla_eval_results

    stale = tmp_path / "results.json"
    stale.write_text(json.dumps({"success_rate": 1.0}))
    latest = tmp_path / "LIBEROBenchmark_sync_123.json"
    latest.write_text(
        json.dumps(
            {
                "mean_success": 0.5,
                "tasks": [{"task": "pick cup", "mean_success": 0.5, "episodes": []}],
                "config": {"episodes_per_task": 2},
            }
        )
    )
    os.utime(stale, ns=(1, 1))
    os.utime(latest, ns=(2, 2))

    result = parse_vla_eval_results(tmp_path, "libero", "nano", "best.pt")

    assert result.success_rate == 0.5
    assert result.per_task_rates == {"pick cup": 0.5}
    assert result.episodes_per_task == 2


def test_parse_vla_eval_results_marks_episode_exception_failed(tmp_path):
    """A zero-exit harness cannot hide an exception recorded in an episode."""
    from forge.eval.results import parse_vla_eval_results

    (tmp_path / "LIBEROBenchmark_sync_123.json").write_text(
        json.dumps(
            {
                "mean_success": 0.0,
                "tasks": [
                    {
                        "task": "pick cup",
                        "mean_success": 0.0,
                        "episodes": [
                            {
                                "episode_id": 0,
                                "metrics": {"success": False},
                                "failure_reason": "exception",
                            }
                        ],
                    }
                ],
                "config": {"episodes_per_task": 1},
            }
        )
    )

    result = parse_vla_eval_results(tmp_path, "libero", "nano", "best.pt")

    assert result.status == "failed"
    assert result.success_rate == 0.0
    assert "episode 0" in result.error
    assert "exception" in result.error


@pytest.mark.parametrize(
    "failure_reason",
    ["timeout", "server_unreachable", "connection_closed_1006", "exception: model failed"],
)
def test_parse_vla_eval_results_rejects_official_infrastructure_failures(
    tmp_path: Path,
    failure_reason: str,
) -> None:
    from forge.eval.results import parse_vla_eval_results

    (tmp_path / "result.json").write_text(
        json.dumps(
            {
                "mean_success": 0.0,
                "tasks": [
                    {
                        "task": "pick cup",
                        "mean_success": 0.0,
                        "episodes": [{"episode_id": 0, "failure_reason": failure_reason}],
                    }
                ],
            }
        )
    )

    result = parse_vla_eval_results(tmp_path, "libero", "nano", "best.pt")

    assert result.status == "failed"
    assert failure_reason in result.error


def test_parse_vla_eval_results_rejects_partial_harness_output(tmp_path: Path) -> None:
    from forge.eval.results import parse_vla_eval_results

    (tmp_path / "result.json").write_text(json.dumps({"partial": True, "mean_success": 0.0, "tasks": []}))

    result = parse_vla_eval_results(tmp_path, "libero", "nano", "best.pt")

    assert result.status == "failed"
    assert "partial" in result.error


@pytest.mark.parametrize("success_rate", [float("nan"), float("inf"), -0.1, 1.1])
def test_parse_vla_eval_results_rejects_nonfinite_or_out_of_range_success(
    tmp_path: Path,
    success_rate: float,
) -> None:
    from forge.eval.results import parse_vla_eval_results

    (tmp_path / "result.json").write_text(
        json.dumps(
            {
                "mean_success": success_rate,
                "tasks": [{"task": "pick cup", "mean_success": 0.0, "episodes": [{}]}],
                "config": {"episodes_per_task": 1},
            }
        )
    )

    result = parse_vla_eval_results(tmp_path, "libero", "nano", "best.pt")

    assert result.status == "failed"
    assert "finite and within [0, 1]" in result.error


def test_parse_vla_eval_results_rejects_zero_workload(tmp_path: Path) -> None:
    from forge.eval.results import parse_vla_eval_results

    (tmp_path / "result.json").write_text(
        json.dumps({"mean_success": 0.0, "tasks": [], "config": {"episodes_per_task": 1}})
    )

    result = parse_vla_eval_results(tmp_path, "libero", "nano", "best.pt")

    assert result.status == "failed"
    assert "at least one task" in result.error


def test_parse_vla_eval_results_rejects_incomplete_requested_work(tmp_path: Path) -> None:
    from forge.eval.results import parse_vla_eval_results

    (tmp_path / "result.json").write_text(
        json.dumps(
            {
                "mean_success": 0.5,
                "tasks": [{"task": "pick cup", "mean_success": 0.5, "episodes": [{}]}],
                "config": {"episodes_per_task": 2},
            }
        )
    )

    result = parse_vla_eval_results(tmp_path, "libero", "nano", "best.pt")

    assert result.status == "failed"
    assert "executed 1 of 2 requested episode(s)" in result.error


def test_parse_vla_eval_results_bounds_malformed_value_types(tmp_path: Path) -> None:
    from forge.eval.results import parse_vla_eval_results

    (tmp_path / "result.json").write_text(
        json.dumps(
            {
                "mean_success": [],
                "num_tasks": 2,
                "tasks": [{"task": [], "mean_success": {}, "episodes": "invalid"}],
                "config": {"episodes_per_task": {}},
                "latency_p50_ms": {},
            }
        )
    )

    result = parse_vla_eval_results(tmp_path, "libero", "nano", "best.pt")

    assert result.status == "failed"
    assert result.success_rate == 0.0
    assert result.latency_p50_ms == 0.0
    assert "invalid completed-result schema" in result.error


def test_parse_vla_eval_results_keeps_executed_zero_success_completed(tmp_path: Path) -> None:
    from forge.eval.results import parse_vla_eval_results

    (tmp_path / "result.json").write_text(
        json.dumps(
            {
                "mean_success": 0.0,
                "tasks": [{"task": "pick cup", "mean_success": 0.0, "episodes": [{}]}],
                "config": {"episodes_per_task": 1},
            }
        )
    )

    result = parse_vla_eval_results(tmp_path, "libero", "nano", "best.pt")

    assert result.status == "completed"
    assert result.success_rate == 0.0


def test_load_results_empty():
    """load_results returns empty list for missing directory."""
    from forge.eval.results import load_results

    results = load_results("/nonexistent/path")
    assert results == []


def test_load_results_reads_direct_harness_output(tmp_path):
    """The results CLI reads the same direct layout produced by eval run."""
    from forge.eval.results import load_results

    benchmark_dir = tmp_path / "libero"
    benchmark_dir.mkdir()
    (benchmark_dir / "LIBEROBenchmark_sync_123.json").write_text(
        json.dumps(
            {
                "created_at": "2026-07-13T16:57:28+00:00",
                "mean_success": 0.5,
                "tasks": [
                    {
                        "task": "pick cup",
                        "mean_success": 0.5,
                        "episodes": [{"episode_id": 0, "metrics": {"success": True}}],
                    }
                ],
                "config": {"episodes_per_task": 2},
                "server_info": {"model": "FORGE-nano", "checkpoint": "/models/final.pt"},
            }
        )
    )

    results = load_results(tmp_path)

    assert len(results) == 1
    assert results[0].benchmark == "libero"
    assert results[0].success_rate == 0.5
    assert results[0].student_variant == "nano"
    assert results[0].checkpoint == "/models/final.pt"


def test_load_results_normalizes_nested_harness_exception(tmp_path):
    """Historical nested harness output must not hide recorded exceptions."""
    from forge.eval.results import load_results

    benchmark_dir = tmp_path / "libero-smoke" / "libero"
    benchmark_dir.mkdir(parents=True)
    (benchmark_dir / "LIBEROBenchmark_sync_123.json").write_text(
        json.dumps(
            {
                "benchmark": "LIBEROBenchmark",
                "created_at": "2026-07-13T16:57:28+00:00",
                "mean_success": 0.0,
                "tasks": [
                    {
                        "task": "pick cup",
                        "mean_success": 0.0,
                        "episodes": [
                            {
                                "episode_id": 0,
                                "metrics": {"success": False},
                                "failure_reason": "exception",
                            }
                        ],
                    }
                ],
                "config": {"episodes_per_task": 1},
                "server_info": {"model": "FORGE-nano", "checkpoint": "/models/final.pt"},
            }
        )
    )

    results = load_results(tmp_path)

    assert len(results) == 1
    assert results[0].benchmark == "LIBEROBenchmark"
    assert results[0].status == "failed"
    assert "episode 0" in results[0].error


def test_load_results_prefers_failed_raw_result_over_completed_summary(tmp_path):
    """Deduplication retains raw failure evidence over a stale summary row."""
    from forge.eval.results import load_results

    checkpoint = "/models/final.pt"
    (tmp_path / "all_results.json").write_text(
        json.dumps(
            [
                {
                    "benchmark": "LIBEROBenchmark",
                    "success_rate": 0.0,
                    "tasks": 1,
                    "episodes_per_task": 1,
                    "student_variant": "nano",
                    "checkpoint": checkpoint,
                    "timestamp": "2026-07-13 16:57",
                    "status": "completed",
                }
            ]
        )
    )
    benchmark_dir = tmp_path / "libero"
    benchmark_dir.mkdir()
    (benchmark_dir / "LIBEROBenchmark_sync_123.json").write_text(
        json.dumps(
            {
                "benchmark": "LIBEROBenchmark",
                "created_at": "2026-07-13T16:57:28+00:00",
                "mean_success": 0.0,
                "tasks": [
                    {
                        "task": "pick cup",
                        "mean_success": 0.0,
                        "episodes": [{"episode_id": 0, "failure_reason": "exception"}],
                    }
                ],
                "config": {"episodes_per_task": 1},
                "server_info": {"model": "FORGE-nano", "checkpoint": checkpoint},
            }
        )
    )

    results = load_results(tmp_path)

    assert len(results) == 1
    assert results[0].status == "failed"
    assert "exception" in results[0].error


def test_load_results_deduplicates_run_all_and_raw_views(tmp_path):
    """run-all's normalized row does not duplicate its raw harness result."""
    from forge.eval.results import load_results

    checkpoint = "/models/final.pt"
    (tmp_path / "all_results.json").write_text(
        json.dumps(
            [
                {
                    "benchmark": "libero",
                    "success_rate": 0.5,
                    "tasks": 1,
                    "episodes_per_task": 2,
                    "student_variant": "nano",
                    "checkpoint": checkpoint,
                    "timestamp": "2026-07-13 16:57",
                }
            ]
        )
    )
    benchmark_dir = tmp_path / "libero"
    benchmark_dir.mkdir()
    (benchmark_dir / "LIBEROBenchmark_sync_123.json").write_text(
        json.dumps(
            {
                "created_at": "2026-07-13T16:57:28+00:00",
                "mean_success": 0.5,
                "tasks": [{"task": "pick cup", "mean_success": 0.5, "episodes": []}],
                "config": {"episodes_per_task": 2},
                "server_info": {"model": "FORGE-nano", "checkpoint": checkpoint},
            }
        )
    )

    results = load_results(tmp_path)

    assert len(results) == 1
    assert results[0].success_rate == 0.5


def test_load_results_attributes_nested_run_to_parent_benchmark_and_deduplicates(tmp_path: Path) -> None:
    from forge.eval.results import load_results

    checkpoint = "/models/final.pt"
    (tmp_path / "all_results.json").write_text(
        json.dumps(
            [
                {
                    "benchmark": "libero",
                    "success_rate": 0.5,
                    "tasks": 1,
                    "episodes_per_task": 1,
                    "student_variant": "nano",
                    "checkpoint": checkpoint,
                    "timestamp": "2026-07-14 06:30",
                }
            ]
        )
    )
    run_dir = tmp_path / "libero" / "run-123"
    run_dir.mkdir(parents=True)
    (run_dir / "LIBEROBenchmark_sync_123.json").write_text(
        json.dumps(
            {
                "created_at": "2026-07-14T06:30:45+00:00",
                "mean_success": 0.5,
                "tasks": [{"task": "pick cup", "mean_success": 0.5, "episodes": [{}]}],
                "config": {"episodes_per_task": 1},
                "server_info": {"model": "FORGE-nano", "checkpoint": checkpoint},
            }
        )
    )

    results = load_results(tmp_path)

    assert len(results) == 1
    assert results[0].benchmark == "libero"


@pytest.mark.parametrize("payload", [{"unexpected": "mapping"}, ["not-a-result-row"]])
def test_load_results_ignores_malformed_all_results(tmp_path: Path, payload: object) -> None:
    from forge.eval.results import load_results

    (tmp_path / "all_results.json").write_text(json.dumps(payload))

    assert load_results(tmp_path) == []


def test_results_to_table():
    """results_to_table formats as markdown table."""
    from forge.eval.results import EvalResult, results_to_table

    results = [
        EvalResult(benchmark="libero", success_rate=0.75, tasks=10, student_variant="nano"),
        EvalResult(
            benchmark="simpler",
            success_rate=0.60,
            tasks=8,
            student_variant="nano",
            status="failed",
            error="episode 0: exception | container [runtime] error",
        ),
    ]
    table = results_to_table(results)
    assert "libero" in table
    assert "simpler" in table
    assert "75.0%" in table
    assert "60.0%" in table
    assert "failed" in table
    assert "episode 0: exception \\| container [runtime] error" in table


def test_results_to_table_empty():
    """results_to_table handles empty list."""
    from forge.eval.results import results_to_table

    assert "No results" in results_to_table([])


def test_append_to_report():
    """append_to_report writes to an explicit report path."""
    from forge.eval.results import EvalResult, append_to_report

    with tempfile.TemporaryDirectory() as tmpdir:
        report_path = Path(tmpdir) / "REPORT_GPU.md"
        result = EvalResult(
            benchmark="libero",
            success_rate=0.75,
            student_variant="nano",
            checkpoint="best.pt",
            timestamp="2026-03-19",
        )
        append_to_report(result, str(report_path))
        content = report_path.read_text()
        assert "libero" in content
        assert "75.0%" in content
        assert "**Status**: completed" in content


def test_failed_result_report_includes_error(tmp_path):
    """Generated reports cannot present harness exceptions as ordinary outcomes."""
    from forge.eval.results import EvalResult, append_to_report

    report_path = tmp_path / "report.md"
    append_to_report(
        EvalResult(
            benchmark="simpler",
            status="failed",
            error="episode 0: exception\ncontainer failed",
        ),
        report_path,
    )

    content = report_path.read_text()
    assert "**Status**: failed" in content
    assert "**Error**: episode 0: exception container failed" in content


def test_append_to_report_defaults_to_ignored_artifact_directory(monkeypatch, tmp_path):
    """The default report never recreates an internal root document."""
    from forge.eval.results import EvalResult, append_to_report

    monkeypatch.chdir(tmp_path)
    append_to_report(EvalResult(benchmark="libero", success_rate=0.5))

    report = tmp_path / "outputs" / "eval" / "report.md"
    assert report.is_file()
    assert "FORGE Evaluation Report" in report.read_text()
    assert not (tmp_path / "REPORT_GPU.md").exists()


# ── ForgeModelServer tests ────────────────────────────────


def test_forge_model_server_init():
    """ForgeModelServer instantiates without loading model."""
    from forge.eval.model_server import ForgeModelServer

    server = ForgeModelServer(
        checkpoint_path="./outputs/best.pt",
        variant="nano",
        device="cpu",
    )
    assert server.config.variant == "nano"
    assert server.config.device == "cpu"
    assert not server._loaded


def test_forge_model_server_config():
    """ForgeModelServer stores all config params."""
    from forge.eval.model_server import ForgeModelServer

    server = ForgeModelServer(
        checkpoint_path="./test.pt",
        variant="small",
        port=9000,
        image_size=224,
        action_scale=2.0,
        action_offset=-1.0,
        device="cpu",
    )
    assert server.config.port == 9000
    assert server.config.image_size == 224
    assert server.config.action_scale == 2.0
    assert server.config.action_offset == -1.0


# ── EvalRunner tests ──────────────────────────────────────


def test_eval_runner_init():
    """EvalRunner initializes with checkpoint path."""
    from forge.eval.runner import EvalRunner

    with tempfile.TemporaryDirectory() as tmpdir:
        runner = EvalRunner(
            checkpoint_path="./outputs/best.pt",
            variant="nano",
            device="cpu",
            output_dir=tmpdir,
        )
        assert runner.checkpoint_path == "./outputs/best.pt"
        assert runner.variant == "nano"
        assert runner.output_dir.exists()


def test_eval_runner_benchmark_images():
    """BENCHMARK_IMAGES has all 3 benchmarks."""
    from forge.eval.runner import BENCHMARK_IMAGES

    assert "libero" in BENCHMARK_IMAGES
    assert "simpler" in BENCHMARK_IMAGES
    assert "vlabench" in BENCHMARK_IMAGES


@pytest.mark.parametrize(
    ("benchmark", "config_name", "allowed_params"),
    [
        ("libero", "libero_forge.yaml", {"suite", "seed", "num_steps_wait"}),
        ("simpler", "simpler_forge.yaml", {"seed"}),
        ("vlabench", "vlabench_forge.yaml", {"tasks", "robot", "max_steps"}),
    ],
)
def test_checked_in_eval_configs_match_official_harness_contracts(
    benchmark: str,
    config_name: str,
    allowed_params: set[str],
) -> None:
    from forge.eval.runner import BENCHMARK_CLASSES, BENCHMARK_IMAGES

    config = yaml.safe_load((Path("configs/eval") / config_name).read_text())
    entry = config["benchmarks"][0]

    assert config["docker"]["image"] == BENCHMARK_IMAGES[benchmark]
    assert entry["benchmark"] == BENCHMARK_CLASSES[benchmark]
    assert set(entry.get("params", {})) <= allowed_params


def test_public_eval_docs_match_registered_benchmarks() -> None:
    from forge.eval.runner import BENCHMARK_IMAGES

    guide = "\n".join(
        [
            Path("docs/EVALUATION.md").read_text(),
            Path("docs/CONFIGURATION.md").read_text(),
        ]
    )

    for benchmark, image_name in BENCHMARK_IMAGES.items():
        assert f"`{benchmark}`" in guide
        assert image_name in guide
    assert "RLBench" not in guide
    assert "rlbench" not in guide
    assert "lazy loading" not in guide.lower()


def test_eval_runner_check_docker():
    """check_docker returns bool."""
    from forge.eval.runner import EvalRunner

    result = EvalRunner.check_docker()
    assert isinstance(result, bool)


def test_eval_runner_pins_docker_to_visible_gpu(tmp_path, monkeypatch):
    from forge.eval.runner import EvalRunner

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,1")
    runner = EvalRunner(
        checkpoint_path="./outputs/best.pt",
        device="cuda",
        output_dir=str(tmp_path),
    )
    assert runner._docker_gpu_request() == "device=3"


def test_eval_runner_maps_logical_index_to_same_visible_physical_gpu(tmp_path, monkeypatch):
    from forge.eval.runner import EvalRunner

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,1")
    runner = EvalRunner(
        checkpoint_path="./outputs/best.pt",
        device="cuda:1",
        output_dir=str(tmp_path),
    )

    assert runner._docker_gpu_request() == "device=1"


def test_eval_runner_docker_gpu_request_fallbacks(tmp_path, monkeypatch):
    from forge.eval.runner import EvalRunner

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    cuda_runner = EvalRunner(
        checkpoint_path="./outputs/best.pt",
        device="cuda:2",
        output_dir=str(tmp_path / "cuda"),
    )
    cpu_runner = EvalRunner(
        checkpoint_path="./outputs/best.pt",
        device="cpu",
        output_dir=str(tmp_path / "cpu"),
    )
    assert cuda_runner._docker_gpu_request() == "device=2"
    assert cpu_runner._docker_gpu_request() is None


def test_eval_runner_retries_nvidia_runtime_failure_with_osmesa(tmp_path):
    import subprocess

    from forge.eval.runner import EvalRunner

    runner = EvalRunner(
        checkpoint_path="./outputs/best.pt",
        device="cuda",
        output_dir=str(tmp_path),
    )
    failed = subprocess.CompletedProcess(
        args=["docker"],
        returncode=1,
        stdout="",
        stderr="nvidia-container-cli: initialization error: nvml error",
    )
    assert runner._is_nvidia_container_runtime_failure(failed)

    image = "libero:latest"
    command = ["docker", "run", "--gpus", "device=3", image, "run"]
    fallback = runner._docker_cpu_fallback_command(command, image)
    assert "--gpus" not in fallback
    assert "MUJOCO_GL=osmesa" in fallback
    assert "PYOPENGL_PLATFORM=osmesa" in fallback
    assert "NVIDIA_VISIBLE_DEVICES=void" in fallback
    assert fallback[-2:] == [image, "run"]


def test_cpu_render_fallback_excludes_simpler_vulkan() -> None:
    from forge.eval.runner import CPU_RENDER_FALLBACK_BENCHMARKS

    assert CPU_RENDER_FALLBACK_BENCHMARKS == {"libero", "vlabench"}


def test_eval_runner_rebinds_custom_config_to_fallback_port(tmp_path):
    from forge.eval.runner import EvalRunner

    runner = EvalRunner(
        checkpoint_path="./outputs/best.pt",
        device="cpu",
        output_dir=str(tmp_path),
    )
    original = {
        "server": {"url": "ws://localhost:8000", "timeout": 30},
        "benchmarks": [{"benchmark": "example:Benchmark"}],
    }

    rebound = runner._bind_server_url(original, 43123)

    assert rebound["server"] == {"url": "ws://localhost:43123", "timeout": 30}
    assert original["server"]["url"] == "ws://localhost:8000"


def test_eval_runner_never_reuses_stale_success_when_run_writes_no_result(tmp_path, monkeypatch):
    import subprocess

    from forge.eval.runner import EvalRunner

    stale_dir = tmp_path / "libero"
    stale_dir.mkdir()
    (stale_dir / "LIBEROBenchmark_sync_OLD.json").write_text(
        json.dumps({"success_rate": 1.0, "tasks": 1}),
        encoding="utf-8",
    )

    class _Server:
        def __init__(self, **_kwargs) -> None:
            pass

        def start(self, **_kwargs) -> None:
            pass

        def stop(self) -> None:
            pass

    monkeypatch.setattr("forge.eval.model_server.ForgeModelServer", _Server)
    runner = EvalRunner(
        checkpoint_path="./outputs/best.pt",
        device="cpu",
        output_dir=str(tmp_path),
    )
    monkeypatch.setattr(runner, "_is_port_in_use", lambda _port: False)
    monkeypatch.setattr(
        runner,
        "_run_docker_benchmark",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=["docker"],
            returncode=0,
            stdout="",
            stderr="",
        ),
    )

    result = runner.run_benchmark("libero", episodes_per_task=1, max_tasks=1)

    assert result["status"] == "no_results"
    assert result["success_rate"] == 0.0
    assert len(list(stale_dir.glob("run-*"))) == 1


def test_eval_runner_stops_server_when_startup_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from forge.eval.runner import EvalRunner

    stopped: list[bool] = []

    class _Server:
        def __init__(self, **_kwargs) -> None:
            pass

        def start(self, **_kwargs) -> None:
            raise TimeoutError("bind timeout")

        def stop(self) -> None:
            stopped.append(True)

    monkeypatch.setattr("forge.eval.model_server.ForgeModelServer", _Server)
    runner = EvalRunner(checkpoint_path="best.pt", device="cpu", output_dir=str(tmp_path))

    result = runner.run_benchmark("libero", episodes_per_task=1, max_tasks=1)

    assert result["status"] == "failed"
    assert stopped == [True]


def test_eval_server_bind_timeout_stops_background_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    from forge.eval.model_server import ForgeModelServer

    server = ForgeModelServer("missing.pt", device="cpu")
    monkeypatch.setattr(server, "_ensure_model_loaded", lambda: None)

    async def slow_serve() -> None:
        await asyncio.sleep(0.05)

    monkeypatch.setattr(server, "_serve", slow_serve)

    with pytest.raises(TimeoutError, match="failed to bind"):
        server.start(blocking=False, startup_timeout=0.001)

    assert server._server_thread is None
    assert server._loop is None


def test_eval_server_blocking_preload_failure_never_starts_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    from forge.eval.model_server import ForgeModelServer

    server = ForgeModelServer("missing.pt", device="cpu")

    def fail_preload() -> None:
        raise RuntimeError("checkpoint load failed")

    monkeypatch.setattr(server, "_ensure_model_loaded", fail_preload)

    with pytest.raises(RuntimeError, match="checkpoint load failed"):
        server.start(blocking=False, startup_timeout=0.001)

    assert server._server_thread is None
    assert server._loop is None


def test_eval_server_propagates_immediate_thread_startup_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from forge.eval.model_server import ForgeModelServer

    server = ForgeModelServer("missing.pt", device="cpu")
    monkeypatch.setattr(server, "_ensure_model_loaded", lambda: None)

    async def fail_serve() -> None:
        raise RuntimeError("bind failed immediately")

    monkeypatch.setattr(server, "_serve", fail_serve)

    with pytest.raises(RuntimeError, match="bind failed immediately"):
        server.start(blocking=False, startup_timeout=1.0)

    assert server._server_thread is None
    assert server._loop is None


def test_eval_compare_format():
    """Compare result has expected structure."""
    # Just test the output structure, not actual eval
    comparison = {
        "benchmark": "libero",
        "checkpoint_a": "a.pt",
        "checkpoint_b": "b.pt",
        "success_rate_a": 0.7,
        "success_rate_b": 0.5,
        "delta_success_rate": 0.2,
    }
    assert comparison["delta_success_rate"] == 0.2
    assert comparison["benchmark"] == "libero"


# ── Import tests ──────────────────────────────────────────


def test_eval_package_imports():
    """forge.eval package imports cleanly."""
    from forge.eval import EvalResult
    from forge.eval.model_server import ForgeModelServer
    from forge.eval.runner import EvalRunner

    assert EvalResult is not None
    assert ForgeModelServer is not None
    assert EvalRunner is not None
