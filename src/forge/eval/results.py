"""Results parser for vla-evaluation-harness output.

Parses JSON results from benchmark containers into EvalResult dataclass.
Supports generated artifact reports and JSON export.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _strict_integer(value: Any) -> int:
    """Parse an integer without silently truncating floats or accepting booleans."""
    if isinstance(value, bool):
        raise ValueError
    parsed = int(value)
    if isinstance(value, float) and (not math.isfinite(value) or value != parsed):
        raise ValueError
    return parsed


@dataclass
class EvalResult:
    """Result from a single benchmark evaluation."""

    benchmark: str = ""
    success_rate: float = 0.0
    tasks: int = 0
    episodes_per_task: int = 0
    per_task_rates: dict[str, float] = field(default_factory=dict)
    latency_p50_ms: float = 0.0
    student_variant: str = ""
    checkpoint: str = ""
    timestamp: str = ""
    status: str = "completed"
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, data: dict) -> EvalResult:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    def to_report_markdown(self) -> str:
        """Format as markdown for a generated evaluation report."""
        lines = [
            f"### [{self.timestamp}] VLA Eval: {self.benchmark}",
            f"- **Status**: {self.status}",
            f"- **Config**: FORGE-{self.student_variant}, checkpoint={Path(self.checkpoint).name}",
            f"- **Result**: success_rate={self.success_rate:.1%}, tasks={self.tasks}, "
            f"episodes/task={self.episodes_per_task}",
            f"- **Latency**: p50={self.latency_p50_ms:.1f}ms",
        ]
        if self.error:
            error = " ".join(self.error.splitlines())
            lines.append(f"- **Error**: {error}")
        if self.per_task_rates:
            lines.append("- **Per-task rates**:")
            for task, rate in self.per_task_rates.items():
                lines.append(f"  - {task}: {rate:.1%}")
        return "\n".join(lines)


def _result_from_data(
    data: dict,
    *,
    benchmark: str,
    variant: str,
    checkpoint: str,
    timestamp: str,
) -> EvalResult:
    """Normalize one already-decoded vla-eval result object."""
    aggregate_keys = ("success_rate", "overall_success_rate", "mean_success")
    aggregate_present = any(key in data for key in aggregate_keys)
    success_rate_raw = data.get(
        "success_rate",
        data.get("overall_success_rate", data.get("mean_success", 0.0)),
    )
    schema_failures: list[str] = []
    try:
        if isinstance(success_rate_raw, bool):
            raise ValueError
        success_rate = float(success_rate_raw)
    except (OverflowError, TypeError, ValueError):
        success_rate = 0.0
        schema_failures.append("aggregate success rate is not numeric")
    if not aggregate_present:
        schema_failures.append("aggregate success rate is missing")
    elif not math.isfinite(success_rate) or not 0.0 <= success_rate <= 1.0:
        schema_failures.append("aggregate success rate must be finite and within [0, 1]")

    per_task = data.get("per_task_success_rates", data.get("per_task_rates", {}))
    if not isinstance(per_task, dict):
        per_task = {}
        schema_failures.append("per-task success rates must be a mapping")

    # tasks can be a list of task objects (vla-eval format) or an int
    task_results = data.get("tasks")
    task_list = task_results if isinstance(task_results, list) else None
    tasks_raw = data.get("num_tasks", len(task_list) if task_list is not None else task_results)
    if tasks_raw is None:
        tasks_raw = len(per_task)
    if task_list is not None:
        # vla-eval format: tasks is a list of task result objects. Recent
        # harness versions call the aggregate metric ``mean_success``.
        tasks = tasks_raw
        if not per_task:
            derived_per_task: dict[str, object] = {}
            for index, task in enumerate(task_list):
                if not isinstance(task, dict):
                    schema_failures.append(f"task result {index} must be a mapping")
                    continue
                task_name = str(task.get("task", f"task_{index}"))
                if "success_rate" in task:
                    derived_per_task[task_name] = task["success_rate"]
                elif "mean_success" in task:
                    derived_per_task[task_name] = task["mean_success"]
                else:
                    schema_failures.append(f"task result {task_name} is missing a success rate")
            per_task = derived_per_task
    else:
        tasks = tasks_raw

    try:
        task_count = _strict_integer(tasks)
    except (OverflowError, TypeError, ValueError):
        task_count = 0
        schema_failures.append("task count is not an integer")
    if task_count < 1:
        schema_failures.append("completed evaluation must contain at least one task")
    if task_list is not None and task_count != len(task_list):
        schema_failures.append(f"result contains {len(task_list)} of {task_count} requested task(s)")

    # episodes_per_task: try config block too
    episodes = data.get("episodes_per_task", data.get("num_episodes_per_task", 0))
    config = data.get("config", {})
    if episodes == 0 and isinstance(config, dict):
        episodes = config.get("episodes_per_task", 0)
    if not episodes and task_list is not None:
        episode_counts = [
            len(task.get("episodes", []))
            for task in task_list
            if isinstance(task, dict) and isinstance(task.get("episodes", []), list)
        ]
        if episode_counts and len(set(episode_counts)) == 1:
            episodes = episode_counts[0]
    try:
        episode_count = _strict_integer(episodes)
    except (OverflowError, TypeError, ValueError):
        episode_count = 0
        schema_failures.append("episodes-per-task is not an integer")
    if episode_count < 1:
        schema_failures.append("completed evaluation must contain at least one episode per task")

    latency_raw = data.get("latency_p50_ms", data.get("avg_latency_ms", 0.0))
    try:
        if isinstance(latency_raw, bool):
            raise ValueError
        latency = float(latency_raw)
    except (OverflowError, TypeError, ValueError):
        latency = 0.0
        schema_failures.append("latency is not numeric")
    if not math.isfinite(latency) or latency < 0.0:
        schema_failures.append("latency must be finite and non-negative")

    normalized_per_task: dict[str, float] = {}
    for task_name, rate_raw in per_task.items():
        try:
            if isinstance(rate_raw, bool):
                raise ValueError
            rate = float(rate_raw)
        except (OverflowError, TypeError, ValueError):
            schema_failures.append(f"per-task success rate for {task_name!s} is not numeric")
            continue
        if not math.isfinite(rate) or not 0.0 <= rate <= 1.0:
            schema_failures.append(f"per-task success rate for {task_name!s} must be finite and within [0, 1]")
            continue
        normalized_per_task[str(task_name)] = rate

    episode_failures: list[str] = []
    if task_list is not None:
        for task_index, task in enumerate(task_list):
            if not isinstance(task, dict):
                continue
            task_name = str(task.get("task", "unknown task"))
            task_episodes = task.get("episodes", [])
            if not isinstance(task_episodes, list):
                schema_failures.append(f"episodes for {task_name} must be a list")
                continue
            if len(task_episodes) != episode_count:
                schema_failures.append(
                    f"task {task_name} executed {len(task_episodes)} of {episode_count} requested episode(s)"
                )
            for episode_index, episode in enumerate(task_episodes):
                if not isinstance(episode, dict):
                    schema_failures.append(f"episode {episode_index} for {task_name} must be a mapping")
                    continue
                reason = str(episode.get("failure_reason") or "").strip()
                if reason:
                    episode_id = episode.get("episode_id", "?")
                    episode_failures.append(f"{task_name} (episode {episode_id}): {reason}")

    status = "completed"
    error = ""
    declared_status = str(data.get("status") or "").strip().lower()
    partial = data.get("partial") is True
    if partial or declared_status in {"failed", "error", "timeout", "partial"} or episode_failures or schema_failures:
        status = "failed"
        reasons: list[str] = []
        if partial:
            reasons.append("harness marked the result partial")
        if declared_status in {"failed", "error", "timeout", "partial"}:
            reasons.append(f"harness status={declared_status}")
        if episode_failures:
            details = "; ".join(episode_failures[:5])
            remaining = len(episode_failures) - 5
            suffix = f"; and {remaining} more" if remaining > 0 else ""
            reasons.append(f"{len(episode_failures)} episode failure(s): {details}{suffix}")
        if schema_failures:
            reasons.append("invalid completed-result schema: " + "; ".join(dict.fromkeys(schema_failures)))
        error = "; ".join(reasons)

    return EvalResult(
        benchmark=benchmark,
        success_rate=success_rate if math.isfinite(success_rate) else 0.0,
        tasks=task_count,
        episodes_per_task=episode_count,
        per_task_rates=normalized_per_task,
        latency_p50_ms=latency if math.isfinite(latency) else 0.0,
        student_variant=variant,
        checkpoint=checkpoint,
        timestamp=timestamp,
        status=status,
        error=error,
    )


def parse_vla_eval_results(
    results_dir: Path,
    benchmark: str,
    variant: str,
    checkpoint: str,
) -> EvalResult:
    """Parse the newest vla-eval JSON output into an ``EvalResult``."""
    timestamp = time.strftime("%Y-%m-%d %H:%M")
    json_files = list(results_dir.glob("*.json"))
    if not json_files:
        return EvalResult(
            benchmark=benchmark,
            student_variant=variant,
            checkpoint=checkpoint,
            timestamp=timestamp,
            status="no_results",
            error="No JSON result files found",
        )

    # Harnesses commonly emit timestamped files. Reusing an output directory
    # must never make a new run parse a stale result merely because it has a
    # canonical filename.
    results_file = max(json_files, key=lambda path: (path.stat().st_mtime_ns, path.name))
    try:
        data = json.loads(results_file.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        return EvalResult(
            benchmark=benchmark,
            student_variant=variant,
            checkpoint=checkpoint,
            timestamp=timestamp,
            status="parse_error",
            error=str(exc),
        )
    if not isinstance(data, dict):
        return EvalResult(
            benchmark=benchmark,
            student_variant=variant,
            checkpoint=checkpoint,
            timestamp=timestamp,
            status="parse_error",
            error=f"Expected a JSON object in {results_file.name}",
        )
    return _result_from_data(
        data,
        benchmark=benchmark,
        variant=variant,
        checkpoint=checkpoint,
        timestamp=timestamp,
    )


def load_results(output_dir: str | Path = "./outputs/eval") -> list[EvalResult]:
    """Load all eval results from output directory."""
    output_dir = Path(output_dir)
    results: list[EvalResult] = []

    if not output_dir.exists():
        return results

    # Check for all_results.json
    all_results = output_dir / "all_results.json"
    if all_results.exists():
        try:
            data = json.loads(all_results.read_text())
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        results.append(EvalResult.from_dict(item))
        except (json.JSONDecodeError, OSError):
            pass

    # Scan both the current harness layout (JSON directly under each benchmark
    # directory) and the historical nested normalized-result layout.
    for bench_dir in output_dir.iterdir():
        if not bench_dir.is_dir():
            continue
        for json_file in sorted(bench_dir.glob("*.json")):
            try:
                data = json.loads(json_file.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            if not isinstance(data, dict):
                continue
            result = _loaded_result_from_data(data, benchmark_hint=bench_dir.name)
            if result is not None:
                results.append(result)
        for run_dir in sorted(bench_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            for json_file in run_dir.glob("*.json"):
                try:
                    data = json.loads(json_file.read_text())
                except (json.JSONDecodeError, OSError):
                    continue
                if not isinstance(data, dict):
                    continue
                result = _loaded_result_from_data(data, benchmark_hint=bench_dir.name)
                if result is not None:
                    results.append(result)

    # ``run-all`` writes normalized rows to all_results.json in addition to the
    # raw per-benchmark harness files. Treat those as two views of one run, not
    # two evaluation results.
    unique_results: list[EvalResult] = []
    result_indexes: dict[tuple[str, str, str, str], int] = {}
    for result in results:
        timestamp_minute = result.timestamp.replace("T", " ")[:16]
        identity = (
            _canonical_benchmark_identity(result.benchmark),
            result.student_variant,
            result.checkpoint,
            timestamp_minute,
        )
        existing_index = result_indexes.get(identity)
        if existing_index is not None:
            existing = unique_results[existing_index]
            if existing.status == "completed" and result.status != "completed":
                unique_results[existing_index] = result
            continue
        result_indexes[identity] = len(unique_results)
        unique_results.append(result)
    return unique_results


def _canonical_benchmark_identity(benchmark: str) -> str:
    """Normalize maintained harness class names for result deduplication only."""
    normalized = benchmark.strip().lower().replace("_", "").replace("-", "")
    for name in ("libero", "simpler", "vlabench"):
        if name in normalized:
            return name
    return normalized


def _loaded_result_from_data(data: dict, *, benchmark_hint: str) -> EvalResult | None:
    """Normalize either raw harness output or a persisted result row."""
    if isinstance(data.get("tasks"), list):
        server_info = data.get("server_info", {})
        if not isinstance(server_info, dict):
            server_info = {}
        model_name = str(server_info.get("model", ""))
        return _result_from_data(
            data,
            benchmark=str(data.get("benchmark") or benchmark_hint),
            variant=model_name.removeprefix("FORGE-"),
            checkpoint=str(server_info.get("checkpoint", "")),
            timestamp=str(data.get("created_at", "")),
        )
    if "success_rate" in data or "benchmark" in data:
        return EvalResult.from_dict(data)
    return None


def append_to_report(result: EvalResult, report_path: str | Path = "outputs/eval/report.md") -> None:
    """Append an evaluation result to a generated artifact report."""
    report = Path(report_path)
    report.parent.mkdir(parents=True, exist_ok=True)
    entry = "\n\n" + result.to_report_markdown()

    if report.exists():
        with open(report, "a") as f:
            f.write(entry)
    else:
        with open(report, "w") as f:
            f.write("# FORGE Evaluation Report\n\n## Experiment Log\n" + entry)

    logger.info(f"Appended eval result to {report}")


def results_to_table(results: list[EvalResult]) -> str:
    """Format results as a markdown comparison table."""
    if not results:
        return "No results to display."

    lines = [
        "| Benchmark | Status | Success Rate | Tasks | Episodes | Latency (p50) | Variant | Checkpoint | Error |",
        "|-----------|--------|--------------|-------|----------|---------------|---------|------------|-------|",
    ]
    for r in results:
        ckpt_name = Path(r.checkpoint).name if r.checkpoint else "N/A"
        error = " ".join(r.error.splitlines()).replace("|", "\\|") if r.error else "—"
        lines.append(
            f"| {r.benchmark} | {r.status} | {r.success_rate:.1%} | {r.tasks} | "
            f"{r.episodes_per_task} | {r.latency_p50_ms:.1f}ms | "
            f"{r.student_variant} | {ckpt_name} | {error} |"
        )
    return "\n".join(lines)
