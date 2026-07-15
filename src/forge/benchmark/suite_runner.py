"""Run the packaged real-world benchmark suites in isolated processes."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, TextIO

from forge.benchmark.artifacts import write_json_artifact
from forge.benchmark.execution import benchmark_execution


@dataclass(frozen=True, slots=True)
class SuiteSpec:
    """One packaged benchmark suite."""

    number: str
    slug: str
    module: str
    artifact: str
    description: str


SUITES = (
    SuiteSpec(
        "01", "vision-encoder", "bench_01_vision_encoder", "bench_01_vision_encoder.json", "SigLIP latency and memory"
    ),
    SuiteSpec(
        "02",
        "student-build",
        "bench_02_student_build",
        "bench_02_student_build.json",
        "student construction and inference",
    ),
    SuiteSpec("03", "training", "bench_03_training", "bench_03_training.json", "knowledge-distillation training loop"),
    SuiteSpec("04", "pruning", "bench_04_pruning", "bench_04_pruning.json", "chunk-aware pruning"),
    SuiteSpec("05", "quantization", "bench_05_quantization", "bench_05_quantization.json", "chunk-aware quantization"),
    SuiteSpec("06", "autosense", "bench_06_autosense", "bench_06_autosense.json", "local model detection"),
    SuiteSpec(
        "07",
        "cross-embodiment",
        "bench_07_cross_embodiment",
        "bench_07_cross_embodiment.json",
        "cross-embodiment mappings",
    ),
    SuiteSpec(
        "08",
        "e2e-pipeline",
        "bench_08_e2e_pipeline",
        "bench_08_e2e_pipeline.json",
        "end-to-end build/train/compress flow",
    ),
    SuiteSpec("09", "multi-gpu", "bench_09_multi_gpu", "bench_09_multi_gpu.json", "multi-GPU training and inference"),
    SuiteSpec(
        "10", "multi-teacher", "bench_10_multi_teacher", "bench_10_multi_teacher.json", "multi-teacher distillation"
    ),
    SuiteSpec(
        "11",
        "student-variants",
        "bench_11_student_variants",
        "bench_11_student_variants.json",
        "student architecture variants",
    ),
    SuiteSpec(
        "12",
        "pipeline-combos",
        "bench_12_full_pipeline_combos",
        "bench_12_full_pipeline_combos.json",
        "full pipeline combinations",
    ),
    SuiteSpec(
        "13",
        "real-data-training",
        "bench_13_real_data_training",
        "bench_13_real_data_training.json",
        "PushT real-data training",
    ),
    SuiteSpec(
        "14", "export-tensorrt", "bench_14_export_tensorrt", "bench_14_export_tensorrt.json", "ONNX and TensorRT export"
    ),
    SuiteSpec(
        "15", "auto-hp-400", "bench_15_auto_hp_400", "bench_15_auto_hp_400.json", "400-trial hyperparameter search"
    ),
)

REAL_DATA_SUITE_NUMBERS = frozenset(spec.number for spec in SUITES if spec.number != "06")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _execution_provenance(spec: SuiteSpec, *, requested_device: str) -> dict[str, str]:
    return benchmark_execution(
        command="suite",
        requested_device=requested_device,
        suite=spec.slug,
        suite_number=spec.number,
    )


def _write_enriched_artifact(path: Path, payload: dict[str, Any], execution: dict[str, str]) -> None:
    enriched = {**payload, "execution": execution}
    write_json_artifact(path, enriched)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json_object(path: Path, *, context: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{context} is not strict JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{context} must contain a JSON object")
    return payload


def _resolve_canonical_artifact(results_dir: Path, artifact_path: object) -> tuple[str, Path]:
    if not isinstance(artifact_path, str) or not artifact_path:
        raise ValueError("Completed suite artifact path must be a nonempty relative string")
    if "\\" in artifact_path or "\x00" in artifact_path:
        raise ValueError(f"Suite artifact path is not canonical: {artifact_path!r}")

    relative = PurePosixPath(artifact_path)
    if (
        relative.is_absolute()
        or relative.as_posix() != artifact_path
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
        or relative.parts[0].endswith(":")
    ):
        raise ValueError(f"Suite artifact path is not canonical: {artifact_path!r}")

    root = results_dir.expanduser().resolve()
    candidate = root.joinpath(*relative.parts)
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError(f"Suite artifact is missing: {artifact_path}") from exc
    try:
        resolved_relative = resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"Suite artifact escapes the results directory: {artifact_path}") from exc
    if resolved_relative != artifact_path or not resolved.is_file():
        raise ValueError(f"Suite artifact path is not canonical: {artifact_path!r}")
    return resolved_relative, resolved


def verify_suite_summary_artifacts(summary: dict[str, Any], *, results_dir: str | Path) -> None:
    """Reject completed suite records whose content-bound artifacts are unavailable or changed."""
    records = summary.get("suites")
    if not isinstance(records, list):
        raise ValueError("Benchmark suite summary must contain a suites list")

    root = Path(results_dir)
    seen_paths: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Benchmark suite record {index} must be a mapping")
        if record.get("status") != "completed":
            continue

        suite_name = str(record.get("suite") or record.get("number") or index)
        canonical, artifact = _resolve_canonical_artifact(root, record.get("artifact"))
        if canonical in seen_paths:
            raise ValueError(f"Completed suites reuse artifact path: {canonical}")
        seen_paths.add(canonical)

        expected_sha256 = record.get("artifact_sha256")
        if not isinstance(expected_sha256, str) or _SHA256_PATTERN.fullmatch(expected_sha256) is None:
            raise ValueError(f"Completed suite {suite_name} has no canonical artifact SHA-256")
        actual_sha256 = _sha256_file(artifact)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"Completed suite {suite_name} artifact SHA-256 mismatch: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )

        matching_specs = [
            spec for spec in SUITES if record.get("suite") == spec.slug and record.get("number") == spec.number
        ]
        if len(matching_specs) != 1:
            raise ValueError(f"Completed suite {suite_name} does not identify one canonical suite")
        spec = matching_specs[0]
        if canonical != spec.artifact:
            raise ValueError(f"Completed suite {suite_name} artifact does not match canonical output {spec.artifact}")
        payload = _strict_json_object(artifact, context=f"Completed suite {suite_name} artifact")
        execution = payload.get("execution")
        expected_execution = {
            "schema": "forge.benchmark-execution.v1",
            "command": "suite",
            "suite": spec.slug,
            "suite_number": spec.number,
            "requested_device": record.get("requested_device"),
        }
        if not isinstance(execution, dict) or any(
            execution.get(key) != value for key, value in expected_execution.items()
        ):
            raise ValueError(
                f"Completed suite {suite_name} artifact execution lineage does not match its summary record"
            )


def suite_catalog() -> list[dict[str, str]]:
    """Return the stable public suite catalog."""
    return [asdict(spec) for spec in SUITES]


def resolve_suite(name: str) -> SuiteSpec:
    """Resolve a suite by number, slug, module stem, or artifact stem."""
    normalized = name.strip().lower().replace("_", "-")
    for spec in SUITES:
        aliases = {
            spec.number,
            str(int(spec.number)),
            spec.slug,
            spec.module.replace("_", "-"),
            Path(spec.artifact).stem.replace("_", "-"),
        }
        if normalized in aliases:
            return spec
    choices = ", ".join(f"{spec.number} ({spec.slug})" for spec in SUITES)
    raise ValueError(f"Unknown benchmark suite {name!r}. Available suites: {choices}")


def _contains_failure(value: object) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, dict):
        status = str(value.get("status", "")).lower()
        required_truths = (
            "all_real",
            "all_teachers_real",
            "actions_finite",
            "finite",
            "coverage_passed",
            "quality_passed",
        )
        total_failed = value.get("total_failed", 0)
        if (
            status
            in {"blocked", "cancelled", "error", "failed", "failure", "incomplete", "partial", "skipped", "timeout"}
            or value.get("error") not in (None, "")
            or value.get("skipped") is True
            or any(value.get(key) is False for key in required_truths if key in value)
            or (isinstance(total_failed, (int, float)) and total_failed > 0)
        ):
            return True
        return any(_contains_failure(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_failure(item) for item in value)
    return False


def _quarantine_previous_artifact(path: Path) -> Path | None:
    """Remove the previous result from the child process output path."""
    if not path.exists() and not path.is_symlink():
        return None
    if not path.is_file() and not path.is_symlink():
        raise ValueError(f"Suite artifact path is not a file: {path}")
    backup = path.with_name(f".{path.name}.previous-{uuid.uuid4().hex}")
    os.replace(path, backup)
    return backup


def _finish_previous_artifact(path: Path, backup: Path | None, *, accepted: bool) -> None:
    if backup is None:
        return
    if accepted:
        backup.unlink(missing_ok=True)
        return
    os.replace(backup, path)


def run_suite(
    suite: str | SuiteSpec,
    *,
    results_dir: str | Path,
    model_dir: str | Path = "models",
    export_dir: str | Path = "outputs/export",
    data_dir: str | Path | None = None,
    device: str = "auto",
    progress: TextIO | None = None,
) -> dict[str, Any]:
    """Run one suite and return a machine-readable execution record."""
    spec = resolve_suite(suite) if isinstance(suite, str) else suite
    if device not in {"auto", "cuda", "cpu"}:
        raise ValueError("Benchmark suite device must be one of: auto, cuda, cpu")
    output_dir = Path(results_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = output_dir / spec.artifact

    environment = os.environ.copy()
    environment["FORGE_BENCHMARK_RESULTS_DIR"] = str(output_dir)
    environment["FORGE_MODEL_DIR"] = str(Path(model_dir).expanduser())
    environment["FORGE_BENCHMARK_EXPORT_DIR"] = str(Path(export_dir).expanduser())
    if data_dir is not None:
        environment["FORGE_BENCHMARK_DATA_DIR"] = str(Path(data_dir).expanduser())
    if device == "cpu":
        environment["CUDA_VISIBLE_DEVICES"] = ""
    command = [sys.executable, "-m", f"forge.benchmark.suites.{spec.module}"]
    execution = _execution_provenance(spec, requested_device=device)
    started = datetime.now(UTC)
    previous_artifact = _quarantine_previous_artifact(artifact)
    accepted = False
    try:
        process = subprocess.Popen(
            command,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        lines: list[str] = []
        assert process.stdout is not None
        for line in process.stdout:
            lines.append(line.rstrip())
            if progress is not None:
                print(line, end="", file=progress, flush=True)
        return_code = process.wait()
        finished = datetime.now(UTC)
        artifact_updated = artifact.is_file()
        skip_lines = [line.strip() for line in lines if line.strip().startswith("SKIP:")]

        record: dict[str, Any] = {
            "suite": spec.slug,
            "number": spec.number,
            "description": spec.description,
            "requested_device": device,
            "started_at": started.isoformat(),
            "finished_at": finished.isoformat(),
            "elapsed_seconds": round((finished - started).total_seconds(), 3),
            "return_code": return_code,
            "status": "failed",
            "artifact": spec.artifact if artifact_updated else None,
        }
        if return_code != 0:
            record["error"] = f"Suite process exited with code {return_code}"
        elif skip_lines and not artifact_updated:
            record["status"] = "skipped"
            record["reason"] = skip_lines[0]
        elif not artifact_updated:
            record["error"] = "Suite completed without producing a fresh JSON artifact"
        else:
            try:
                payload = _strict_json_object(artifact, context="Suite artifact")
            except ValueError as exc:
                record["error"] = f"Suite produced an invalid JSON artifact: {exc}"
            else:
                provenance = payload.get("data_provenance")
                missing_real_data = spec.number in REAL_DATA_SUITE_NUMBERS and (
                    not isinstance(provenance, dict) or provenance.get("kind") != "real"
                )
                failed_checks = _contains_failure(payload)
                try:
                    _write_enriched_artifact(
                        artifact,
                        payload,
                        execution,
                    )
                    canonical_artifact, finalized_artifact = _resolve_canonical_artifact(output_dir, spec.artifact)
                    artifact_sha256 = _sha256_file(finalized_artifact)
                except (OSError, ValueError) as exc:
                    record["error"] = f"Suite artifact could not be finalized as strict JSON: {exc}"
                else:
                    record["artifact"] = canonical_artifact
                    record["artifact_sha256"] = artifact_sha256
                    record["status"] = "failed" if missing_real_data or failed_checks else "completed"
                if record["status"] == "failed":
                    record.setdefault(
                        "error",
                        (
                            "Suite artifact does not prove real input-data provenance"
                            if missing_real_data
                            else "Suite artifact contains one or more failed checks"
                        ),
                    )
        if record["status"] == "failed" and lines:
            record["output_tail"] = lines[-40:]
        accepted = record["status"] == "completed"
        if previous_artifact is not None and not accepted:
            record["artifact"] = None
            record.pop("artifact_sha256", None)
        return record
    finally:
        _finish_previous_artifact(artifact, previous_artifact, accepted=accepted)


def run_all_suites(
    *,
    results_dir: str | Path,
    model_dir: str | Path = "models",
    export_dir: str | Path = "outputs/export",
    data_dir: str | Path | None = None,
    device: str = "auto",
    progress: TextIO | None = None,
) -> dict[str, Any]:
    """Run every packaged suite sequentially and persist a JSON summary."""
    started = datetime.now(UTC)
    records = [
        run_suite(
            spec,
            results_dir=results_dir,
            model_dir=model_dir,
            export_dir=export_dir,
            data_dir=data_dir,
            device=device,
            progress=progress,
        )
        for spec in SUITES
    ]
    failures = sum(record["status"] == "failed" for record in records)
    skips = sum(record["status"] == "skipped" for record in records)
    status = "failed" if failures else ("completed_with_skips" if skips else "completed")
    output_dir = Path(results_dir).expanduser().resolve()
    output = output_dir / "suite_summary.json"
    summary = {
        "schema": "forge.benchmark-suite-run.v1",
        "started_at": started.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "status": status,
        "completed": sum(record["status"] == "completed" for record in records),
        "skipped": skips,
        "failed": failures,
        "suites": records,
        "artifact": output.name,
    }
    verify_suite_summary_artifacts(summary, results_dir=output_dir)
    write_json_artifact(output, summary)
    return summary


def summarize_existing_suites(*, results_dir: str | Path) -> dict[str, Any]:
    """Persist a content-bound summary for an already completed canonical suite set."""
    output_dir = Path(results_dir).expanduser().resolve()
    records: list[dict[str, Any]] = []
    for spec in SUITES:
        artifact = output_dir / spec.artifact
        payload = _strict_json_object(artifact, context=f"Suite {spec.number} artifact")
        execution = payload.get("execution")
        requested_device = execution.get("requested_device") if isinstance(execution, dict) else None
        provenance = payload.get("data_provenance")
        missing_real_data = spec.number in REAL_DATA_SUITE_NUMBERS and (
            not isinstance(provenance, dict) or provenance.get("kind") != "real"
        )
        failed_checks = _contains_failure(payload)
        artifact_sha256 = _sha256_file(artifact)
        status = "failed" if missing_real_data or failed_checks else "completed"
        record: dict[str, Any] = {
            "suite": spec.slug,
            "number": spec.number,
            "description": spec.description,
            "requested_device": requested_device,
            "status": status,
            "artifact": spec.artifact,
            "artifact_sha256": artifact_sha256,
        }
        if status == "failed":
            record["error"] = (
                "Suite artifact does not prove real input-data provenance"
                if missing_real_data
                else "Suite artifact contains one or more failed checks"
            )
        records.append(record)

    failures = sum(record["status"] == "failed" for record in records)
    output = output_dir / "suite_summary.json"
    now = datetime.now(UTC).isoformat()
    summary = {
        "schema": "forge.benchmark-suite-run.v1",
        "started_at": now,
        "finished_at": now,
        "status": "failed" if failures else "completed",
        "completed": sum(record["status"] == "completed" for record in records),
        "skipped": 0,
        "failed": failures,
        "source": "existing-artifacts",
        "suites": records,
        "artifact": output.name,
    }
    verify_suite_summary_artifacts(summary, results_dir=output_dir)
    write_json_artifact(output, summary)
    return summary
