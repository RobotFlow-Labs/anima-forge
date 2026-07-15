"""Truthful Markdown report generation for FORGE result artifacts."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from forge import __version__


def _value(mapping: object, key: str, suffix: str = "") -> str:
    if not isinstance(mapping, Mapping):
        return "Not measured"
    value = mapping.get(key)
    if value is None:
        return "Not measured"
    return f"{value}{suffix}"


def _first_value(*candidates: tuple[object, str], suffix: str = "") -> str:
    for mapping, key in candidates:
        value = _value(mapping, key, suffix)
        if value != "Not measured":
            return value
    return "Not measured"


def generate_report(
    results: dict[str, Any] | None = None,
    results_path: str | None = None,
    output_path: str = "./outputs/FORGE_REPORT.md",
) -> str:
    """Generate a report without substituting synthetic benchmark values."""
    if results is None and results_path:
        with open(results_path) as stream:
            results = json.load(stream)
    if results is None:
        results = {}
    if not isinstance(results, dict):
        raise ValueError("Results artifact must contain a JSON object")

    model = results.get("model") if isinstance(results.get("model"), dict) else {}
    inference = results.get("inference") if isinstance(results.get("inference"), dict) else {}
    training = results.get("training") if isinstance(results.get("training"), dict) else {}
    compression = results.get("compression") if isinstance(results.get("compression"), dict) else {}
    export = results.get("export") if isinstance(results.get("export"), dict) else {}
    provenance = results.get("provenance") if isinstance(results.get("provenance"), dict) else {}
    input_provenance = results.get("input_provenance") if isinstance(results.get("input_provenance"), dict) else {}
    latency = results.get("latency") if isinstance(results.get("latency"), dict) else {}
    throughput = results.get("throughput") if isinstance(results.get("throughput"), dict) else {}
    distill = results.get("distill") if isinstance(results.get("distill"), dict) else {}
    comparison = results.get("teacher_comparison") if isinstance(results.get("teacher_comparison"), dict) else None
    throughput_value = _value(inference, "throughput_fps", " FPS")
    if throughput_value == "Not measured":
        throughput_value = _value(throughput, "actions_per_second", " actions/s")

    comparison_section = """## Teacher comparison

Not measured. FORGE does not infer teacher/student speedups from parameter counts.
"""
    if comparison:
        rows = "\n".join(f"| {name} | {value} |" for name, value in sorted(comparison.items()))
        comparison_section = f"""## Teacher comparison

| Metric | Measured value |
|---|---:|
{rows}
"""

    report = f"""# FORGE benchmark report

Generated {time.strftime("%Y-%m-%d %H:%M")} by FORGE {__version__}.

Missing metrics are shown as **Not measured**; no random or historical values are
substituted.

## Provenance

| Input | Status |
|---|---|
| Vision weights | {_value(provenance, "vision")} |
| Language weights | {_value(provenance, "language")} |
| Teacher labels | {_value(provenance, "labels")} |
| Source Git SHA | {_value(provenance, "git_sha")} |
| Source checkpoint | {_value(results, "source_checkpoint")} |
| Benchmark input | {_value(input_provenance, "kind")} |
| Dataset | {_value(input_provenance, "dataset")} |
| Instruction source | {_value(input_provenance, "instruction_source")} |

## Student

| Metric | Value |
|---|---:|
| Variant | {_first_value((model, "variant"), (results, "variant"), (results, "model_name"), (results, "config"))} |
| Total parameters | {_first_value((model, "total_params_M"), (compression, "student_params_m"), suffix=" M")} |
| Trainable parameters | {_value(model, "trainable_params_M", " M")} |
| Model size (bf16) | {_value(model, "size_bf16_GB", " GiB")} |

## Inference

| Metric | Value |
|---|---:|
| Mean latency | {_first_value((inference, "latency_avg_ms"), (latency, "mean_ms"), suffix=" ms")} |
| P50 latency | {_first_value((inference, "latency_p50_ms"), (latency, "p50_ms"), suffix=" ms")} |
| P99 latency | {_first_value((inference, "latency_p99_ms"), (latency, "p99_ms"), suffix=" ms")} |
| Throughput | {throughput_value} |
| GPU memory | {_value(inference, "gpu_memory_GB", " GiB")} |
| Device | {_value(results, "device")} |

## Distillation

| Metric | Value |
|---|---:|
| Steps | {_first_value((training, "steps"), (distill, "total_steps"))} |
| Wall time | {_first_value((training, "time_s"), (distill, "elapsed_seconds"), suffix=" s")} |
| Steps per second | {_first_value((training, "steps_per_sec"), (distill, "steps_per_second"))} |
| Initial loss | {_first_value((training, "loss_start"), (distill, "initial_loss"))} |
| Final loss | {_first_value((training, "loss_end"), (distill, "final_loss"))} |
| Loss reduction | {_first_value((training, "loss_reduction_pct"), (distill, "loss_reduction_percent"), suffix="%")} |

## Compression

| Metric | Value |
|---|---:|
| Layers before | {_value(compression, "layers_before")} |
| Layers after | {_value(compression, "layers_after")} |
| Artifact/model size | {_first_value((compression, "int4_size_MB"), (compression, "model_size_mb"), suffix=" MB")} |
| Compression ratio | {_value(compression, "compression_ratio", "×")} |
| Runtime VRAM | {_value(compression, "vram_mb", " MB")} |

## Export

| Metric | Value |
|---|---:|
| Status | {_value(export, "status")} |
| ONNX total size | {_value(export, "onnx_size_MB", " MB")} |
| ONNX Runtime provider | {_value(export, "provider")} |

{comparison_section}
## Pipeline

```text
real teacher labels → SigLIP2 + Qwen3/SmolLM2 student → prune/pack → ONNX/TensorRT/MLX
```
"""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(report, encoding="utf-8")
    return report
