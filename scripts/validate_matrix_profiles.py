#!/usr/bin/env python3
"""Validate that profiled matrix steps emitted CSV/JSON artifacts."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

HEAVY_STEPS = {
    "tier1_json_benchmark_run",
    "tier1_json_quant_bench",
    "tier1_json_quant_run",
    "tier1_json_train_start",
    "tier1_json_ud_start",
    "tier1_json_eval_smoke",
    "tier2_benchmark",
    "tier2_eval_smoke",
    "tier2_eval_libero",
    "tier2_eval_simpler",
    "tier2_eval_vlabench",
    "tier2_train_start",
    "tier2_profile_benchmark",
    "tier3_pipeline_short",
    "tier3_serve",
    "tier3_eval_serve",
    "tier3_eval_smoke_short",
    "tier3_eval_run_all",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate matrix profiler artifacts.")
    parser.add_argument(
        "--matrix-results",
        required=True,
        help="Path to matrix_results.csv to validate.",
    )
    parser.add_argument(
        "--matrix-dir",
        default=".",
        help="Directory containing matrix artifacts (default: parent of matrix-results).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    matrix_path = Path(args.matrix_results)
    if not matrix_path.exists():
        print(f"[skip] matrix file missing: {matrix_path}")
        return 0

    matrix_dir = Path(args.matrix_dir)
    if not matrix_dir.exists():
        matrix_dir = matrix_path.parent

    missing: list[tuple[str, Path, Path]] = []
    with matrix_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = (row.get("step") or "").strip()
            if step not in HEAVY_STEPS:
                continue
            csv_path = matrix_dir / f"{step}_prof.csv"
            json_path = matrix_dir / f"{step}_prof.csv.json"
            if not csv_path.exists() or not json_path.exists():
                missing.append((step, csv_path, json_path))

    if not missing:
        print(f"[ok] profiler artifacts found for all heavy steps in {matrix_path}")
        return 0

    print(f"[error] missing profiler artifacts in {matrix_path}:")
    for step, missing_csv, missing_json in missing:
        print(f"  - {step}")
        print(f"      csv={missing_csv} exists={missing_csv.exists()}")
        print(f"      json={missing_json} exists={missing_json.exists()}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
