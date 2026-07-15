#!/usr/bin/env python3
"""Verify registered teachers with real robot frames and local weights."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from forge.teacher_fleet import build_fleet_report, build_isolated_fleet_report, write_fleet_report
from forge.teachers.registry import get_registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="0,1,2,3", help="Comma-separated CUDA device ids")
    parser.add_argument("--model-dir", default="models", help="Local FORGE model directory")
    parser.add_argument(
        "--dataset-dir",
        default=os.environ.get("FORGE_DATASET_DIR", os.environ.get("FORGE_DATA_DIR", "data")),
        help="Root containing restored LeRobot datasets",
    )
    parser.add_argument("--predictions", type=int, default=5, help="Real frames per teacher")
    parser.add_argument(
        "--teachers",
        default=None,
        help="Optional comma-separated registry names; default verifies every teacher",
    )
    parser.add_argument(
        "--output",
        default="reports/teacher_fleet_2026.json",
        help="JSON report destination",
    )
    parser.add_argument("--in-process", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def _build_isolated_report(
    args: argparse.Namespace,
    teacher_names: list[str],
    gpu_ids: list[int],
) -> dict:
    """Run each heavyweight runtime in a fresh process and aggregate its report."""
    return build_isolated_fleet_report(
        teacher_names=teacher_names,
        model_dir=args.model_dir,
        dataset_dir=args.dataset_dir,
        gpu_ids=gpu_ids,
        predictions=args.predictions,
    )


def main() -> int:
    args = parse_args()
    gpu_ids = [int(value.strip()) for value in args.gpus.split(",") if value.strip()]
    teacher_names = (
        [value.strip() for value in args.teachers.split(",") if value.strip()]
        if args.teachers
        else get_registry().list_teachers()
    )
    if len(teacher_names) > 1 and not args.in_process:
        report = _build_isolated_report(args, teacher_names, gpu_ids)
    else:
        report = build_fleet_report(
            teacher_names=teacher_names,
            model_dir=Path(args.model_dir),
            dataset_dir=Path(args.dataset_dir),
            gpu_ids=gpu_ids,
            predictions=args.predictions,
        )
    path = write_fleet_report(report, args.output)
    print(json.dumps({"report": str(path), "all_real": report["all_real"], "verified": report["teachers_verified"]}))
    return 0 if report["all_real"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
