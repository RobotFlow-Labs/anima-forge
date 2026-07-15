#!/usr/bin/env python3
"""Real-time GPU + memory profiler for FORGE jobs.

Usage:
  uv run python scripts/gpu_fit_profiler.py \
      --command "uv run forge eval run-all --checkpoint ... --json"

  uv run python scripts/gpu_fit_profiler.py --pid 12345
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import time
from collections import defaultdict, deque
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict


class GpuMetric(TypedDict):
    """One parsed nvidia-smi GPU row."""

    index: int
    name: str
    memory_total_mb: float
    memory_used_mb: float
    memory_free_mb: float
    utilization_gpu: float
    utilization_mem: float
    power_watts: float
    temp_c: float


def _run(cmd: list[str], *, text: bool = True) -> str:
    return subprocess.check_output(cmd, text=text).strip()


def query_gpus() -> list[GpuMetric]:
    raw = None
    if shutil.which("nvidia-smi"):
        try:
            raw = _run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,power.draw,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ]
            )
        except (subprocess.CalledProcessError, OSError):
            raw = None
    if raw is None:
        from forge.gpu_utils import _torch_gpu_samples

        return [
            {
                "index": int(sample["index"]),
                "name": str(sample["name"]),
                "memory_total_mb": float(sample["memory_total_mib"]),
                "memory_used_mb": float(sample["memory_used_mib"]),
                "memory_free_mb": float(sample["memory_free_mib"]),
                "utilization_gpu": -1.0,
                "utilization_mem": -1.0,
                "power_watts": -1.0,
                "temp_c": -1.0,
            }
            for sample in _torch_gpu_samples()
        ]
    metrics: list[GpuMetric] = []
    for line in raw.splitlines():
        fields = [p.strip() for p in line.split(",")]
        if len(fields) < 9:
            continue
        metrics.append(
            {
                "index": int(fields[0]),
                "name": fields[1],
                "memory_total_mb": float(fields[2]),
                "memory_used_mb": float(fields[3]),
                "memory_free_mb": float(fields[4]),
                "utilization_gpu": float(fields[5]),
                "utilization_mem": float(fields[6]),
                "power_watts": float(fields[7]),
                "temp_c": float(fields[8]),
            }
        )
    return metrics


def query_proc_mem(pid: int) -> tuple[float, float]:
    # Returns RSS and VSZ in MiB
    status_file = f"/proc/{pid}/status"
    rss_kb = None
    vsz_kb = None
    try:
        with open(status_file, encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss_kb = float(line.split()[1])
                elif line.startswith("VmSize:"):
                    vsz_kb = float(line.split()[1])
        if rss_kb is not None:
            return rss_kb / 1024.0, (0.0 if vsz_kb is None else vsz_kb / 1024.0)
    except FileNotFoundError:
        pass
    return 0.0, 0.0


def iter_proc_pids(pid: int) -> list[int]:
    """Return pid and known descendant pids for shell-launched command trees."""
    seen = set()
    queue = deque([pid])
    result = []

    while queue:
        current = queue.popleft()
        if current in seen:
            continue
        seen.add(current)
        result.append(current)

        try:
            children = _run(["ps", "-o", "pid=", "--ppid", str(current)])
        except subprocess.CalledProcessError:
            children = ""

        for line in children.splitlines():
            line = line.strip()
            if not line.isdigit():
                continue
            queue.append(int(line))

    return result


def query_system_mem() -> tuple[float, float]:
    # Returns used and total in MiB
    meminfo = {}
    with open("/proc/meminfo", encoding="utf-8") as f:
        for line in f:
            if ":" in line:
                key, val = line.split(":", 1)
                val = val.strip().split()[0]
                if val.isdigit():
                    meminfo[key] = float(val) / 1024.0  # KiB -> MiB
    total = meminfo.get("MemTotal", 0.0)
    available = meminfo.get("MemAvailable", 0.0)
    used = total - available
    return used, total


def query_proc_gpu_memory(pids: list[int]) -> dict[str, int]:
    # Returns used memory by GPU for the provided pid in MiB.
    gpu_mem: dict[str, int] = {}
    if not shutil.which("nvidia-smi"):
        return gpu_mem
    try:
        raw = _run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,used_memory",
                "--format=csv,noheader,nounits",
            ]
        )
    except subprocess.CalledProcessError:
        return gpu_mem

    for line in raw.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            pid_seen = int(parts[1])
        except ValueError:
            continue
        if pid_seen not in pids:
            continue
        gpu_uuid = parts[0]
        used = float(parts[2])
        # Keep a reverse index map by matching uuid later in query_gpus.
        gpu_mem[gpu_uuid] = int(used)
    return gpu_mem


def map_gpu_uuid_to_index() -> dict[str, int]:
    uuid_map: dict[str, int] = {}
    if not shutil.which("nvidia-smi"):
        return uuid_map
    try:
        raw = _run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"])
    except (subprocess.CalledProcessError, OSError):
        return uuid_map
    for line in raw.splitlines():
        idx, uuid = [p.strip() for p in line.split(",")]
        uuid_map[uuid] = int(idx)
    return uuid_map


def query_tracked_process_mem(pids: list[int]) -> tuple[float, float]:
    total_rss = 0.0
    total_vsz = 0.0
    for child_pid in pids:
        rss, vsz = query_proc_mem(child_pid)
        total_rss += rss
        total_vsz += vsz
    return total_rss, total_vsz


def format_row(
    ts: str,
    elapsed: float,
    gpus: list[GpuMetric],
    p_rss: float,
    p_vsz: float,
    proc_gpu_mem: dict[int, int],
    pid: int,
) -> str:
    used = []
    for gpu in gpus:
        utilization = gpu["utilization_gpu"]
        utilization_text = "n/a" if utilization < 0 else f"{utilization}%"
        used.append(
            f"GPU{gpu['index']}: {int(gpu['memory_used_mb'])}MB/{int(gpu['memory_total_mb'])}MB {utilization_text}"
        )
    proc = ", ".join(f"GPU{idx}: {used_mb}MB" for idx, used_mb in sorted(proc_gpu_mem.items())) or "none"
    return (
        f"[{ts}] "
        f"pid={pid} elapsed={elapsed:7.1f}s "
        f"cpu_mem={p_rss:.0f}MiB "
        f"system_mem={query_system_mem()[0]:.0f}/{query_system_mem()[1]:.0f}MiB "
        f"gpu_proc={proc} | " + " ; ".join(used)
    )


def launch_and_monitor(command: str, interval: float, csv_path: str, out_json: bool) -> int:
    proc = subprocess.Popen(command, shell=True, executable="/bin/bash")
    return monitor(proc.pid, interval, csv_path, out_json, process_handle=proc)


def monitor(
    pid: int,
    interval: float,
    csv_path: str,
    out_json: bool,
    process_handle: subprocess.Popen | None = None,
) -> int:
    start = time.time()
    peak_gpu: defaultdict[int, float] = defaultdict(float)
    peak_sys = 0.0
    total_cpu_mem = 0.0
    peak_total_gpu_used_mib = 0.0
    end = 0.0

    fieldnames = [
        "ts_iso",
        "elapsed_s",
        "pid",
        "process_rss_mib",
        "process_vsz_mib",
        "system_used_mib",
        "system_total_mib",
        "max_gpu_util_gpu",
        "max_gpu_util_mem",
        "gpu_usage_json",
        "command_status",
    ]

    proc_rows: list[dict[str, str | int]] = []
    uuid_map = map_gpu_uuid_to_index()
    peak_gpu_total_used = 0.0

    def _format_proc_gpu_map(proc_gpu_map: dict[int, int]) -> str:
        return ", ".join(f"GPU{idx}: {used_mb}MB" for idx, used_mb in sorted(proc_gpu_map.items())) or "none"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        print(f"[PROFILE] started pid={pid} | interval={interval}s | log={csv_path}", flush=True)
        status = "running"
        while True:
            elapsed = time.time() - start
            ts = datetime.now(UTC).isoformat()
            gpus = query_gpus()
            tracked_pids = iter_proc_pids(pid)
            p_rss, p_vsz = query_tracked_process_mem(tracked_pids)
            total_gpu_used = sum(g["memory_used_mb"] for g in gpus)
            if total_gpu_used > peak_total_gpu_used_mib:
                peak_total_gpu_used_mib = total_gpu_used
            proc_gpu_bytes: dict[int, int] = {}
            proc_gpu_raw = query_proc_gpu_memory(tracked_pids)
            for gpu_uuid, used_mb in proc_gpu_raw.items():
                idx = uuid_map.get(gpu_uuid)
                if idx is not None:
                    proc_gpu_bytes[idx] = proc_gpu_bytes.get(idx, 0) + used_mb
                    if used_mb > peak_gpu[idx]:
                        peak_gpu[idx] = used_mb

            system_used, system_total = query_system_mem()
            if p_rss > total_cpu_mem:
                total_cpu_mem = p_rss
            if system_used > peak_sys:
                peak_sys = system_used
            max_util_gpu = max((g["utilization_gpu"] for g in gpus), default=0.0)
            max_util_mem = max((g["utilization_mem"] for g in gpus), default=0.0)
            command_status = "running"

            row: dict[str, str | int] = {
                "ts_iso": ts,
                "elapsed_s": f"{elapsed:.2f}",
                "pid": pid,
                "process_rss_mib": f"{p_rss:.2f}",
                "process_vsz_mib": f"{p_vsz:.2f}",
                "system_used_mib": f"{system_used:.2f}",
                "system_total_mib": f"{system_total:.2f}",
                "max_gpu_util_gpu": f"{max_util_gpu:.2f}",
                "max_gpu_util_mem": f"{max_util_mem:.2f}",
                "gpu_usage_json": json.dumps(proc_gpu_bytes, sort_keys=True),
                "command_status": command_status,
            }
            writer.writerow(row)
            f.flush()

            print(format_row(ts, elapsed, gpus, p_rss, p_vsz, proc_gpu_bytes, pid), flush=True)
            proc_rows.append(row)

            if process_handle is not None:
                status_code = process_handle.poll()
            else:
                try:
                    os.kill(pid, 0)
                    status_code = None
                except ProcessLookupError:
                    status_code = 0

            if status_code is not None:
                end = time.time()
                status = "finished" if status_code == 0 else "failed"
                break
            time.sleep(interval)

        runtime = end - start
        if not proc_rows:
            print("[PROFILE] no samples collected", flush=True)
            return 0

        print("\n[PROFILE] summary", flush=True)
        print(f"  status: {status}", flush=True)
        print(f"  runtime_s: {runtime:.2f}", flush=True)
        print(f"  process_peak_rss_mib: {total_cpu_mem:.1f}", flush=True)
        print(f"  system_peak_used_mib: {peak_sys:.1f}/{system_total:.1f}", flush=True)
        if peak_gpu:
            for idx in sorted(peak_gpu):
                print(f"  peak_gpu_{idx}_proc_mem_mib: {int(peak_gpu[idx])}", flush=True)
            peak_gpu_total_used = sum(peak_gpu.values())
            print(f"  peak_sum_proc_gpu_mem_mib: {peak_gpu_total_used:.0f}", flush=True)
        else:
            peak_gpu_total_used = peak_total_gpu_used_mib
            print(f"  peak_sum_proc_gpu_mem_mib: {peak_gpu_total_used:.0f}", flush=True)
            if proc_rows:
                print("  rough_fit_estimate_note: fallback-to-system-gpu-snapshot", flush=True)

        maybe_total = query_gpus()
        if maybe_total:
            total_capacity = sum(g["memory_total_mb"] for g in maybe_total)
            fit = "yes" if peak_gpu_total_used < (0.95 * total_capacity) else "no"
            print(f"  total_gpu_capacity_mib: {total_capacity:.0f}", flush=True)
            print(f"  rough_fit_estimate_by_peak: {fit}", flush=True)
        else:
            print("  total_gpu_capacity_mib: n/a", flush=True)
            print("  rough_fit_estimate_by_peak: n/a", flush=True)

        if out_json:
            json_path = csv_path + ".json"
            with open(json_path, "w", encoding="utf-8") as fjson:
                json.dump(
                    {
                        "pid": pid,
                        "interval_s": interval,
                        "status": status,
                        "runtime_s": runtime,
                        "peak_process_rss_mib": total_cpu_mem,
                        "peak_system_used_mib": peak_sys,
                        "peak_gpu_mem_by_index_mib": dict(peak_gpu),
                        "rows": proc_rows,
                    },
                    fjson,
                    indent=2,
                )
            print(f"  json_summary: {json_path}", flush=True)

    return 0 if status == "finished" else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile CUDA and RAM usage during a FORGE command or PID")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--command", type=str, help="Shell command to run and profile")
    mode.add_argument("--pid", type=int, help="PID to attach to")
    parser.add_argument("--interval", type=float, default=5.0, help="Sampling interval in seconds (default: 5)")
    parser.add_argument(
        "--out",
        type=str,
        default="outputs/validation/gpu_full/gpu_profile.csv",
        help="Output CSV file for profiler samples",
    )
    parser.add_argument("--json", action="store_true", help="Also emit a JSON summary artifact")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_csv = str(out_path)
    if args.command:
        return launch_and_monitor(args.command, args.interval, out_csv, args.json)
    return monitor(args.pid, args.interval, out_csv, args.json, process_handle=None)


if __name__ == "__main__":
    raise SystemExit(main())
