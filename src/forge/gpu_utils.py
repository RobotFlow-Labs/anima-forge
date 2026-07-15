"""FORGE GPU Utility Module — mandatory GPU check and multi-GPU management.

This module is the standard entry point for all GPU-related operations in FORGE.
Before launching any training or inference workload, call `require_free_gpus()` to
assert that sufficient GPU resources are available. It also provides helpers for
DDP (DistributedDataParallel) setup and teardown, and a human-readable status table.

Usage:
    from forge.gpu_utils import require_free_gpus, setup_ddp, cleanup_ddp

    free = require_free_gpus(min_gpus=2, min_free_mb=8000)
    # → [0, 1]  (or raises RuntimeError if not enough GPUs)
"""

import subprocess
from typing import Any

import torch


def _smi(query: str) -> str | None:
    """Run nvidia-smi --query-gpu and return stdout, or None on any failure."""
    try:
        r = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return r.stdout.strip() if r.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


def _torch_gpu_samples() -> list[dict[str, Any]]:
    """Return system-wide CUDA memory snapshots when NVML is unavailable.

    CUDA does not expose device utilization through this API, so utilization
    fields use ``-1`` as an explicit unavailable sentinel. ``mem_get_info`` is
    system-wide and therefore still prevents memory-heavy jobs from being
    mistaken for free devices.
    """
    if not torch.cuda.is_available():
        return []
    samples: list[dict[str, Any]] = []
    for index in range(torch.cuda.device_count()):
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(index)
            name = torch.cuda.get_device_name(index)
        except (RuntimeError, OSError):
            continue
        total_mib = int(total_bytes / 1024**2)
        free_mib = int(free_bytes / 1024**2)
        samples.append(
            {
                "index": index,
                "name": name,
                "memory_total_mib": total_mib,
                "memory_used_mib": max(total_mib - free_mib, 0),
                "memory_free_mib": free_mib,
                "utilization_gpu": -1,
                "utilization_memory": -1,
                "metrics_source": "torch.cuda.mem_get_info",
            }
        )
    return samples


def get_free_gpus(min_free_mb: int = 2000, max_utilization: int = 15) -> list[int]:
    """Return GPU indices with >= min_free_mb MiB free AND <= max_utilization% utilization.

    Parses CSV from nvidia-smi. When NVML is broken, CUDA free-memory data is
    used and the occupied-memory percentage is a conservative utilization
    proxy because compute utilization is unavailable through the CUDA API.
    """
    raw = _smi("index,memory.free,utilization.gpu")
    if raw is None:
        fallback_free = []
        for sample in _torch_gpu_samples():
            total = int(sample["memory_total_mib"])
            used = int(sample["memory_used_mib"])
            memory_occupancy = (100 * used / total) if total > 0 else 100.0
            if int(sample["memory_free_mib"]) >= min_free_mb and memory_occupancy <= max_utilization:
                fallback_free.append(int(sample["index"]))
        return sorted(fallback_free)
    free: list[int] = []
    for line in raw.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            idx, mem, util = int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError:
            continue
        if mem >= min_free_mb and util <= max_utilization:
            free.append(idx)
    return sorted(free)


def get_gpu_samples() -> list[dict[str, Any]]:
    """Return per-GPU memory/utilization snapshots from ``nvidia-smi``.

    Each snapshot includes:
      - ``index``: GPU index
      - ``name``: GPU model name
      - ``memory_total_mib``: total memory in MiB
      - ``memory_used_mib``: used memory in MiB
      - ``memory_free_mib``: free memory in MiB
      - ``utilization_gpu``: GPU utilization percentage
      - ``utilization_memory``: memory controller utilization percentage
    """
    raw = _smi("index,name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory")
    if raw is None:
        return _torch_gpu_samples()

    samples: list[dict[str, Any]] = []
    for line in raw.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 7:
            continue

        idx: int
        mem_total: int
        mem_used: int
        mem_free: int
        gpu_util: int
        mem_util: int

        try:
            idx = int(parts[0])
            mem_total = int(float(parts[2]))
            mem_used = int(float(parts[3]))
            mem_free = int(float(parts[4]))
            gpu_util = int(float(parts[5]))
            mem_util = int(float(parts[6]))
        except (ValueError, TypeError):
            continue

        samples.append(
            {
                "index": idx,
                "name": parts[1],
                "memory_total_mib": mem_total,
                "memory_used_mib": mem_used,
                "memory_free_mib": mem_free,
                "utilization_gpu": gpu_util,
                "utilization_memory": mem_util,
                "metrics_source": "nvidia-smi",
            }
        )

    return sorted(samples, key=lambda sample: sample["index"])


def get_gpu_count() -> int:
    """Return total number of CUDA-capable GPUs (0 if CUDA is unavailable)."""
    return torch.cuda.device_count()


def require_free_gpus(min_gpus: int = 1, min_free_mb: int = 2000) -> list[int]:
    """Return free GPU indices or raise RuntimeError if fewer than min_gpus qualify.

    Raises:
        RuntimeError: Fewer than min_gpus GPUs have >= min_free_mb MiB free.
    """
    free = get_free_gpus(min_free_mb=min_free_mb)
    if len(free) < min_gpus:
        raise RuntimeError(
            f"FORGE requires {min_gpus} free GPU(s) with >= {min_free_mb} MiB, "
            f"but only {len(free)} qualify: {free}. "
            "Check utilization with `forge gpu status` or `nvidia-smi`."
        )
    print(f"[forge.gpu_utils] Free GPUs (>= {min_free_mb} MiB): {free}")
    return free


def setup_ddp(rank: int, world_size: int, backend: str = "nccl") -> None:
    """Initialize the PyTorch Distributed process group and set the CUDA device.

    Callers must set MASTER_ADDR and MASTER_PORT before spawning workers.
    Use backend="gloo" for CPU-only or debugging runs.
    """
    torch.distributed.init_process_group(backend=backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp() -> None:
    """Destroy the PyTorch Distributed process group to release distributed resources."""
    torch.distributed.destroy_process_group()


def print_gpu_status() -> None:
    """Pretty-print a table of all GPUs: index, name, free/total memory, utilization.

    Falls back gracefully when nvidia-smi is unavailable, reporting the torch
    device count instead.

    Example output::

        GPU  Name      Free(MiB)  Total(MiB)  Util%
        ---  --------  ---------  ----------  -----
        0    NVIDIA L4     22412       23028      1
        1    NVIDIA L4      3104       23028     87
    """
    raw = _smi("index,name,memory.free,memory.total,utilization.gpu")
    if raw is None:
        samples = _torch_gpu_samples()
        if not samples:
            print("[forge.gpu_utils] nvidia-smi unavailable and CUDA reported no GPUs.")
            return
        print("[forge.gpu_utils] nvidia-smi unavailable; utilization is n/a (CUDA memory fallback).")
        raw = "\n".join(
            f"{sample['index']},{sample['name']},{sample['memory_free_mib']},{sample['memory_total_mib']},n/a"
            for sample in samples
        )

    rows = []
    for line in raw.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 5:
            rows.append(parts)

    if not rows:
        print("[forge.gpu_utils] No GPUs found.")
        return

    # Dynamic column widths
    headers = ("GPU", "Name", "Free(MiB)", "Total(MiB)", "Util%")
    widths = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    sep = "  ".join("-" * w for w in widths)
    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*row))
