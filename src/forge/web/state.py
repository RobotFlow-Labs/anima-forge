"""Server state singleton — tracks loaded models, active jobs, benchmark history."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass


@dataclass
class Job:
    """Background job metadata."""

    job_id: str
    name: str
    status: str = "running"  # running, completed, failed
    started_at: float = 0.0
    finished_at: float = 0.0
    result: dict | None = None
    error: str | None = None


class ServerState:
    """Singleton holding all server-side state.

    Thread-safe via lock for mutable operations.
    """

    _instance: ServerState | None = None
    _lock = threading.Lock()
    _initialized: bool

    def __new__(cls) -> ServerState:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self._api_ready = False
        self._api_ready_reason = "initializing"
        self._start_time = time.time()
        self.loaded_teachers: dict[str, dict] = {}
        self.loaded_model: dict | None = None
        self.active_jobs: dict[str, Job] = {}
        self.benchmark_history: list[dict] = []
        self.train_state: dict = {"running": False, "step": 0, "loss": 0.0}
        self._job_lock = threading.Lock()

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self._start_time

    @property
    def is_ready(self) -> bool:
        """Indicate that web runtime is initialized and can serve API reads."""
        return self._api_ready

    @property
    def is_bootstrapped(self) -> bool:
        """Alias for compatibility with runtime-dependent UI checks."""
        return self.is_ready

    def set_ready(self, ready: bool, reason: str | None = None) -> None:
        """Mark API layer readiness for status/reporting."""
        self._api_ready = bool(ready)
        if reason:
            self._api_ready_reason = reason

    def add_job(self, name: str) -> str:
        """Create a new background job, return its ID."""
        job_id = uuid.uuid4().hex[:8]
        with self._job_lock:
            self.active_jobs[job_id] = Job(job_id=job_id, name=name, started_at=time.time())
        return job_id

    def complete_job(self, job_id: str, result: dict | None = None) -> None:
        """Mark job as completed."""
        with self._job_lock:
            if job_id in self.active_jobs:
                job = self.active_jobs[job_id]
                job.status = "completed"
                job.finished_at = time.time()
                job.result = result

    def fail_job(self, job_id: str, error: str) -> None:
        """Mark job as failed."""
        with self._job_lock:
            if job_id in self.active_jobs:
                job = self.active_jobs[job_id]
                job.status = "failed"
                job.finished_at = time.time()
                job.error = error

    def get_job(self, job_id: str) -> Job | None:
        return self.active_jobs.get(job_id)

    def get_system_status(self) -> dict:
        """Get system-wide status."""
        from forge import __version__

        gpu_info = _get_gpu_info()
        return {
            "gpu": gpu_info.get("name", "N/A"),
            "vram_total_gb": gpu_info.get("total_gb", 0),
            "vram_used_gb": gpu_info.get("used_gb", 0),
            "disk_free_gb": _get_disk_free_gb(),
            "api_ready": self.is_ready,
            "api_ready_reason": self._api_ready_reason,
            "uptime_s": self.uptime_seconds,
            "active_jobs": len([j for j in self.active_jobs.values() if j.status == "running"]),
            "version": __version__,
        }

    @classmethod
    def reset(cls) -> None:
        """Reset singleton (for testing)."""
        cls._instance = None


def _get_gpu_info() -> dict:
    """Get GPU info via torch.cuda."""
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            free_bytes, total_bytes = torch.cuda.mem_get_info(0)
            total_gb = total_bytes / (1024**3)
            used_gb = (total_bytes - free_bytes) / (1024**3)
            return {"name": props.name, "total_gb": total_gb, "used_gb": used_gb}
    except Exception:
        pass
    return {"name": "N/A", "total_gb": 0, "used_gb": 0}


def _get_disk_free_gb() -> float:
    """Get free disk space in GB."""
    import shutil

    usage = shutil.disk_usage("/")
    return usage.free / (1024**3)
