"""EvalRunner — orchestrates vla-eval benchmarks against FORGE students.

Starts the ForgeModelServer, launches Docker benchmark containers via
the vla-eval harness, waits for results, and parses them.

Usage:
    runner = EvalRunner(
        checkpoint_path="./outputs/checkpoints/best.pt",
        variant="nano",
    )
    result = runner.run_benchmark("libero")
    results = runner.run_all()
"""

from __future__ import annotations

import importlib
import inspect
import json
import logging
import os
import shutil
import socket
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Docker images for each benchmark
BENCHMARK_IMAGES = {
    "libero": "ghcr.io/allenai/vla-evaluation-harness/libero:latest",
    "simpler": "ghcr.io/allenai/vla-evaluation-harness/simpler:latest",
    "vlabench": "ghcr.io/allenai/vla-evaluation-harness/vlabench:latest",
}

# MuJoCo-based harnesses can render through OSMesa when the NVIDIA container
# runtime is unavailable. SimplerEnv uses SAPIEN/Vulkan and requires a working
# Vulkan physical device; OSMesa or Mesa llvmpipe cannot satisfy its extensions.
CPU_RENDER_FALLBACK_BENCHMARKS = frozenset({"libero", "vlabench"})

_ERROR_SNIPPET_LIMIT = 12_000

# Default benchmark suites
BENCHMARK_SUITES = {
    "libero": "libero_spatial",
    "simpler": "google_robot",
    "vlabench": "vlabench",
}

# Benchmark class import paths for vla-eval config
BENCHMARK_CLASSES = {
    "libero": "vla_eval.benchmarks.libero.benchmark:LIBEROBenchmark",
    "simpler": "vla_eval.benchmarks.simpler.benchmark:SimplerEnvBenchmark",
    "vlabench": "vla_eval.benchmarks.vlabench.benchmark:VLABenchBenchmark",
}

_BENCHMARK_PARAM_DEFAULTS: dict[str, dict[str, Any]] = {
    "libero": {"suite": "libero_spatial", "seed": 42, "num_steps_wait": 10},
    "simpler": {"seed": 42},
    "vlabench": {},
}

# Python 3.8 compat patch for Docker images that ship with TypeAlias import
_TYPES_PATCH = '''\
"""Shared type definitions for the vla-eval wire protocol (Python 3.8 compat)."""
from __future__ import annotations
from typing import Any, Dict

Observation = Dict[str, Any]
Action = Dict[str, Any]
Task = Dict[str, Any]
EpisodeResult = Dict[str, Any]
'''


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run."""

    benchmark: str = "libero"
    suite: str = "libero_spatial"
    episodes_per_task: int = 20
    max_tasks: int = 10
    seed: int = 42
    server_url: str = "ws://localhost:8000"
    docker_image: str = ""
    config_path: str | None = None


def _write_types_patch() -> str:
    """Write a Python 3.8 compatible types.py patch to a temp file.

    The vla-eval Docker images ship with Python 3.8 but some versions of
    the baked-in vla-eval code use ``from typing import TypeAlias`` which
    is Python 3.10+. We mount this patched file over the container's
    ``types.py`` to fix the import.

    Returns:
        Path to the temp file (caller must clean up).
    """
    fd, path = tempfile.mkstemp(suffix=".py", prefix="vla-eval-types-patch-")
    with os.fdopen(fd, "w") as f:
        f.write(_TYPES_PATCH)
    return path


class EvalRunner:
    """Runs vla-eval benchmarks against a FORGE student."""

    def __init__(
        self,
        checkpoint_path: str,
        variant: str = "nano",
        model_dir: str | None = None,
        device: str = "cuda",
        output_dir: str = "./outputs/eval",
        port: int = 8000,
        allow_mock: bool = False,
    ):
        self.checkpoint_path = checkpoint_path
        self.variant = variant
        self.model_dir = model_dir
        self.device = device
        self.output_dir = Path(output_dir)
        self.port = port
        self.allow_mock = allow_mock
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _filter_benchmark_params(
        benchmark_class_path: str,
        candidate_params: dict[str, Any],
    ) -> dict[str, Any]:
        """Keep benchmark params that are accepted by the vla-eval benchmark __init__ signature.

        The vla-eval benchmark classes changed between releases; this avoids
        hard failures from passing unsupported kwargs such as
        ``num_steps_wait``.
        """
        try:
            module_name, class_name = benchmark_class_path.split(":")
            mod = importlib.import_module(module_name)
            cls = getattr(mod, class_name)
            signature = inspect.signature(cls.__init__)
            init_parameters = signature.parameters
            if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in init_parameters.values()):
                return dict(candidate_params)
            return {key: value for key, value in candidate_params.items() if key in init_parameters}
        except Exception:
            return dict(candidate_params)

    def _build_eval_config(
        self,
        benchmark: str,
        results_dir: str,
        episodes_per_task: int = 20,
        max_tasks: int = 10,
        seed: int = 42,
        server_port: int | None = None,
    ) -> dict[str, Any]:
        """Build a vla-eval config dict for a benchmark run."""
        docker_image = BENCHMARK_IMAGES.get(benchmark)
        benchmark_class = BENCHMARK_CLASSES.get(benchmark)
        _ = BENCHMARK_SUITES.get(benchmark, benchmark)

        if not docker_image or not benchmark_class:
            raise ValueError(f"Unknown benchmark: {benchmark}. Available: {list(BENCHMARK_IMAGES.keys())}")

        server_port = server_port or self.port
        candidate_params = dict(_BENCHMARK_PARAM_DEFAULTS.get(benchmark, {}))
        if "seed" in candidate_params:
            candidate_params["seed"] = seed
        params = self._filter_benchmark_params(benchmark_class, candidate_params)

        return {
            "server": {"url": f"ws://localhost:{server_port}"},
            "docker": {"image": docker_image},
            "output_dir": results_dir,
            "benchmarks": [
                {
                    "benchmark": benchmark_class,
                    "mode": "sync",
                    "episodes_per_task": episodes_per_task,
                    "max_tasks": max_tasks,
                    "params": params,
                }
            ],
        }

    @staticmethod
    def _bind_server_url(eval_config: dict[str, Any], server_port: int) -> dict[str, Any]:
        """Bind custom and generated configs to the server port chosen for this run."""
        rebound = dict(eval_config)
        configured_server = rebound.get("server", {})
        if configured_server is None:
            configured_server = {}
        if not isinstance(configured_server, dict):
            raise ValueError("Evaluation config 'server' must be a mapping")
        server = dict(configured_server)
        server["url"] = f"ws://localhost:{server_port}"
        rebound["server"] = server
        return rebound

    # Conda env name + Python version per benchmark Docker image
    _DOCKER_CONDA_ENVS: dict[str, tuple[str, str]] = {
        "libero": ("libero", "3.8"),
        "simpler": ("simpler", "3.10"),
        "vlabench": ("vlabench", "3.10"),
    }

    def _run_docker_benchmark(
        self,
        eval_config: dict[str, Any],
        results_dir: Path,
        benchmark: str = "",
    ) -> subprocess.CompletedProcess:
        """Run a benchmark inside a Docker container with TypeAlias compat patch.

        Uses the same approach as ``vla-eval run``: passes --no-docker to the
        container's entrypoint and mounts the results directory.
        """
        docker = shutil.which("docker")
        if not docker:
            raise FileNotFoundError("Docker not found")

        docker_image = eval_config["docker"]["image"]

        # Write docker-side config (output_dir → /workspace/results)
        docker_config = dict(eval_config)
        docker_config["output_dir"] = "/workspace/results"
        docker_config.pop("docker", None)

        config_fd, config_path = tempfile.mkstemp(suffix=".yaml", prefix="forge-eval-")
        with os.fdopen(config_fd, "w") as f:
            yaml.safe_dump(docker_config, f)

        # Write types.py patch for TypeAlias compat (Python <3.10)
        types_patch_path = _write_types_patch()

        try:
            cmd = [
                docker,
                "run",
                "--rm",
                "--network",
                "host",
                "-v",
                f"{results_dir}:/workspace/results",
                "-v",
                f"{config_path}:/tmp/eval_config.yaml:ro",
            ]

            # Mount TypeAlias compat patch into the container's conda env
            env_name, py_ver = self._DOCKER_CONDA_ENVS.get(benchmark, ("libero", "3.8"))
            site_pkg = f"/opt/conda/envs/{env_name}/lib/python{py_ver}/site-packages/vla_eval/types.py"
            cmd.extend(["-v", f"{types_patch_path}:{site_pkg}:ro"])

            gpu_request = self._docker_gpu_request()
            if gpu_request is not None:
                cmd.extend(["--gpus", gpu_request])

            cmd.extend(
                [
                    docker_image,
                    "run",
                    "--no-docker",
                    "--config",
                    "/tmp/eval_config.yaml",
                ]
            )

            logger.info("Running Docker: %s", " ".join(cmd))

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,
            )
            if benchmark in CPU_RENDER_FALLBACK_BENCHMARKS and self._is_nvidia_container_runtime_failure(result):
                logger.warning(
                    "NVIDIA Container Toolkit is unavailable; retrying the benchmark "
                    "container with CPU OSMesa rendering while the model server stays on CUDA"
                )
                result = subprocess.run(
                    self._docker_cpu_fallback_command(cmd, docker_image),
                    capture_output=True,
                    text=True,
                    timeout=3600,
                )
            return result
        finally:
            Path(config_path).unlink(missing_ok=True)
            Path(types_patch_path).unlink(missing_ok=True)

    def _docker_gpu_request(self) -> str | None:
        """Map the model server's CUDA selection to Docker's GPU request."""
        if not self.device.startswith("cuda"):
            return None

        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if visible_devices and visible_devices not in {"-1", "NoDevFiles"}:
            visible = [device.strip() for device in visible_devices.split(",") if device.strip()]
            logical_index = 0
            if ":" in self.device:
                logical_index = int(self.device.split(":", maxsplit=1)[1])
            if logical_index >= len(visible):
                raise ValueError(f"CUDA device {self.device} is outside CUDA_VISIBLE_DEVICES={visible_devices!r}")
            # Map the server's logical device back to the exact physical device
            # requested from the benchmark container.
            physical_device = visible[logical_index]
            if physical_device:
                return f"device={physical_device}"

        if ":" in self.device:
            _, index = self.device.split(":", maxsplit=1)
            if index.isdigit():
                return f"device={index}"

        return "all"

    @staticmethod
    def _is_nvidia_container_runtime_failure(result: subprocess.CompletedProcess) -> bool:
        if result.returncode == 0:
            return False
        detail = f"{result.stderr}\n{result.stdout}".lower()
        return "nvidia-container-cli" in detail or "nvml error" in detail

    @staticmethod
    def _docker_cpu_fallback_command(cmd: list[str], docker_image: str) -> list[str]:
        """Drop the GPU request and force deterministic software rendering."""
        fallback = list(cmd)
        if "--gpus" in fallback:
            index = fallback.index("--gpus")
            del fallback[index : index + 2]
        image_index = fallback.index(docker_image)
        fallback[image_index:image_index] = [
            "-e",
            "MUJOCO_GL=osmesa",
            "-e",
            "PYOPENGL_PLATFORM=osmesa",
            "-e",
            "NVIDIA_VISIBLE_DEVICES=void",
        ]
        return fallback

    @staticmethod
    def _is_port_in_use(port: int) -> bool:
        """Check if a TCP port is already bound locally."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(0.1)
                return sock.connect_ex(("127.0.0.1", port)) == 0
        except OSError:
            return False

    @staticmethod
    def _find_open_port() -> int:
        """Find a free local TCP port."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("", 0))
            return sock.getsockname()[1]

    def run_benchmark(
        self,
        benchmark: str,
        config_path: str | None = None,
        episodes_per_task: int = 20,
        max_tasks: int = 10,
    ) -> dict[str, Any]:
        """Run a single benchmark evaluation.

        1. Start ForgeModelServer in background thread
        2. Launch Docker benchmark container via vla-eval protocol
        3. Wait for evaluation to complete
        4. Parse and return results

        Args:
            benchmark: Benchmark name ("libero", "simpler", "vlabench")
            config_path: Optional path to custom benchmark config YAML
            episodes_per_task: Number of episodes per task
            max_tasks: Maximum number of tasks to evaluate

        Returns:
            Dict with benchmark results including success_rate, per_task_rates, etc.
        """
        from forge.eval.model_server import ForgeModelServer
        from forge.eval.results import parse_vla_eval_results

        logger.info("Starting evaluation: %s", benchmark)
        logger.info("  Checkpoint: %s", self.checkpoint_path)
        logger.info("  Variant: %s", self.variant)
        logger.info("  Episodes/task: %d, Max tasks: %d", episodes_per_task, max_tasks)

        # 1. Start model server in background
        server_port = self.port
        if self._is_port_in_use(server_port):
            server_port = self._find_open_port()
            logger.warning("Server port %d unavailable; using fallback port %d", self.port, server_port)

        server = ForgeModelServer(
            checkpoint_path=self.checkpoint_path,
            variant=self.variant,
            model_dir=self.model_dir,
            device=self.device,
            allow_mock=self.allow_mock,
            port=server_port,
        )
        try:
            try:
                server.start(blocking=False, startup_timeout=20.0)
            except (RuntimeError, TimeoutError) as exc:
                return {
                    "benchmark": benchmark,
                    "status": "failed",
                    "error": f"Server startup failed: {exc}",
                }

            benchmark_results_dir = (self.output_dir / benchmark).resolve()
            benchmark_results_dir.mkdir(parents=True, exist_ok=True)
            results_dir = Path(tempfile.mkdtemp(prefix="run-", dir=benchmark_results_dir))

            # 2. Build config and run Docker benchmark
            if config_path:
                with open(config_path) as f:
                    eval_config = yaml.safe_load(f)
            else:
                eval_config = self._build_eval_config(
                    benchmark=benchmark,
                    results_dir=str(results_dir),
                    episodes_per_task=episodes_per_task,
                    max_tasks=max_tasks,
                    server_port=server_port,
                )
            if not isinstance(eval_config, dict):
                raise ValueError("Evaluation config must be a mapping")
            eval_config = self._bind_server_url(eval_config, server_port)

            # 3. Run and wait
            result = self._run_docker_benchmark(eval_config, results_dir, benchmark=benchmark)

            if result.returncode != 0:
                logger.error("Docker exited with code %d", result.returncode)
                logger.error("stderr: %s", result.stderr[:_ERROR_SNIPPET_LIMIT])
                logger.error("stdout: %s", result.stdout[:_ERROR_SNIPPET_LIMIT])
                return {
                    "benchmark": benchmark,
                    "status": "failed",
                    "error": (result.stderr or result.stdout)[:_ERROR_SNIPPET_LIMIT],
                    "returncode": result.returncode,
                }

            # 4. Parse results
            eval_result = parse_vla_eval_results(results_dir, benchmark, self.variant, self.checkpoint_path)
            if eval_result.status != "completed":
                container_output = "\n".join(
                    part.strip() for part in (result.stderr, result.stdout) if part and part.strip()
                )
                if container_output:
                    eval_result.error = (
                        f"{eval_result.error}\nContainer output:\n{container_output[-_ERROR_SNIPPET_LIMIT:]}"
                    ).strip()
            return eval_result.to_dict()

        except subprocess.TimeoutExpired:
            logger.error("Benchmark %s timed out after 1 hour", benchmark)
            return {"benchmark": benchmark, "status": "timeout", "error": "Exceeded 1 hour timeout"}
        except FileNotFoundError:
            logger.error("Docker not found. Install Docker to run benchmarks.")
            return {"benchmark": benchmark, "status": "failed", "error": "Docker not installed"}
        finally:
            server.stop()

    def run_all(
        self,
        benchmarks: list[str] | None = None,
        episodes_per_task: int = 20,
        max_tasks: int = 10,
    ) -> list[dict[str, Any]]:
        """Run all configured benchmarks sequentially.

        Args:
            benchmarks: List of benchmarks to run. Defaults to all 3.
            episodes_per_task: Number of episodes per task.
            max_tasks: Maximum number of tasks to evaluate.

        Returns:
            List of result dicts, one per benchmark.
        """
        benchmarks = benchmarks or list(BENCHMARK_IMAGES.keys())
        results = []
        for bench in benchmarks:
            logger.info("\n" + "=" * 60)
            logger.info("Running benchmark: %s", bench)
            logger.info("=" * 60)
            try:
                result = self.run_benchmark(
                    bench,
                    episodes_per_task=episodes_per_task,
                    max_tasks=max_tasks,
                )
            except Exception as exc:
                logger.error("Benchmark %s failed unexpectedly: %s", bench, exc)
                logger.exception("Benchmark %s exception details:", bench)
                result = {
                    "benchmark": bench,
                    "status": "failed",
                    "error": str(exc),
                }
            results.append(result)

            # Save intermediate results
            results_file = self.output_dir / "all_results.json"
            results_file.write_text(json.dumps(results, indent=2, default=str))

        return results

    def compare(
        self,
        other_checkpoint: str,
        benchmark: str = "libero",
        episodes_per_task: int = 20,
        max_tasks: int = 10,
    ) -> dict[str, Any]:
        """Compare two checkpoints on the same benchmark.

        Returns:
            Dict with comparison results including delta_success_rate.
        """
        logger.info("Comparing checkpoints on %s:", benchmark)
        logger.info("  A: %s", self.checkpoint_path)
        logger.info("  B: %s", other_checkpoint)

        result_a = self.run_benchmark(benchmark, episodes_per_task=episodes_per_task, max_tasks=max_tasks)

        runner_b = EvalRunner(
            checkpoint_path=other_checkpoint,
            variant=self.variant,
            model_dir=self.model_dir,
            device=self.device,
            output_dir=str(self.output_dir),
            port=self.port + 1,
            allow_mock=self.allow_mock,
        )
        result_b = runner_b.run_benchmark(benchmark, episodes_per_task=episodes_per_task, max_tasks=max_tasks)

        failed = [result for result in (result_a, result_b) if result.get("status", "completed") != "completed"]
        if failed:
            errors = [str(result.get("error") or result.get("status")) for result in failed]
            return {
                "benchmark": benchmark,
                "checkpoint_a": self.checkpoint_path,
                "checkpoint_b": other_checkpoint,
                "status": "failed",
                "error": "; ".join(errors),
                "result_a": result_a,
                "result_b": result_b,
            }

        sr_a = result_a.get("success_rate", 0.0)
        sr_b = result_b.get("success_rate", 0.0)

        return {
            "benchmark": benchmark,
            "checkpoint_a": self.checkpoint_path,
            "checkpoint_b": other_checkpoint,
            "success_rate_a": sr_a,
            "success_rate_b": sr_b,
            "delta_success_rate": sr_a - sr_b,
            "result_a": result_a,
            "result_b": result_b,
        }

    @staticmethod
    def check_docker() -> bool:
        """Check if Docker is available."""
        return shutil.which("docker") is not None

    @staticmethod
    def pull_images() -> dict[str, str]:
        """Pull all benchmark Docker images."""
        results = {}
        for name, image in BENCHMARK_IMAGES.items():
            logger.info("Pulling %s: %s", name, image)
            try:
                subprocess.run(
                    ["docker", "pull", image],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=1800,
                )
                results[name] = "ok"
            except subprocess.CalledProcessError as e:
                results[name] = f"failed: {e.stderr[:200]}"
            except FileNotFoundError:
                results[name] = "docker not installed"
                break
        return results
