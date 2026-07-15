"""FastAPI app — all /api/* endpoints + dashboard serving."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from importlib.resources import files
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

from forge.config import ForgeConfig
from forge.web.state import ServerState
from forge.web.websockets import metrics_manager, train_manager

logger = logging.getLogger(__name__)


def create_app(config: ForgeConfig | None = None) -> FastAPI:
    """Create FastAPI application with all endpoints."""
    config = config or ForgeConfig.default()
    state = ServerState()
    state.set_ready(False, "initializing")

    from forge import __version__

    app = FastAPI(title="FORGE Command Center", version=__version__)

    def cli_only(operation: str, command: str) -> JSONResponse:
        return JSONResponse(
            {
                "status": "input_required",
                "error": f"{operation} needs explicit artifact and data paths.",
                "hint": command,
            },
            status_code=409,
        )

    # CORS middleware
    from fastapi.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.web.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Dashboard ──────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    async def dashboard():
        """Serve the dashboard HTML."""
        try:
            html = files("forge.web").joinpath("dashboard.html").read_text(encoding="utf-8")
        except (FileNotFoundError, ModuleNotFoundError):
            return HTMLResponse("<h1>FORGE Command Center</h1><p>Dashboard not found.</p>", status_code=500)
        return HTMLResponse(html)

    # ── System ─────────────────────────────────────────────

    @app.get("/api/runtime/ready")
    async def api_runtime_ready():
        """Return API readiness independent of runtime state."""
        return {
            "api_ready": state.is_ready,
            "api_ready_reason": state.get_system_status()["api_ready_reason"],
        }

    @app.get("/api/status")
    async def api_status():
        return state.get_system_status()

    @app.get("/api/config")
    async def api_config_get():
        return _config_to_dict(config)

    @app.put("/api/config")
    async def api_config_put(updates: dict):
        return cli_only("Configuration updates", "forge config init > forge.yaml")

    # ── Teachers ───────────────────────────────────────────

    @app.get("/api/teachers")
    async def api_teachers_list():
        try:
            from forge.teachers.registry import get_registry

            registry = get_registry()
            teachers = []
            for name in registry.list_teachers():
                adapter = registry.create(name)
                info = adapter.info()
                loaded = name in state.loaded_teachers
                teachers.append(
                    {
                        "name": info.name,
                        "architecture": info.architecture,
                        "params_b": info.param_count,
                        "supports_chunking": info.supports_chunking,
                        "loaded": loaded,
                    }
                )
            return teachers
        except Exception as e:
            logger.warning(f"Teacher registry error: {e}")
            return []

    @app.post("/api/teachers/{name}/load")
    async def api_teacher_load(name: str):
        try:
            from forge.teachers.registry import get_registry

            registry = get_registry()
            adapter = registry.create(name)
            # Use model path from config
            model_path = Path(config.paths.model_dir) / name
            adapter.load(model_path)
            state.loaded_teachers[name] = {"adapter": adapter}
            return {"status": "loaded", "name": name}
        except Exception:
            logger.exception("Failed to load teacher")
            return JSONResponse({"status": "error", "error": "Failed to load teacher"}, status_code=500)

    @app.post("/api/teachers/{name}/unload")
    async def api_teacher_unload(name: str):
        if name in state.loaded_teachers:
            entry = state.loaded_teachers.pop(name)
            adapter = entry.get("adapter")
            if adapter and hasattr(adapter, "unload"):
                adapter.unload()
        return {"status": "unloaded", "name": name}

    # ── Models ─────────────────────────────────────────────

    @app.get("/api/models")
    async def api_models_list():
        models = []
        output_dir = Path(config.paths.output_dir)
        if output_dir.exists():
            for p in output_dir.glob("*.pt"):
                models.append(
                    {
                        "name": p.stem,
                        "size_mb": round(p.stat().st_size / (1024 * 1024), 1),
                    }
                )
        return models

    # ── Training ───────────────────────────────────────────

    @app.post("/api/train/start")
    async def api_train_start():
        return cli_only(
            "Training",
            "forge train start --config forge.yaml --data-dir <labels> --output-dir <run>",
        )

    @app.post("/api/train/stop")
    async def api_train_stop():
        return cli_only("Stopping training", "forge train stop <run-id>")

    @app.get("/api/train/status")
    async def api_train_status():
        return state.train_state

    @app.websocket("/api/train/stream")
    async def ws_train_stream(websocket: WebSocket):
        await train_manager.connect(websocket)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            train_manager.disconnect(websocket)

    # ── Compression ────────────────────────────────────────

    @app.post("/api/compress/start")
    async def api_compress_start():
        return cli_only(
            "Compression",
            "forge pipeline --stage compress --checkpoint <checkpoint.pt> "
            "--data-dir <real-label-root> --output-dir <run>",
        )

    @app.get("/api/compress/status")
    async def api_compress_status():
        return {"running": False, "stage": "idle", "progress_pct": 0}

    # ── Benchmarks ─────────────────────────────────────────

    @app.post("/api/benchmarks/run")
    async def api_benchmarks_run():
        return cli_only(
            "Benchmarking",
            "forge benchmark run --checkpoint <checkpoint.pt> "
            "--data-dir <real-lerobot-dataset> --instruction '<real-task>'",
        )

    @app.get("/api/benchmarks")
    async def api_benchmarks_list():
        return state.benchmark_history

    @app.get("/api/benchmarks/{benchmark_id}")
    async def api_benchmark_get(benchmark_id: str):
        for b in state.benchmark_history:
            if b.get("id") == benchmark_id:
                return b
        return JSONResponse({"error": "not found"}, status_code=404)

    # ── Embodiments ────────────────────────────────────────

    @app.get("/api/embodiments")
    async def api_embodiments_list():
        try:
            from forge.embodiments.registry import EmbodimentRegistry

            registry = EmbodimentRegistry()
            return [
                {
                    "name": name,
                    "dof": registry.get(name).dof,
                    "action_dim": registry.get(name).action_dim,
                }
                for name in registry.list_embodiments()
            ]
        except Exception:
            return []

    @app.get("/api/embodiments/{name}")
    async def api_embodiment_get(name: str):
        try:
            from forge.embodiments.registry import EmbodimentRegistry

            registry = EmbodimentRegistry()
            profile = registry.get(name)
            return asdict(profile)
        except Exception:
            return JSONResponse({"error": "Embodiment not found"}, status_code=404)

    @app.get("/api/embodiments/{name}/config")
    async def api_embodiment_config(name: str):
        try:
            from forge.embodiments.registry import EmbodimentRegistry

            registry = EmbodimentRegistry()
            yaml_str = registry.generate_yaml_config(name)
            return {"name": name, "yaml": yaml_str}
        except Exception:
            return JSONResponse({"error": "Embodiment not found"}, status_code=404)

    # ── Inference ──────────────────────────────────────────

    @app.post("/api/predict")
    async def api_predict():
        return cli_only("Inference", "forge serve --checkpoint <checkpoint.pt>")

    @app.get("/api/runtime/status")
    async def api_runtime_status():
        return {"is_running": False, "frames_processed": 0, "actions_served": 0}

    @app.post("/api/runtime/start")
    async def api_runtime_start():
        return cli_only("Runtime startup", "forge serve --checkpoint <checkpoint.pt>")

    @app.post("/api/runtime/stop")
    async def api_runtime_stop():
        return cli_only("Runtime shutdown", "forge train stop <run-id>")

    @app.websocket("/api/stream")
    async def ws_inference_stream(websocket: WebSocket):
        await metrics_manager.connect(websocket)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            metrics_manager.disconnect(websocket)

    # ── Experiments ────────────────────────────────────

    @app.get("/api/experiments/auto_hp")
    async def api_experiments_auto_hp():
        """Load auto-HP results from outputs/auto_hp/auto_hp_results.json."""
        results_path = Path(config.paths.output_dir) / "auto_hp" / "auto_hp_results.json"
        if not results_path.exists():
            return {"status": "no_data", "trials": []}
        try:
            data = json.loads(results_path.read_text())
            return data
        except Exception:
            logger.exception("Failed to load experiment data")
            return JSONResponse({"error": "Failed to load experiment data"}, status_code=500)

    # ── VLA Eval (PRD-32) ────────────────────────────────

    @app.get("/api/eval/results")
    async def api_eval_results():
        """Load eval results from outputs/eval/."""
        try:
            from forge.eval.results import load_results

            results = load_results(Path(config.paths.output_dir) / "eval")
            return [r.to_dict() for r in results]
        except Exception:
            logger.exception("Failed to load eval results")
            return []

    # ── Demo ───────────────────────────────────────────────

    @app.post("/api/demo/run")
    async def api_demo_run():
        return cli_only("Demo generation", "forge demo --help")

    @app.get("/api/demo/report")
    async def api_demo_report():
        from forge.demo.report import generate_html_report

        data = {
            "benchmark": {},
            "teachers": await api_teachers_list(),
            "embodiments": await api_embodiments_list(),
            "architecture": {},
            "version": __version__,
        }
        return HTMLResponse(generate_html_report(data))

    state.set_ready(True, "running")
    return app


def _config_to_dict(config: ForgeConfig) -> dict:
    """Convert ForgeConfig to JSON-safe dict."""
    return {
        "paths": asdict(config.paths),
        "student": asdict(config.student),
        "vision": asdict(config.vision),
        "distill": asdict(config.distill),
        "pruning": asdict(config.pruning),
        "quant": asdict(config.quant),
        "export": asdict(config.export),
        "web": asdict(config.web),
        "universal": asdict(config.universal),
        "curriculum": asdict(config.curriculum),
    }
