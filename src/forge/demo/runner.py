"""Demo runner — orchestrates benchmark + report generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from forge.config import ForgeConfig


class DemoRunner:
    """Run full demo pipeline and generate report.

    Steps:
    1. Initialize student model
    2. Run benchmark suite
    3. Collect system info
    4. Generate HTML report
    """

    def __init__(
        self,
        config: ForgeConfig,
        device: str = "cuda",
        *,
        model: Any | None = None,
        provenance: dict[str, str] | None = None,
        source_checkpoint: str | None = None,
    ):
        self.config = config
        self.device = device
        self.model = model
        self.provenance = provenance
        self.source_checkpoint = source_checkpoint

    def run(
        self,
        output_path: str = "forge_v3_report.html",
        *,
        images: object,
        language_text: str,
        input_provenance: dict[str, object],
        samples: int = 30,
        duration: float = 2.0,
    ) -> dict:
        """Run demo and generate report."""
        from forge import __version__
        from forge.benchmark.runner import BenchmarkRunner
        from forge.demo.report import generate_html_report

        if self.model is None or self.provenance is None:
            raise ValueError("DemoRunner requires a provenance-verified trained model")
        student = self.model
        student = student.to(self.device)
        student.eval()

        # Run benchmarks
        runner = BenchmarkRunner(student, self.config, device=self.device)
        report = runner.run(
            n_latency_samples=samples,
            throughput_duration=duration,
            images=images,
            language_text=language_text,
            input_provenance=input_provenance,
        )
        report.provenance = dict(self.provenance)
        report.source_checkpoint = self.source_checkpoint

        # Collect extra info
        demo_data = {
            "benchmark": report.to_dict(),
            "teachers": self._get_teacher_info(),
            "embodiments": self._get_embodiment_info(),
            "architecture": self._get_architecture_info(),
            "provenance": dict(self.provenance),
            "version": __version__,
        }

        # Generate HTML
        html = generate_html_report(demo_data)
        Path(output_path).write_text(html)

        return demo_data

    def _get_teacher_info(self) -> list[dict]:
        """Get info about available teachers."""
        try:
            from forge.teachers.registry import get_registry

            registry = get_registry()
            teachers = []
            for name in registry.list_teachers():
                adapter = registry.create(name)
                info = adapter.info()
                teachers.append(
                    {
                        "name": info.name,
                        "architecture": info.architecture,
                        "params_b": info.param_count,
                        "supports_chunking": info.supports_chunking,
                    }
                )
            return teachers
        except Exception:
            return []

    def _get_embodiment_info(self) -> list[dict]:
        """Get info about supported embodiments."""
        try:
            from forge.embodiments.registry import EmbodimentRegistry

            registry = EmbodimentRegistry()
            return [{"name": name, "dof": registry.get(name).dof} for name in registry.list_embodiments()]
        except Exception:
            return []

    def _get_architecture_info(self) -> dict:
        """Get architecture pipeline info."""
        return {
            "pipeline": "Teacher Labels → Multi-KD → Chunk Compress → Flow Export",
            "student_variants": [
                "FORGE-Nano (0.5B)",
                "FORGE-Small (1.5B)",
                "FORGE-Micro (0.2B)",
            ],
            "action_heads": [
                "Diffusion (DDPM)",
                "Flow Matching (1-4 step)",
                "Consistency (1 step)",
            ],
            "export_targets": [
                "TensorRT (Jetson)",
                "CoreML (Apple)",
                "MLX (Apple Silicon)",
                "ONNX",
            ],
        }
