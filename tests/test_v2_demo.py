"""PRD-18: VC Demo Dashboard & HTML Report tests."""

from forge.config import ForgeConfig
from forge.demo.report import _generate_architecture_svg, generate_html_report
from forge.demo.runner import DemoRunner


def test_demo_runner_init():
    """DemoRunner initializes with config and device."""
    config = ForgeConfig.default()
    runner = DemoRunner(config, device="cpu")
    assert runner.config is config
    assert runner.device == "cpu"


def test_demo_runner_collects_teachers():
    """DemoRunner._get_teacher_info returns list (may be empty without teachers)."""
    config = ForgeConfig.default()
    runner = DemoRunner(config, device="cpu")
    teachers = runner._get_teacher_info()
    assert isinstance(teachers, list)


def test_demo_runner_collects_embodiments():
    """DemoRunner._get_embodiment_info returns embodiment profiles."""
    config = ForgeConfig.default()
    runner = DemoRunner(config, device="cpu")
    embodiments = runner._get_embodiment_info()
    assert isinstance(embodiments, list)
    # Should find builtin profiles from embodiment registry
    if embodiments:
        assert "name" in embodiments[0]
        assert "dof" in embodiments[0]


def test_html_report_contains_key_sections():
    """HTML report contains all required sections."""
    data = {
        "benchmark": {
            "model_name": "FORGE-nano",
            "latency": {"mean_ms": 18.5, "p95_ms": 22.0},
            "throughput": {"actions_per_second": 1200, "chunk_gain": 8},
            "compression": {"compression_ratio": 15.0, "model_size_mb": 480},
        },
        "teachers": [],
        "embodiments": [],
        "architecture": {
            "pipeline": "Teacher Labels → Multi-KD → Chunk Compress → Flow Export",
        },
        "version": "2.0.0",
    }

    html = generate_html_report(data)

    # Hero section
    assert "FORGE v3" in html
    assert "VLA Distillation and Deployment Report" in html

    # Key numbers
    assert "15x" in html  # compression ratio
    assert "1200" in html  # actions/sec
    assert "480MB" in html  # model size

    # Sections
    assert "Architecture" in html
    assert "Universal Teacher Support" in html
    assert "Robot Embodiment Profiles" in html
    assert "Distillation Pipeline" in html

    # Pipeline stages
    assert "Multi-Teacher Labels" in html
    assert "Multi-Path KD" in html
    assert "Chunk Compression" in html
    assert "Edge Export" in html


def test_html_report_valid_html():
    """HTML report is valid self-contained HTML."""
    data = {
        "benchmark": {
            "latency": {"mean_ms": 10},
            "throughput": {"actions_per_second": 500, "chunk_gain": 4},
            "compression": {"compression_ratio": 10, "model_size_mb": 200},
        },
        "teachers": [
            {
                "name": "OpenVLA",
                "architecture": "token-AR",
                "params_b": 7.6,
                "supports_chunking": False,
            }
        ],
        "embodiments": [{"name": "franka", "dof": 7}],
        "architecture": {},
        "version": "2.0.0",
    }

    html = generate_html_report(data)

    assert html.startswith("<!DOCTYPE html>")
    assert "<html" in html
    assert "</html>" in html
    assert "<head>" in html
    assert "</head>" in html
    assert "<body>" in html
    assert "</body>" in html
    assert "<style>" in html

    # Teacher data rendered
    assert "OpenVLA" in html
    assert "7.6B" in html

    # Embodiment data rendered
    assert "franka" in html
    assert "7-DoF" in html


def test_architecture_svg_renders():
    """Architecture SVG contains all pipeline stages."""
    svg = _generate_architecture_svg()

    assert "<svg" in svg
    assert "</svg>" in svg
    assert "viewBox" in svg

    # Pipeline stages in SVG
    assert "Teacher VLA" in svg
    assert "FORGE Student" in svg
    assert "Compressed" in svg
    assert "Edge" in svg

    # Feature labels
    assert "Multi-Teacher" in svg
    assert "Action Chunking" in svg
    assert "Flow Matching" in svg
    assert "Async Runtime" in svg

    # Arrow markers
    assert "marker" in svg
    assert "arrow" in svg


def test_demo_report_never_invents_missing_registry_rows():
    html = generate_html_report(
        {
            "benchmark": {},
            "teachers": [],
            "embodiments": [],
            "architecture": {},
            "provenance": {"vision": "mock", "language": "mock", "labels": "mock"},
        }
    )

    assert "No teacher registry metadata was available" in html
    assert "No embodiment registry metadata was available" in html
    assert "Franka Panda" not in html
    assert "OpenVLA-7B (token-AR)" not in html
    assert '<div class="value">1200</div>' not in html
    assert '<div class="value">480MB</div>' not in html
    assert "Inference Latency (L4)" not in html
    assert "Benchmark input is not verified as real" in html


def test_demo_runner_requires_verified_checkpoint_model(tmp_path):
    config = ForgeConfig.default()
    runner = DemoRunner(config, device="cpu")

    import pytest

    with pytest.raises(ValueError, match="provenance-verified trained model"):
        runner.run(
            output_path=str(tmp_path / "report.html"),
            images=object(),
            language_text="move",
            input_provenance={"kind": "real"},
        )
