"""PRD-20: Dashboard HTML validation tests."""

from __future__ import annotations

from pathlib import Path

from typer.main import get_command

from forge.cli_v2 import app

DASHBOARD_PATH = Path(__file__).parent.parent / "src" / "forge" / "web" / "dashboard.html"


def test_dashboard_html_exists():
    """Dashboard HTML file exists."""
    assert DASHBOARD_PATH.exists(), f"Dashboard not found at {DASHBOARD_PATH}"
    content = DASHBOARD_PATH.read_text()
    assert len(content) > 1000
    assert "<!DOCTYPE html>" in content


def test_dashboard_contains_all_routes():
    """Dashboard contains every live page route."""
    content = DASHBOARD_PATH.read_text()
    routes = [
        "page-status",
        "page-teachers",
        "page-models",
        "page-distill",
        "page-compress",
        "page-benchmarks",
        "page-robots",
        "page-inference",
        "page-export",
        "page-eval",
        "page-experiments",
    ]
    for route in routes:
        assert route in content, f"Missing page: {route}"


def test_dashboard_does_not_call_nonfunctional_write_endpoints():
    content = DASHBOARD_PATH.read_text()
    for endpoint in (
        "/api/train/start",
        "/api/compress/start",
        "/api/benchmarks/run",
        "/api/runtime/start",
        "/api/demo/run",
    ):
        assert f"API.post('{endpoint}')" not in content
    assert "SHOW TRAIN COMMAND" in content
    assert "SHOW BENCHMARK COMMAND" in content


def test_dashboard_never_substitutes_hardcoded_benchmark_claims():
    """Loss charts must come from live benchmark records or render an honest empty state."""
    content = DASHBOARD_PATH.read_text()
    assert "await API.get('/api/benchmarks')" in content
    assert "No measured reduction is displayed" in content
    assert "RAW_LOSS_" not in content
    assert "hardcoded benchmark data" not in content
    assert "diff+p75+INT4" not in content


def test_dashboard_contains_design_system():
    """Dashboard uses SACRED design system (industrial cyberpunk)."""
    content = DASHBOARD_PATH.read_text()
    # Required CSS variables
    assert "#FF3B00" in content  # orange
    assert "#050505" in content  # black
    assert "Oswald" in content  # headline font
    assert "JetBrains Mono" in content  # data font
    assert "uppercase" in content  # headlines uppercase
    # No forbidden elements
    assert "border-radius" not in content  # NO rounded corners


def test_demo_cli_requires_checkpoint_and_real_input_instead_of_ignored_steps():
    demo = get_command(app).commands["demo"]
    assert callable(getattr(demo, "list_commands", None))
    option_names = {option for parameter in demo.params for option in getattr(parameter, "opts", ())}

    assert {"--checkpoint", "--data-dir", "--instruction", "--samples"} <= option_names
    assert "--steps" not in option_names
