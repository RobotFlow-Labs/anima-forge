"""PRD-20: Web API endpoint tests."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from forge import __version__
from forge.config import ForgeConfig
from forge.web.api import create_app
from forge.web.state import ServerState


@pytest.fixture(autouse=True)
def reset_state():
    """Reset server state between tests."""
    ServerState.reset()
    yield
    ServerState.reset()


@pytest.fixture
def client():
    config = ForgeConfig.default()
    app = create_app(config)
    return TestClient(app)


def test_api_status_returns_valid_json(client):
    """GET /api/status returns system status."""
    r = client.get("/api/status")
    assert r.status_code == 200
    data = r.json()
    assert "gpu" in data
    assert "vram_total_gb" in data
    assert "uptime_s" in data
    assert "version" in data
    assert data["version"] == __version__


def test_api_teachers_list(client):
    """GET /api/teachers returns teacher list."""
    r = client.get("/api/teachers")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    if data:
        assert "name" in data[0]
        assert "architecture" in data[0]


def test_api_config_get_put(client):
    """GET /api/config returns config dict."""
    r = client.get("/api/config")
    assert r.status_code == 200
    data = r.json()
    assert "student" in data
    assert "web" in data
    assert "paths" in data

    update = client.put("/api/config", json={"student": {"variant": "nano"}})
    assert update.status_code == 409
    assert "forge config init" in update.json()["hint"]


def test_api_benchmarks_list(client):
    """GET /api/benchmarks returns history list."""
    r = client.get("/api/benchmarks")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_api_embodiments_list(client):
    """GET /api/embodiments returns embodiment profiles."""
    r = client.get("/api/embodiments")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    if data:
        assert "name" in data[0]
        assert "dof" in data[0]


@pytest.mark.parametrize(
    ("path", "hint"),
    [
        ("/api/train/start", "forge train start"),
        ("/api/compress/start", "--data-dir <real-label-root>"),
        ("/api/benchmarks/run", "--data-dir <real-lerobot-dataset>"),
        ("/api/runtime/start", "forge serve"),
        ("/api/demo/run", "forge demo"),
        ("/api/predict", "forge serve"),
    ],
)
def test_api_write_facades_are_rejected_with_real_cli_hint(client, path, hint):
    response = client.post(path)
    assert response.status_code == 409
    assert response.json()["status"] == "input_required"
    assert hint in response.json()["hint"]
