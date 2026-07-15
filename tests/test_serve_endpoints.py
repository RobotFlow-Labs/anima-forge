"""Endpoint-level contract tests for the maintained FORGE inference server."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image
from torch import nn

import forge.serve as serve_module


def _provenance() -> dict[str, str]:
    return {
        "vision": "real",
        "language": "real",
        "labels": "real",
        "model_dir": "test-assets",
        "git_sha": "a" * 40,
        "forge_version": "3.0.1",
        "torch_version": torch.__version__,
    }


def _checkpoint_payload() -> dict[str, Any]:
    return {
        "model_state_dict": {"weight": torch.ones(1)},
        "student_config": {
            "variant": "micro",
            "action_horizon": 2,
            "action_dim": 3,
        },
        "provenance": _provenance(),
    }


def _png_bytes(color: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), color=color).save(buffer, format="PNG")
    return buffer.getvalue()


class _Tokenizer:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, instructions: list[str], **_kwargs: object) -> dict[str, torch.Tensor]:
        self.calls.append(list(instructions))
        return {"input_ids": torch.ones((len(instructions), 2), dtype=torch.long)}


class _TinyStudent(nn.Module):
    instances: list[_TinyStudent] = []

    def __init__(self, config: Any, model_dir: str | None = None) -> None:
        super().__init__()
        del model_dir
        self.config = config
        self.weight = nn.Parameter(torch.zeros(1))
        self.tokenizer = _Tokenizer()
        self.component_provenance = {"vision": "real", "language": "real"}
        self.output_override: torch.Tensor | None = None
        self.__class__.instances.append(self)

    @property
    def total_params(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, images: torch.Tensor, *, language_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        assert language_ids.shape[0] == images.shape[0]
        if self.output_override is not None:
            return {"actions": self.output_override.to(images.device)}
        values = torch.arange(
            images.shape[0] * self.config.action_horizon * self.config.action_dim,
            dtype=torch.float32,
            device=images.device,
        )
        return {"actions": values.reshape(images.shape[0], self.config.action_horizon, self.config.action_dim)}


@pytest.fixture
def served_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    checkpoint = tmp_path / "trained.pt"
    checkpoint.write_bytes(b"verified checkpoint placeholder")
    payload = _checkpoint_payload()
    _TinyStudent.instances.clear()
    monkeypatch.setattr("forge.student.FORGEStudent", _TinyStudent)
    monkeypatch.setattr(serve_module, "_load_checkpoint_payload", lambda *_args, **_kwargs: payload)

    app = serve_module.create_app(checkpoint=str(checkpoint), model_dir=str(tmp_path), device="cpu")
    return app, _TinyStudent.instances[-1]


def test_serve_requires_checkpoint_for_every_python_entrypoint() -> None:
    with pytest.raises(TypeError):
        serve_module.create_app()  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        serve_module.start_server()  # type: ignore[call-arg]


def test_serve_rejects_checkpoint_without_verified_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "legacy.pt"
    checkpoint.write_bytes(b"legacy")
    payload = _checkpoint_payload()
    payload.pop("provenance")
    monkeypatch.setattr(serve_module, "_load_checkpoint_payload", lambda *_args, **_kwargs: payload)

    with pytest.raises(ValueError, match="checkpoint provenance"):
        serve_module.create_app(checkpoint=str(checkpoint), device="cpu")


def test_health_reports_checkpoint_derived_runtime_contract(served_app) -> None:
    app, _student = served_app
    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "version": "3.0.1",
        "model": "FORGE-micro",
        "variant": "micro",
        "action_horizon": 2,
        "action_dim": 3,
        "checkpoint": "trained.pt",
        "device": "cpu",
        "params_M": 0.0,
        "provenance": {"vision": "real", "language": "real"},
    }
    public_paths = {route.path for route in app.routes if route.path in {"/health", "/predict", "/batch_predict"}}
    assert public_paths == {"/health", "/predict", "/batch_predict"}
    assert not any(route.path in {"/status", "/stream", "/info"} for route in app.routes)


def test_predict_accepts_real_multipart_image_and_instruction(served_app) -> None:
    app, student = served_app
    response = TestClient(app).post(
        "/predict",
        files={"image": ("frame.png", _png_bytes((255, 0, 0)), "image/png")},
        data={"instruction": "  pick up the red block  "},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["actions"] == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
    assert body["action_horizon"] == 2
    assert body["action_dim"] == 3
    assert body["instruction"] == "pick up the red block"
    assert body["model"] == "FORGE-micro"
    assert body["version"] == "3.0.1"
    assert student.tokenizer.calls == [["pick up the red block"]]


def test_batch_predict_accepts_repeated_images_and_forwards_each_instruction(served_app) -> None:
    app, student = served_app
    response = TestClient(app).post(
        "/batch_predict",
        files=[
            ("images", ("first.png", _png_bytes((255, 0, 0)), "image/png")),
            ("images", ("second.png", _png_bytes((0, 255, 0)), "image/png")),
        ],
        data={"instruction": "close the drawer"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["batch_size"] == 2
    assert len(body["actions"]) == 2
    assert all(len(chunk) == 2 and all(len(action) == 3 for action in chunk) for chunk in body["actions"])
    assert student.tokenizer.calls == [["close the drawer", "close the drawer"]]


@pytest.mark.parametrize(
    "actions",
    [
        torch.zeros((1, 2, 4)),
        torch.tensor([[[float("nan"), 0.0, 0.0], [0.0, 0.0, 0.0]]]),
    ],
)
def test_predict_rejects_wrong_shape_or_nonfinite_actions(served_app, actions: torch.Tensor) -> None:
    app, student = served_app
    student.output_override = actions
    response = TestClient(app, raise_server_exceptions=False).post(
        "/predict",
        files={"image": ("frame.png", _png_bytes((0, 0, 255)), "image/png")},
        data={"instruction": "move"},
    )

    assert response.status_code == 500


@pytest.mark.parametrize("path", ["/predict", "/batch_predict"])
def test_prediction_endpoints_require_instruction(served_app, path: str) -> None:
    app, _student = served_app
    field = "image" if path == "/predict" else "images"
    response = TestClient(app).post(
        path,
        files={field: ("frame.png", _png_bytes((0, 0, 0)), "image/png")},
    )

    assert response.status_code == 422
