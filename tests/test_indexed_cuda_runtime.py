"""Mocked multi-GPU contracts for explicit ``cuda:N`` runtime selection."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


class _DeviceContext:
    def __init__(self, device: object, calls: list[str]) -> None:
        self.device = str(device)
        self.calls = calls

    def __enter__(self) -> None:
        self.calls.append(self.device)

    def __exit__(self, *_args: object) -> None:
        return None


def test_pipeline_and_matrix_preserve_indexed_cuda_device() -> None:
    from forge.benchmark.matrix import _validated_backend_result
    from forge.pipeline import _normalize_device

    assert _normalize_device("cuda:1") == "cuda:1"
    result = _validated_backend_result(
        {
            "status": "success",
            "device": "cuda:1",
            "execution": {"requested_device": "cuda:1", "resolved_device": "cuda:1"},
            "actions_finite": True,
            "actions_shape": [1, 7],
            "action_samples": 1,
            "input_provenance": {"kind": "real"},
        },
        target="PyTorch",
    )

    assert result["status"] == "success"
    assert result["device"] == "cuda:1"


def test_latency_sync_targets_selected_second_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    from forge.benchmark import metrics

    synchronized: list[str] = []

    class Tensor:
        def to(self, _device: str):
            return self

    class Model:
        def to(self, device: str):
            assert device == "cuda:1"
            return self

        def eval(self) -> None:
            return None

        def __call__(self, _input: object) -> object:
            return object()

    monkeypatch.setattr(metrics.torch, "randn", lambda *_args, **_kwargs: Tensor())
    monkeypatch.setattr(metrics.torch.cuda, "synchronize", lambda device: synchronized.append(str(device)))

    result = metrics.profile_latency(Model(), input_shape=(1,), n_warmup=1, n_samples=2, device="cuda:1")

    assert result.samples == 2
    assert synchronized == ["cuda:1"] * 5


def test_training_memory_snapshot_reads_selected_second_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    from forge import training_runtime

    contexts: list[str] = []
    allocated_devices: list[int] = []
    reserved_devices: list[int] = []
    monkeypatch.setattr(training_runtime.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        training_runtime.torch.cuda,
        "device",
        lambda device: _DeviceContext(device, contexts),
    )
    monkeypatch.setattr(training_runtime.torch.cuda, "mem_get_info", lambda: (6 * 1024**3, 8 * 1024**3))
    monkeypatch.setattr(
        training_runtime.torch.cuda,
        "memory_allocated",
        lambda device: allocated_devices.append(device) or 2 * 1024**3,
    )
    monkeypatch.setattr(
        training_runtime.torch.cuda,
        "memory_reserved",
        lambda device: reserved_devices.append(device) or 4 * 1024**3,
    )

    snapshot = training_runtime.cuda_memory_snapshot("cuda:1")

    assert contexts == ["1"]
    assert allocated_devices == [1]
    assert reserved_devices == [1]
    assert snapshot["allocated_gib"] == 2.0
    assert snapshot["reserved_gib"] == 4.0


def test_backend_uses_indexed_cuda_dtype_and_properties(monkeypatch: pytest.MonkeyPatch) -> None:
    from forge.backend import TorchBackend

    property_devices: list[str] = []
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: (
            property_devices.append(str(device))
            or SimpleNamespace(name="GPU 1", total_memory=16 * 1024**3, major=8, minor=9)
        ),
    )

    backend = TorchBackend("cuda:1")
    info = backend.get_device_info()

    assert backend.dtype is torch.bfloat16
    assert property_devices == ["cuda:1"]
    assert info.device_name == "GPU 1"


def test_tensorrt_wrappers_bind_selected_second_gpu(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from forge.export import tensorrt_export

    contexts: list[str] = []
    observed: list[tuple[str, str]] = []
    engine = tmp_path / "forge.engine"
    engine.write_bytes(b"engine")
    onnx = tmp_path / "forge.onnx"
    onnx.write_bytes(b"onnx")

    monkeypatch.setattr(tensorrt_export.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        tensorrt_export.torch.cuda,
        "device",
        lambda device: _DeviceContext(device, contexts),
    )
    monkeypatch.setattr(
        tensorrt_export,
        "_benchmark_tensorrt_runtime_on_selected_device",
        lambda *_args, device, **_kwargs: observed.append(("benchmark", device)) or {"status": "success"},
    )
    monkeypatch.setattr(
        tensorrt_export,
        "_export_tensorrt_on_selected_device",
        lambda *_args, device, **_kwargs: observed.append(("export", device)) or engine,
    )

    benchmark = tensorrt_export.benchmark_tensorrt_runtime(engine, device="cuda:1")
    exported = tensorrt_export.export_tensorrt(onnx, engine, device="cuda:1")

    assert benchmark["status"] == "success"
    assert exported == engine
    assert contexts == ["cuda:1", "cuda:1"]
    assert observed == [("benchmark", "cuda:1"), ("export", "cuda:1")]
