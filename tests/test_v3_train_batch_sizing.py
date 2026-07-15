"""VRAM-targeted batch sizing contracts for production training."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from forge import training_runtime
from forge.config import ForgeConfig
from forge.profiler import vram


@pytest.fixture
def deterministic_vram_estimate(monkeypatch):
    monkeypatch.setattr(training_runtime.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(training_runtime, "_cuda_total_gib", lambda _device: 10.0)
    monkeypatch.setattr(
        vram,
        "estimate_vram",
        lambda _config, gpu_vram_gb: SimpleNamespace(
            training_fp16_mb=2000.0,
            per_sample_activation_mb=100.0,
            recommended_batch_size=64,
        ),
    )


def test_default_cuda_batch_targets_sixty_to_eighty_percent(deterministic_vram_estimate) -> None:
    config = ForgeConfig.default()
    batch_size, details = training_runtime.choose_batch_size(
        config,
        device="cuda",
        requested=None,
        dataset_size=100,
    )
    assert batch_size == 51
    assert details["target_batch_size"] == 51
    assert details["estimated_target_met"] is True
    assert 0.60 <= details["estimated_utilization"] <= 0.80
    assert details["limiting_factor"] is None


def test_cuda_batch_rejects_dataset_limit_below_vram_floor(deterministic_vram_estimate) -> None:
    config = ForgeConfig.default()
    with pytest.raises(
        training_runtime.TrainingRuntimeError,
        match="dataset is too small.*explicit --batch-size",
    ):
        training_runtime.choose_batch_size(
            config,
            device="cuda",
            requested=None,
            dataset_size=5,
        )


def test_cuda_batch_rejects_unavoidable_out_of_target_estimate(monkeypatch) -> None:
    monkeypatch.setattr(training_runtime.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(training_runtime, "_cuda_total_gib", lambda _device: 10.0)
    monkeypatch.setattr(
        vram,
        "estimate_vram",
        lambda _config, gpu_vram_gb: SimpleNamespace(
            training_fp16_mb=9000.0,
            per_sample_activation_mb=100.0,
            recommended_batch_size=64,
        ),
    )

    with pytest.raises(training_runtime.TrainingRuntimeError, match="required 60-80% VRAM"):
        training_runtime.choose_batch_size(
            ForgeConfig.default(),
            device="cuda",
            requested=None,
            dataset_size=100,
        )
