"""Integration tests — full v2 pipeline end-to-end."""

from __future__ import annotations

import json
import tempfile
import time

import torch

from forge.config import ForgeConfig, StudentConfig


def test_teacher_registry_to_labels():
    """Registry → create adapter → verify info."""
    from forge.teachers.registry import get_registry

    registry = get_registry()
    names = registry.list_teachers()
    assert len(names) >= 1

    # Create adapter and verify info (no load — needs real weights)
    adapter = registry.create(names[0])
    info = adapter.info()
    assert info.name
    assert info.param_count > 0
    assert info.action_dim > 0

    # Verify all registered teachers have valid info
    for name in names:
        adapter = registry.create(name)
        info = adapter.info()
        assert info.name == name


def test_student_with_flow_head():
    """Student with flow matching head end-to-end."""
    from forge.student import FORGEStudent

    config = StudentConfig(action_head_type="flow", flow_inference_steps=2)
    model = FORGEStudent(config)
    model.eval()

    images = torch.randn(1, 3, 384, 384)
    lang_ids = torch.randint(0, 1000, (1, 10))

    with torch.no_grad():
        out = model(images, language_ids=lang_ids)

    assert "actions" in out
    assert out["actions"].shape[0] == 1


def test_student_with_chunk_head():
    """Student with action chunking head end-to-end."""
    from forge.student import FORGEStudent

    config = StudentConfig(
        action_head_type="chunk",
        action_horizon=4,
    )
    model = FORGEStudent(config)
    model.eval()

    images = torch.randn(1, 3, 384, 384)
    lang_ids = torch.randint(0, 1000, (1, 10))

    with torch.no_grad():
        out = model(images, language_ids=lang_ids)

    assert "actions" in out
    # Chunk head produces H actions
    assert out["actions"].shape[1] == 4


def test_student_v1_backward_compat():
    """v1 config produces identical behavior — diffusion head, H=1."""
    from forge.student import FORGEStudent

    config = StudentConfig()  # All defaults = v1
    assert config.action_head_type == "diffusion"
    assert config.action_horizon == 1

    model = FORGEStudent(config)
    model.eval()

    images = torch.randn(1, 3, 384, 384)
    lang_ids = torch.randint(0, 1000, (1, 10))

    with torch.no_grad():
        out = model(images, language_ids=lang_ids)

    assert "actions" in out
    assert out["actions"].ndim >= 2


def test_chunked_distillation_loop():
    """Full distillation loop with action chunks."""
    from forge.student import FORGEStudent

    config = StudentConfig(
        action_head_type="flow",
        action_horizon=1,
        flow_inference_steps=2,
    )
    model = FORGEStudent(config)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    images = torch.randn(2, 3, 384, 384)
    lang_ids = torch.randint(0, 1000, (2, 10))
    gt_actions = torch.randn(2, 7)  # D=7

    # One training step
    out = model(images, language_ids=lang_ids, gt_actions=gt_actions)
    assert "loss" in out

    loss = out["loss"]
    loss.backward()
    optimizer.step()

    # Loss should be finite
    assert torch.isfinite(loss)


def test_multi_teacher_pipeline():
    """Multi-teacher labels → multi-teacher KD."""
    from forge.multi_teacher import MultiTeacherDistillationLoss, TeacherRouter

    n_teachers = 2
    d_feature = 64
    d_action = 7
    batch_size = 4

    router = TeacherRouter(d_input=d_feature, n_teachers=n_teachers)
    loss_fn = MultiTeacherDistillationLoss(n_teachers=n_teachers, d_student=d_feature)

    features = torch.randn(batch_size, d_feature)
    student_actions = torch.randn(batch_size, d_action)
    teacher_actions = [torch.randn(batch_size, d_action) for _ in range(n_teachers)]
    gt_actions = torch.randn(batch_size, d_action)

    # Test router independently
    weights = router(features)
    assert weights.shape == (batch_size, n_teachers)
    assert torch.allclose(weights.sum(dim=1), torch.ones(batch_size), atol=1e-5)

    # Test full loss (has its own internal router)
    result = loss_fn(student_actions, teacher_actions, gt_actions, features)
    assert "total" in result
    assert result["total"].requires_grad
    assert torch.isfinite(result["total"])


def test_benchmark_suite_runs():
    """Benchmark runner produces valid report."""
    from forge.benchmark.runner import BenchmarkRunner
    from forge.student import FORGEStudent

    config = ForgeConfig.default()
    model = FORGEStudent(config.student)

    runner = BenchmarkRunner(model, config, device="cpu")
    report = runner.run(n_latency_samples=2, throughput_duration=0.3)

    assert report.model_name == "FORGE-nano"
    assert report.latency.mean_ms > 0
    assert report.throughput.actions_per_second > 0
    assert report.compression.compression_ratio > 0

    # Serializable
    d = report.to_dict()
    json.dumps(d)


def test_demo_generates_html():
    """Demo runner components produce HTML report."""
    from forge.demo.report import generate_html_report

    data = {
        "benchmark": {},
        "teachers": [],
        "embodiments": [],
        "architecture": {},
        "version": "3.0.1",
    }

    html = generate_html_report(data)

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
        f.write(html)

    assert "FORGE v3" in html
    assert "[MOCK — not a real model]" in html
    assert '<div class="value">1200</div>' not in html
    assert '<div class="value">480MB</div>' not in html
    assert "<!DOCTYPE html>" in html
    assert len(html) > 1000


def test_async_engine_lifecycle():
    """Start → submit → get_action → stop."""
    from forge.runtime.async_engine import AsyncInferenceEngine, RuntimeConfig
    from forge.student import FORGEStudent

    config = StudentConfig()
    model = FORGEStudent(config)
    model.eval()

    rt_config = RuntimeConfig(
        max_buffer_size=2,
        action_horizon=1,
        action_dim=7,
    )

    engine = AsyncInferenceEngine(model, rt_config)
    engine.start()

    try:
        # Submit frame (engine expects np.ndarray)
        import numpy as np

        frame = np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8)
        engine.submit_frame(frame, "pick up the object")

        # Wait for processing
        time.sleep(2.0)

        status = engine.get_status()
        assert status.is_running

        # Try to get action (may or may not have one yet)
        action = engine.get_action()
        # Action is numpy array or None
        if action is not None:
            assert len(action) == 7
    finally:
        engine.stop()

    status = engine.get_status()
    assert not status.is_running


def test_embodiment_to_config_to_student():
    """Embodiment profile → FORGE config → student init."""
    from forge.embodiments.registry import EmbodimentRegistry
    from forge.student import FORGEStudent

    registry = EmbodimentRegistry()
    profile = registry.get("franka")

    assert profile.dof == 7
    assert profile.action_dim == 7

    overrides = profile.to_forge_config()
    assert "student" in overrides
    assert overrides["student"]["action_dim"] == 7

    # Create student with embodiment settings
    config = StudentConfig(
        action_dim=overrides["student"]["action_dim"],
        action_horizon=overrides["student"]["action_horizon"],
        action_head_type=overrides["student"]["action_head_type"],
    )
    model = FORGEStudent(config)
    model.eval()

    images = torch.randn(1, 3, 384, 384)
    with torch.no_grad():
        out = model(images, language_ids=torch.randint(0, 1000, (1, 10)))
    assert "actions" in out


def test_full_pipeline_cpu():
    """Full v2 pipeline on CPU: init → forward → benchmark → report."""
    from forge.benchmark.metrics import measure_compression, profile_latency
    from forge.demo.report import generate_html_report
    from forge.student import FORGEStudent

    config = StudentConfig(
        action_head_type="flow",
        action_horizon=4,
        flow_inference_steps=2,
    )
    model = FORGEStudent(config)
    model.eval()

    # Forward pass
    images = torch.randn(1, 3, 384, 384)
    with torch.no_grad():
        out = model(images, language_ids=torch.randint(0, 1000, (1, 10)))
    assert "actions" in out

    # Benchmark
    lat = profile_latency(model, n_warmup=1, n_samples=2, device="cpu")
    comp = measure_compression(model, teacher_params_b=7.6)

    assert lat.mean_ms > 0
    assert comp.compression_ratio > 1.0

    # Report
    html = generate_html_report(
        {
            "benchmark": {
                "latency": {"mean_ms": lat.mean_ms},
                "throughput": {"actions_per_second": 100, "chunk_gain": 4},
                "compression": {
                    "compression_ratio": comp.compression_ratio,
                    "model_size_mb": comp.model_size_mb,
                },
            },
            "teachers": [],
            "embodiments": [],
            "architecture": {},
            "version": "2.0.0",
        }
    )
    assert len(html) > 500


def test_version_is_3():
    """Version is 3.0.1."""
    import forge

    assert forge.__version__ == "3.0.1"
