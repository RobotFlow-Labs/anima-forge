"""PRD-15: Async Runtime & Inference Server tests."""

import time

import numpy as np
import pytest
import torch

from forge.runtime.async_engine import (
    AsyncInferenceEngine,
    ChunkBuffer,
    RuntimeConfig,
    RuntimeStatus,
)


def test_chunk_buffer_push_pop():
    """ChunkBuffer correctly stores and serves actions."""
    buf = ChunkBuffer(max_size=4, horizon=3, action_dim=7)

    # Push a chunk of 3 actions
    chunk = np.array([[1, 0, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0, 0], [3, 0, 0, 0, 0, 0, 0]], dtype=np.float32)
    buf.push(chunk)

    # Pop all 3 actions
    a1 = buf.pop_action()
    assert a1 is not None
    assert a1[0] == 1.0

    a2 = buf.pop_action()
    assert a2[0] == 2.0

    a3 = buf.pop_action()
    assert a3[0] == 3.0

    # Buffer should be empty now
    assert buf.pop_action() is None


def test_chunk_buffer_empty():
    """Empty buffer returns None."""
    buf = ChunkBuffer(max_size=4, horizon=8, action_dim=7)
    assert buf.is_empty
    assert buf.pop_action() is None
    assert buf.size == 0


def test_chunk_buffer_overflow():
    """Buffer respects max_size (ring buffer behavior)."""
    buf = ChunkBuffer(max_size=2, horizon=2, action_dim=3)

    # Push 3 chunks (max_size=2, so oldest gets dropped from queue)
    for i in range(3):
        chunk = np.full((2, 3), float(i), dtype=np.float32)
        buf.push(chunk)

    # The active chunk counts toward max_size, so only one queued chunk remains.
    assert buf.size == 2
    # Pop actions from chunk 0
    a = buf.pop_action()
    assert a is not None


def test_chunk_buffer_rejects_invalid_actions():
    buf = ChunkBuffer(max_size=2, horizon=2, action_dim=3)

    with pytest.raises(ValueError, match="expected"):
        buf.push(np.zeros((1, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="finite"):
        buf.push(np.full((2, 3), np.nan, dtype=np.float32))


def test_runtime_config_defaults():
    """RuntimeConfig has sensible defaults."""
    config = RuntimeConfig()
    assert config.max_buffer_size == 4
    assert config.action_horizon == 8
    assert config.chunk_overlap == 2
    assert config.target_hz == 50
    assert config.vision_timeout_ms == 100
    assert config.action_dim == 7


def test_runtime_status_dataclass():
    """RuntimeStatus initializes to zero state."""
    status = RuntimeStatus()
    assert status.is_running is False
    assert status.frames_processed == 0
    assert status.actions_served == 0
    assert status.buffer_size == 0
    assert status.avg_vision_ms == 0.0


def test_async_engine_start_stop():
    """AsyncInferenceEngine starts and stops cleanly."""
    from forge.config import StudentConfig
    from forge.student import FORGEStudent

    config = StudentConfig()
    model = FORGEStudent(config)
    model.eval()

    engine = AsyncInferenceEngine(model, RuntimeConfig())
    engine.start()
    assert engine.is_running

    # Brief sleep to let thread start
    time.sleep(0.05)

    engine.stop()
    assert not engine.is_running


def test_async_engine_submit_frame():
    """Engine processes submitted frames."""
    from forge.config import StudentConfig
    from forge.student import FORGEStudent

    config = StudentConfig()
    model = FORGEStudent(config)
    model.eval()

    engine = AsyncInferenceEngine(model, RuntimeConfig(action_horizon=1))
    engine.start()

    # Submit a frame
    image = np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8)
    engine.submit_frame(image, "pick up the cup")

    # Wait for processing
    time.sleep(0.5)

    status = engine.get_status()
    assert status.frames_processed >= 1

    engine.stop()


def test_async_engine_get_action_empty():
    """get_action returns None when buffer is empty."""
    from forge.config import StudentConfig
    from forge.student import FORGEStudent

    config = StudentConfig()
    model = FORGEStudent(config)

    engine = AsyncInferenceEngine(model, RuntimeConfig())
    # Don't start engine — buffer is empty
    assert engine.get_action() is None


def test_async_engine_passes_instruction_and_validates_actions():
    class _Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.instructions: list[str] = []

        def forward(self, images, *, language_text):
            self.instructions.append(language_text)
            return {"actions": torch.zeros((1, 2, 3), device=images.device)}

    model = _Model()
    engine = AsyncInferenceEngine(model, RuntimeConfig(action_horizon=2, action_dim=3))
    image = np.zeros((8, 8, 3), dtype=np.uint8)

    chunk = engine._run_inference(image, "move left")

    assert model.instructions == ["move left"]
    assert chunk is not None and chunk.shape == (2, 3)


def test_async_engine_rejects_empty_instruction_and_wrong_action_shape():
    class _Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def forward(self, images, *, language_text):
            return {"actions": torch.zeros((1, 1, 3), device=images.device)}

    engine = AsyncInferenceEngine(_Model(), RuntimeConfig(action_horizon=2, action_dim=3))
    image = np.zeros((8, 8, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="non-empty"):
        engine.submit_frame(image, "")
    assert engine._run_inference(image, "move") is None
    assert engine.get_status().last_error == "Model inference returned action shape (1, 3); expected (2, 3)"
