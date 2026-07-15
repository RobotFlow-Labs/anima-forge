"""Async inference engine — decouples vision from action.

Design:
- VisionThread: runs perception pipeline (vision encoder + bridge + backbone)
  on new camera frames as they arrive. Updates shared feature state.
- ActionThread: serves actions from chunk buffer. When buffer depletes,
  triggers new perception cycle.
- ChunkBuffer: ring buffer of predicted action chunks with overlap blending.

Threading model: VisionThread is the bottleneck (~30ms on L4).
ActionThread is fast (<1ms per action lookup).
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class RuntimeConfig:
    """Async runtime configuration."""

    max_buffer_size: int = 4  # Max action chunks in buffer
    vision_timeout_ms: float = 100  # Max wait for vision processing
    action_horizon: int = 8  # Actions per chunk
    chunk_overlap: int = 2  # Overlap for blending
    target_hz: float = 50  # Target action frequency (Hz)
    action_dim: int = 7  # Action dimension


@dataclass
class RuntimeStatus:
    """Current runtime state."""

    is_running: bool = False
    frames_processed: int = 0
    actions_served: int = 0
    buffer_size: int = 0
    avg_vision_ms: float = 0.0
    avg_action_us: float = 0.0  # Microseconds
    uptime_seconds: float = 0.0
    last_error: str | None = None


class ChunkBuffer:
    """Thread-safe ring buffer for action chunks.

    Stores predicted action chunks and serves individual actions sequentially.
    When the current chunk is exhausted, automatically advances to the next.

    Args:
        max_size: Maximum chunks to buffer
        horizon: Actions per chunk
        action_dim: Dimension of each action
    """

    def __init__(self, max_size: int = 4, horizon: int = 8, action_dim: int = 7):
        if max_size < 1 or horizon < 1 or action_dim < 1:
            raise ValueError("ChunkBuffer max_size, horizon, and action_dim must all be positive")
        self.max_size = max_size
        self.horizon = horizon
        self.action_dim = action_dim
        # The active chunk counts toward max_size; the queue owns only the
        # remaining capacity.
        self._buffer: deque[np.ndarray] = deque(maxlen=max_size - 1)
        self._current_chunk: np.ndarray | None = None
        self._current_step: int = 0
        self._lock = threading.Lock()

    def push(self, chunk: np.ndarray) -> None:
        """Add a new action chunk to the buffer.

        Args:
            chunk: (H, D_action) action chunk array
        """
        array = np.asarray(chunk)
        expected = (self.horizon, self.action_dim)
        if array.shape != expected:
            raise ValueError(f"Action chunk has shape {array.shape}; expected {expected}")
        if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
            raise ValueError("Action chunk must contain only finite numeric values")
        canonical = np.ascontiguousarray(array, dtype=np.float32)
        with self._lock:
            self._buffer.append(canonical)
            if self._current_chunk is None:
                self._current_chunk = self._buffer.popleft()
                self._current_step = 0

    def pop_action(self) -> np.ndarray | None:
        """Get the next action from the current chunk.

        Returns:
            (D_action,) action array, or None if buffer is empty
        """
        with self._lock:
            if self._current_chunk is None:
                return None

            action = self._current_chunk[self._current_step].copy()
            self._current_step += 1

            if self._current_step >= len(self._current_chunk):
                # Advance to next chunk
                if len(self._buffer) > 0:
                    self._current_chunk = self._buffer.popleft()
                    self._current_step = 0
                else:
                    self._current_chunk = None
                    self._current_step = 0

            return action

    @property
    def size(self) -> int:
        """Number of chunks available (including current)."""
        with self._lock:
            return len(self._buffer) + (1 if self._current_chunk is not None else 0)

    @property
    def is_empty(self) -> bool:
        return self.size == 0

    def clear(self) -> None:
        """Clear all buffered chunks."""
        with self._lock:
            self._buffer.clear()
            self._current_chunk = None
            self._current_step = 0


class AsyncInferenceEngine:
    """Main async inference engine.

    Decouples vision processing from action serving using a background
    thread for model inference and a chunk buffer for zero-latency
    action delivery.

    Usage:
        engine = AsyncInferenceEngine(model, config)
        engine.start()

        # In robot control loop:
        engine.submit_frame(image, instruction="pick up the cup")
        action = engine.get_action()

        engine.stop()
    """

    def __init__(self, model: torch.nn.Module, config: RuntimeConfig | None = None):
        self.model = model
        self.config = config or RuntimeConfig()
        self.buffer = ChunkBuffer(
            max_size=self.config.max_buffer_size,
            horizon=self.config.action_horizon,
            action_dim=self.config.action_dim,
        )
        self._status = RuntimeStatus()
        self._frame_queue: deque[tuple[np.ndarray, str]] = deque(maxlen=2)
        self._frame_lock = threading.Lock()
        self._running = False
        self._vision_thread: threading.Thread | None = None
        self._start_time: float = 0.0

    def start(self) -> None:
        """Start the async engine background thread."""
        if self._running:
            return
        self._running = True
        self._status.is_running = True
        self._start_time = time.time()
        self._vision_thread = threading.Thread(target=self._vision_loop, daemon=True, name="forge-vision")
        self._vision_thread.start()
        logger.info("Async inference engine started")

    def stop(self) -> None:
        """Stop the async engine and wait for thread to finish."""
        self._running = False
        if self._vision_thread is not None:
            self._vision_thread.join(timeout=2.0)
            self._vision_thread = None
        self._status.is_running = False
        logger.info("Async inference engine stopped")

    def submit_frame(self, image: np.ndarray, instruction: str) -> None:
        """Submit a new camera frame for processing.

        Args:
            image: (H, W, C) uint8 or float32 camera image
            instruction: Language instruction for the task
        """
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("Async inference requires a non-empty language instruction")
        frame = np.asarray(image)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Async inference image must have shape (H, W, 3), got {frame.shape}")
        if not np.issubdtype(frame.dtype, np.number) or not np.isfinite(frame).all():
            raise ValueError("Async inference image must contain finite numeric pixels")
        with self._frame_lock:
            self._frame_queue.append((np.ascontiguousarray(frame), instruction.strip()))

    def get_action(self) -> np.ndarray | None:
        """Get the next action from the chunk buffer.

        Non-blocking, <1ms latency.

        Returns:
            (D_action,) action array, or None if buffer is empty
        """
        t0 = time.perf_counter()
        action = self.buffer.pop_action()
        elapsed_us = (time.perf_counter() - t0) * 1e6

        if action is not None:
            self._status.actions_served += 1
            n = self._status.actions_served
            self._status.avg_action_us = (self._status.avg_action_us * (n - 1) + elapsed_us) / n

        return action

    def get_status(self) -> RuntimeStatus:
        """Get current runtime status."""
        self._status.buffer_size = self.buffer.size
        if self._running:
            self._status.uptime_seconds = time.time() - self._start_time
        return self._status

    @property
    def is_running(self) -> bool:
        return self._running

    def _vision_loop(self) -> None:
        """Background thread: process frames and fill chunk buffer."""
        while self._running:
            with self._frame_lock:
                has_frame = len(self._frame_queue) > 0
                if has_frame:
                    image, instruction = self._frame_queue.popleft()
            if has_frame:
                t0 = time.time()

                chunk = self._run_inference(image, instruction)
                if chunk is not None:
                    self.buffer.push(chunk)
                    self._status.last_error = None

                elapsed_ms = (time.time() - t0) * 1000
                self._status.frames_processed += 1
                n = self._status.frames_processed
                self._status.avg_vision_ms = (self._status.avg_vision_ms * (n - 1) + elapsed_ms) / n
            else:
                time.sleep(0.001)  # 1ms sleep when idle

    def _run_inference(self, image: np.ndarray, instruction: str) -> np.ndarray | None:
        """Run model inference on a frame.

        Args:
            image: (H, W, C) camera image

        Returns:
            (H, D_action) action chunk, or None on error
        """
        try:
            # Convert to tensor: (H, W, C) → (1, C, H, W)
            if image.dtype == np.uint8:
                tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            else:
                tensor = torch.from_numpy(image).permute(2, 0, 1).float()

            tensor = tensor.unsqueeze(0)

            device = next(self.model.parameters()).device
            tensor = tensor.to(device)

            with torch.no_grad():
                output = self.model(tensor, language_text=instruction)

            if not isinstance(output, dict) or "actions" not in output:
                raise RuntimeError("Model inference returned no actions")
            actions_tensor = output["actions"]
            if not isinstance(actions_tensor, torch.Tensor):
                raise RuntimeError("Model inference actions must be a tensor")
            actions = actions_tensor.detach().float().cpu().numpy()
            if actions.ndim == 3:
                if actions.shape[0] != 1:
                    raise RuntimeError(f"Model inference returned batch {actions.shape[0]}; expected 1")
                actions = actions[0]
            elif actions.ndim == 2 and actions.shape == (1, self.config.action_dim):
                actions = actions.reshape(1, self.config.action_dim)
            expected = (self.config.action_horizon, self.config.action_dim)
            if actions.shape != expected:
                raise RuntimeError(f"Model inference returned action shape {actions.shape}; expected {expected}")
            if not np.isfinite(actions).all():
                raise RuntimeError("Model inference returned non-finite actions")
            return np.ascontiguousarray(actions, dtype=np.float32)

        except Exception as e:
            logger.warning(f"Inference failed: {e}")
            self._status.last_error = str(e)
            return None
