"""ForgeModelServer — wraps a FORGE student as a vla-eval compatible model server.

Uses WebSocket/msgpack protocol expected by the vla-evaluation-harness Docker
benchmark containers. Loads a FORGE student checkpoint and serves predictions.

Usage:
    server = ForgeModelServer(
        checkpoint_path="./outputs/checkpoints/best.pt",
        variant="nano",
        device="cuda",
    )
    server.start(port=8000)
"""

from __future__ import annotations

import asyncio
import io
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from forge.checkpoint_compat import (
    CheckpointLoadReport,
    apply_checkpoint_structure,
    extract_checkpoint_state_dict,
    load_checkpoint_payload,
    load_model_weights_with_compatibility,
    summarize_checkpoint_report,
)

logger = logging.getLogger(__name__)


def _resolve_runtime_device(requested_device: str) -> str:
    """Use the same strict indexed-device contract as every public CLI command."""
    from forge.cli_commands.shared import resolve_runtime_device

    return resolve_runtime_device(requested_device, command="eval", default="auto", strict=True)


@dataclass
class ServerConfig:
    """Configuration for the model server."""

    checkpoint_path: str = ""
    variant: str = "nano"
    model_dir: str | None = None
    device: str = "cuda"
    allow_mock: bool = False
    port: int = 8000
    host: str = "0.0.0.0"
    chunk_size: int = 1
    action_dim: int = 7
    image_size: int = 384
    action_scale: float = 1.0
    action_offset: float = 0.0


class ForgeModelServer:
    """Wraps any FORGE student as a vla-eval compatible model server.

    Implements the WebSocket/msgpack protocol used by vla-evaluation-harness.
    Lazy-loads the model on first prediction to match vla-eval patterns.
    """

    def __init__(
        self,
        checkpoint_path: str,
        variant: str = "nano",
        model_dir: str | None = None,
        device: str = "cuda",
        allow_mock: bool = False,
        chunk_size: int = 1,
        port: int = 8000,
        host: str = "0.0.0.0",
        image_size: int = 384,
        action_scale: float = 1.0,
        action_offset: float = 0.0,
    ):
        self.config = ServerConfig(
            checkpoint_path=checkpoint_path,
            variant=variant,
            model_dir=model_dir,
            device=_resolve_runtime_device(device),
            allow_mock=allow_mock,
            port=port,
            host=host,
            chunk_size=chunk_size,
            image_size=image_size,
            action_scale=action_scale,
            action_offset=action_offset,
        )
        self._model: Any | None = None
        self._loaded = False
        self._server_thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._server: Any | None = None
        self._ready_event = threading.Event()
        self._stop_event = threading.Event()
        self._thread_error: BaseException | None = None

    def _ensure_model_loaded(self) -> None:
        """Lazy-load model on first prediction."""
        if self._loaded:
            return

        from forge.config import ForgeConfig, apply_checkpoint_student_config, apply_student_variant
        from forge.student import FORGEStudent

        logger.info(f"Loading FORGE-{self.config.variant} from {self.config.checkpoint_path}")

        config = ForgeConfig.default()
        apply_student_variant(config.student, self.config.variant)
        config.student.allow_mock = bool(config.student.allow_mock or self.config.allow_mock)

        # Load checkpoint and apply saved HP config before building student
        ckpt_path = Path(self.config.checkpoint_path)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        checkpoint = load_checkpoint_payload(
            str(ckpt_path),
            map_location=self.config.device,
            verify_provenance_for="eval",
            allow_mock=config.student.allow_mock,
        )
        if checkpoint is None:
            raise ValueError(f"Unsupported or unreadable checkpoint payload in {ckpt_path}")
        if not isinstance(checkpoint, dict):
            raise ValueError(f"Unsupported checkpoint payload in {ckpt_path}")

        apply_checkpoint_student_config(config.student, checkpoint)
        action_horizon = int(config.student.action_horizon)
        action_dim = int(config.student.action_dim)
        if action_horizon < 1:
            raise ValueError(f"Checkpoint action_horizon must be positive, got {action_horizon}")
        if action_dim < 1:
            raise ValueError(f"Checkpoint action_dim must be positive, got {action_dim}")
        self.config.chunk_size = action_horizon
        self.config.action_dim = action_dim

        student = FORGEStudent(config.student, model_dir=self.config.model_dir)

        apply_checkpoint_structure(student, checkpoint)
        state_dict, extracted_key = extract_checkpoint_state_dict(checkpoint)
        if not isinstance(state_dict, dict) or not state_dict:
            raise ValueError(f"Checkpoint has no usable state dict: {ckpt_path}")

        report = CheckpointLoadReport(source=str(ckpt_path), extracted_key=extracted_key)
        report.raw_key_count = len(state_dict)
        try:
            missing, report = load_model_weights_with_compatibility(
                student,
                state_dict,
                context=f"eval_model_server:{ckpt_path}",
                minimum_coverage=0.8,
            )
        except RuntimeError:
            logger.exception("Failed loading checkpoint %s", ckpt_path)
            raise

        for warning in report.warnings:
            logger.warning("%s", warning)
        logger.info(summarize_checkpoint_report("eval_model_server", report))
        logger.info(f"Loaded checkpoint from {ckpt_path}")

        if missing.unexpected_keys:
            logger.warning("Unexpected checkpoint keys dropped: %s", ", ".join(missing.unexpected_keys[:12]))
        if missing.missing_keys:
            logger.warning("Model keys missing in checkpoint: %s", ", ".join(missing.missing_keys[:12]))

        student = student.to(self.config.device)
        student.eval()
        self._model = student
        self._loaded = True
        logger.info("Model loaded and ready for predictions")

    def predict(self, observation: dict, context: dict | None = None) -> dict:
        """Process observation → action using FORGE student.

        Args:
            observation: Dict with keys:
                - images: camera mapping containing a real numpy (H,W,3)
                  uint8 frame. Official harnesses use keys including
                  ``agentview`` (LIBERO) and ``primary`` (Simpler/VLABench).
                - task_description: language instruction string
                - state: robot proprioception (optional)
            context: Session context from vla-eval (optional)

        Returns:
            Dict with "actions" key: np.ndarray shape (7,) or (H, 7)
        """
        self._ensure_model_loaded()

        # Extract image
        images = observation.get("images", {})
        if not isinstance(images, dict):
            raise ValueError("Evaluation observation 'images' must be a camera mapping")
        image = self._select_camera_frame(images)
        if image is None:
            raise ValueError("Evaluation observation requires a real base_camera, agentview, primary, or image frame")

        # Preprocess image: numpy (H,W,3) uint8 → torch tensor (1, 3, 384, 384)
        image_tensor = self._preprocess_image(image)

        # Extract language instruction
        instruction = observation.get("task_description", "")
        if isinstance(instruction, bytes):
            instruction = instruction.decode("utf-8")

        # Run forward pass
        model = self._model
        if model is None:
            raise RuntimeError("FORGE model failed to load")
        with torch.no_grad():
            result = model(
                image_tensor,
                language_text=instruction,
            )

        # Extract actions
        actions = result.get("actions", result.get("action"))
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()
        if not isinstance(actions, np.ndarray) or not np.isfinite(actions).all():
            raise RuntimeError("FORGE evaluation returned missing or non-finite actions")

        # Handle shape: ensure (action_dim,) or (H, action_dim)
        if actions.ndim == 3:
            actions = actions[0]  # Remove batch dim: (1, H, D) → (H, D)
        if actions.ndim == 2 and actions.shape[0] == 1:
            actions = actions[0]  # (1, D) → (D,)

        # Denormalize
        actions = actions * self.config.action_scale + self.config.action_offset

        return {"actions": actions}

    @staticmethod
    def _select_camera_frame(images: dict[str, Any]) -> Any | None:
        """Select the primary real frame from official and generic harness mappings."""
        preferred_keys = (
            "base_camera",
            "agentview",
            "primary",
            "image",
            "rgb_static",
            "viewport",
        )
        for key in preferred_keys:
            frame = images.get(key)
            if frame is not None:
                return frame
        return next((frame for frame in images.values() if frame is not None), None)

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Convert numpy image to model input tensor."""
        from PIL import Image
        from torchvision import transforms  # type: ignore[import-untyped]

        # Handle different input formats
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            pil_image = Image.fromarray(image)
        else:
            pil_image = image

        transform = transforms.Compose(
            [
                transforms.Resize((self.config.image_size, self.config.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        tensor = transform(pil_image).unsqueeze(0)  # (1, 3, H, W)
        return tensor.to(self.config.device)

    def _next_protocol_action(
        self,
        actions: Any,
        action_buffer: deque[np.ndarray],
    ) -> np.ndarray:
        """Validate one exact checkpoint-shaped chunk and return its next action."""
        array = np.asarray(actions)
        if not np.isfinite(array).all():
            raise RuntimeError("FORGE evaluation returned non-finite actions")
        if array.ndim == 1:
            expected_vector = (self.config.action_dim,)
            if array.shape != expected_vector:
                raise RuntimeError(
                    f"FORGE evaluation returned action shape {array.shape}; "
                    f"checkpoint requires {expected_vector} or "
                    f"({self.config.chunk_size}, {self.config.action_dim})"
                )
            return array
        expected_chunk = (self.config.chunk_size, self.config.action_dim)
        if array.ndim != 2 or array.shape != expected_chunk:
            raise RuntimeError(
                f"FORGE evaluation returned action shape {array.shape}; checkpoint requires {expected_chunk}"
            )
        action_buffer.extend(row.copy() for row in array[1:])
        return array[0]

    @staticmethod
    def _decode_ndarray_payload(payload: Any) -> np.ndarray | dict[str, Any] | Any:
        """Decode numpy/image payload dictionaries created by the vla_eval codec."""
        if not isinstance(payload, dict):
            return payload

        if payload.get("__ndarray__"):
            try:
                dtype = np.dtype(payload["dtype"])
                data = payload["data"]
                shape = payload["shape"]
                if dtype.kind not in {"b", "i", "u", "f"}:
                    return None
                return np.frombuffer(data, dtype=dtype).reshape(shape).copy()
            except Exception:
                return None

        if payload.get("__image__"):
            try:
                fmt = payload["format"]
                data = payload["data"]
                shape = payload["shape"]

                if fmt == "raw":
                    dtype = np.dtype(payload.get("dtype", "uint8"))
                    return np.frombuffer(data, dtype=dtype).reshape(shape).copy()

                from PIL import Image

                with Image.open(io.BytesIO(data)) as pil_image:
                    return np.array(pil_image)
            except Exception:
                return None

        return payload

    @classmethod
    def _decode_msgpack_payload(cls, data: Any) -> dict[str, Any]:
        """Decode protocol wrapper objects embedded in msgpack payloads."""
        if not isinstance(data, dict):
            return {}

        def _decode_value(value: Any) -> Any:
            decoded_value = cls._decode_ndarray_payload(value)
            if isinstance(decoded_value, np.ndarray):
                return decoded_value
            if isinstance(decoded_value, dict):
                return cls._decode_msgpack_payload(decoded_value)
            if isinstance(decoded_value, list):
                return [_decode_value(item) for item in decoded_value]
            return decoded_value

        decoded = {}
        for key, value in data.items():
            decoded[key] = _decode_value(value)
        return decoded

    async def _handle_client(self, websocket: Any) -> None:
        """Handle a single WebSocket client using msgpack protocol."""
        try:
            import msgpack  # type: ignore[import-untyped]
        except ImportError:
            logger.error("msgpack is missing from the FORGE installation. Reinstall anima-forge.")
            return

        try:
            from vla_eval.protocol.messages import (  # type: ignore[import-not-found]
                Message,
                MessageType,
                pack_message,
                unpack_message,
            )

            protocol_message_cls = Message
            protocol_message_type_cls = MessageType
            protocol_codec = True
            protocol_message_types = {str(m.value).lower(): m for m in protocol_message_type_cls}
        except Exception:
            protocol_message_cls = None
            protocol_message_type_cls = None
            pack_message = None
            unpack_message = None
            protocol_message_types = {}
            protocol_codec = False

        action_buffer: deque[np.ndarray] = deque()
        logger.info(f"Client connected: {websocket.remote_address}")
        try:
            async for message in websocket:
                try:
                    if isinstance(message, memoryview):
                        message = message.tobytes()
                    elif isinstance(message, str):
                        message = message.encode("utf-8")
                    if not isinstance(message, (bytes, bytearray)):
                        logger.warning("Ignoring non-binary message payload: %s", type(message).__name__)
                        continue

                    def _normalize_msg_type(raw_type: Any) -> str:
                        if isinstance(raw_type, str):
                            return raw_type.lower()
                        if protocol_message_type_cls is not None:
                            try:
                                return str(raw_type.value).lower()
                            except Exception:
                                return str(raw_type).lower()
                        return str(raw_type).lower()

                    if protocol_codec:
                        msg = unpack_message(message)
                        msg_type = _normalize_msg_type(msg.type)
                        payload = self._decode_msgpack_payload(msg.payload)
                        seq = int(msg.seq)
                    else:
                        # Decode msgpack message (legacy format)
                        data = msgpack.unpackb(
                            message,
                            raw=False,
                            object_hook=self._decode_ndarray_payload,
                        )
                        if not isinstance(data, dict):
                            logger.warning("Ignoring malformed message payload: %s", type(data).__name__)
                            continue
                        data = self._decode_msgpack_payload(data)
                        msg_type = _normalize_msg_type(data.get("type", "predict"))
                        payload = data
                        seq = int(data.get("seq", 0))

                    if msg_type == "hello":
                        response_payload = {
                            "model": f"FORGE-{self.config.variant}",
                            "checkpoint": self.config.checkpoint_path,
                            "chunk_size": self.config.chunk_size,
                            "action_dim": self.config.action_dim,
                            "protocol_version": 1,
                        }
                        if protocol_codec:
                            response_type = protocol_message_types.get("hello")
                            if response_type is not None:
                                response = pack_message(
                                    protocol_message_cls(
                                        type=response_type,
                                        payload=response_payload,
                                        seq=seq,
                                    )
                                )
                            else:
                                logger.warning("Protocol hello response type not available; ignoring.")
                                continue
                        else:
                            response = msgpack.packb(
                                {
                                    "type": "hello",
                                    "payload": response_payload,
                                    "seq": seq,
                                    "timestamp": time.time(),
                                },
                                use_bin_type=True,
                            )
                        await websocket.send(response)
                        continue

                    if msg_type in {"observation", "predict"}:
                        observation = payload.get("observation") if isinstance(payload, dict) else None
                        if observation is None:
                            observation = payload.get("payload", payload) if isinstance(payload, dict) else payload
                        if observation is None:
                            observation = payload
                        if not isinstance(observation, dict):
                            observation = {"images": {"base_camera": observation}}
                        context = payload.get("context", {}) if isinstance(payload, dict) else {}
                        if not isinstance(context, dict):
                            context = {}
                        if action_buffer:
                            protocol_action = action_buffer.popleft()
                        else:
                            result = self.predict(observation, context)
                            protocol_action = self._next_protocol_action(result["actions"], action_buffer)

                        # Official harnesses consume exactly one action vector
                        # for each observation even when the student predicts a chunk.
                        actions = protocol_action.tolist()

                        if protocol_codec:
                            response_type = protocol_message_types.get("action")
                            if response_type is None:
                                logger.warning("Protocol action response type not available; skipping.")
                                continue
                            response = pack_message(
                                protocol_message_cls(
                                    type=response_type,
                                    payload={"actions": actions},
                                    seq=seq,
                                )
                            )
                        else:
                            response = msgpack.packb(
                                {
                                    "type": "action",
                                    "payload": {"actions": actions},
                                    "seq": seq,
                                    "timestamp": time.time(),
                                },
                                use_bin_type=True,
                            )
                        await websocket.send(response)

                    elif msg_type in {"reset", "episode_start", "episode_end"}:
                        action_buffer.clear()
                        logger.info("Received lifecycle signal: %s", msg_type)
                        continue

                    elif msg_type == "info":
                        payload = {
                            "model": f"FORGE-{self.config.variant}",
                            "checkpoint": self.config.checkpoint_path,
                            "chunk_size": self.config.chunk_size,
                            "action_dim": self.config.action_dim,
                        }
                        if protocol_codec:
                            response_type = protocol_message_types.get(
                                "info", protocol_message_types.get("info_payload")
                            )
                            if response_type is None:
                                response_type = protocol_message_types.get("error")
                                if response_type is None:
                                    logger.warning("Protocol info response not available; skipping.")
                                    continue
                                payload = {"error": "Info channel is unsupported for this protocol"}
                            response = pack_message(
                                protocol_message_cls(
                                    type=response_type,
                                    payload=payload,
                                    seq=seq,
                                )
                            )
                        else:
                            response = msgpack.packb(
                                {
                                    "type": "info",
                                    "payload": payload,
                                    "seq": seq,
                                    "timestamp": time.time(),
                                },
                                use_bin_type=True,
                            )
                        await websocket.send(response)

                    else:
                        logger.warning("Ignoring unknown message type: %s", msg_type)

                except Exception as e:
                    logger.error("Error processing message", exc_info=True)
                    try:
                        if protocol_codec:
                            if isinstance(message, (bytes, bytearray)):
                                try:
                                    error_msg = unpack_message(message)
                                    seq = int(error_msg.seq)
                                except Exception:
                                    pass
                        elif isinstance(message, (bytes, bytearray)):
                            try:
                                data = msgpack.unpackb(message, raw=False)
                                if isinstance(data, dict):
                                    seq = int(data.get("seq", 0))
                            except Exception:
                                pass
                        if protocol_codec:
                            response_type = protocol_message_types.get("error")
                            if response_type is not None:
                                error_response = pack_message(
                                    protocol_message_cls(
                                        type=response_type,
                                        payload={"error": str(e)},
                                        seq=seq,
                                    )
                                )
                            else:
                                error_response = None
                        else:
                            error_response = msgpack.packb(
                                {
                                    "type": "error",
                                    "payload": {"error": str(e)},
                                    "seq": seq,
                                    "timestamp": time.time(),
                                },
                                use_bin_type=True,
                            )
                        if error_response is not None:
                            await websocket.send(error_response)
                    except Exception:
                        logger.warning("Unable to send error response", exc_info=True)

        except Exception as e:
            logger.info(f"Client disconnected: {e}")

    async def _serve(self) -> None:
        """Start the WebSocket server."""
        try:
            import websockets
        except ImportError:
            logger.error("websockets is missing from the FORGE installation. Reinstall anima-forge.")
            return

        self._ensure_model_loaded()
        self._ready_event.clear()
        self._thread_error = None

        logger.info(f"Starting FORGE model server on ws://{self.config.host}:{self.config.port}")
        try:
            self._server = await websockets.serve(self._handle_client, self.config.host, self.config.port)
            self._ready_event.set()
            logger.info("FORGE model server started and ready")

            # Poll for stop to avoid stop() hard-shutdown corruption.
            while not self._stop_event.is_set():
                await asyncio.sleep(0.05)
        finally:
            if self._server is not None:
                self._server.close()
                await self._server.wait_closed()
                self._server = None
            self._ready_event.clear()

    def start(self, blocking: bool = True, startup_timeout: float = 10.0, wait_for_ready: bool = True) -> bool:
        """Start the model server.

        Args:
            blocking: If True, blocks the current thread. If False, runs in background.
            startup_timeout: Max seconds to wait when running non-blocking.
            wait_for_ready: Wait for server bind confirmation before returning.
        """
        self._stop_event.clear()
        self._thread_error = None
        self._ready_event.clear()
        # Model construction/checkpoint loading is synchronous and cannot be
        # cancelled safely. Complete it before starting the timeout-governed
        # server thread so a bind timeout can never orphan a model-loading job.
        self._ensure_model_loaded()

        if blocking:
            asyncio.run(self._serve())
            return True
        else:
            loop = asyncio.new_event_loop()
            self._loop = loop

            def _run():
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(self._serve())
                except BaseException as e:  # pragma: no cover - defensive: thread boundary
                    self._thread_error = e
                    self._ready_event.set()
                    logger.exception("Model server thread exited with error", exc_info=e)
                finally:
                    if not loop.is_closed():
                        loop.close()

            self._server_thread = threading.Thread(target=_run, daemon=True)
            self._server_thread.start()
            if wait_for_ready:
                signaled = self._ready_event.wait(startup_timeout)
                if self._thread_error:
                    self.stop()
                    raise RuntimeError(f"Model server failed to start: {self._thread_error}") from self._thread_error
                if not signaled:
                    self.stop()
                    raise TimeoutError(f"Model server failed to bind within {startup_timeout}s")
            logger.info("Model server started in background")
            return True

    def stop(self) -> None:
        """Stop the background server."""
        self._stop_event.set()
        if self._loop:
            self._loop.call_soon_threadsafe(lambda: None)
        if self._server_thread:
            self._server_thread.join(timeout=5)
            self._server_thread = None
        self._loop = None

    def wait_until_ready(self, timeout: float = 10.0) -> bool:
        """Wait for background server readiness."""
        return self._ready_event.wait(timeout)

    @property
    def startup_error(self) -> BaseException | None:
        """Last startup exception from background thread."""
        return self._thread_error
