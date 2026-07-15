"""Raw-instruction and fail-closed observation contracts for student evaluation."""

from __future__ import annotations

import asyncio
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from forge.eval.model_server import ForgeModelServer
from forge.student import FORGEStudent, _ensure_tokenizer_padding


class _Tokenizer:
    def __init__(self) -> None:
        self.instructions: list[str] = []

    def __call__(self, instructions: list[str], **kwargs: object) -> dict[str, torch.Tensor]:
        self.instructions = instructions
        assert kwargs["return_tensors"] == "pt"
        assert kwargs["padding"] is True
        values = torch.arange(len(instructions) * 2, dtype=torch.long).reshape(len(instructions), 2)
        return {"input_ids": values}


def test_decoder_only_tokenizer_uses_eos_for_padding() -> None:
    tokenizer = SimpleNamespace(pad_token=None, pad_token_id=None, eos_token="</s>", eos_token_id=2)
    model = SimpleNamespace(config=SimpleNamespace(pad_token_id=None))

    _ensure_tokenizer_padding(tokenizer, model)

    assert tokenizer.pad_token == "</s>"
    assert tokenizer.pad_token_id == 2
    assert model.config.pad_token_id == 2


def test_tokenizer_padding_refuses_missing_eos() -> None:
    tokenizer = SimpleNamespace(pad_token=None, pad_token_id=None, eos_token=None, eos_token_id=None)

    with pytest.raises(RuntimeError, match="neither a padding token"):
        _ensure_tokenizer_padding(tokenizer, SimpleNamespace(config=SimpleNamespace()))


def _student(tokenizer: object | None, provenance: str) -> FORGEStudent:
    student = FORGEStudent.__new__(FORGEStudent)
    nn.Module.__init__(student)
    student.tokenizer = tokenizer
    student.language_provenance = provenance
    return student


def test_student_tokenizes_raw_instruction_for_every_image() -> None:
    tokenizer = _Tokenizer()
    student = _student(tokenizer, "real")

    ids = student._tokenize_language_text("move the block", batch_size=2, device=torch.device("cpu"))

    assert tokenizer.instructions == ["move the block", "move the block"]
    assert ids.tolist() == [[0, 1], [2, 3]]


def test_student_refuses_missing_real_tokenizer() -> None:
    student = _student(None, "real")

    with pytest.raises(RuntimeError, match="will not ignore task instructions"):
        student._tokenize_language_text("move", batch_size=1, device=torch.device("cpu"))


@pytest.mark.parametrize(
    ("instructions", "batch_size"),
    [(["one", "two"], 1), ("", 1), (["one", ""], 2)],
)
def test_student_rejects_invalid_instruction_batches(instructions: str | list[str], batch_size: int) -> None:
    student = _student(_Tokenizer(), "real")

    with pytest.raises(ValueError):
        student._tokenize_language_text(instructions, batch_size=batch_size, device=torch.device("cpu"))


def test_eval_server_refuses_missing_camera_frame() -> None:
    server = ForgeModelServer("missing.pt", device="cpu")
    server._loaded = True
    server._model = SimpleNamespace()

    with pytest.raises(ValueError, match="requires a real base_camera"):
        server.predict({"images": {}, "task_description": "move"})


@pytest.mark.parametrize("camera_key", ["base_camera", "agentview", "primary", "image"])
def test_eval_server_accepts_official_harness_camera_keys(
    camera_key: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Model:
        def __call__(self, *_args: object, **kwargs: object) -> dict[str, np.ndarray]:
            assert kwargs["language_text"] == "move"
            return {"actions": np.zeros((1, 7), dtype=np.float32)}

    frame = np.full((8, 8, 3), 7, dtype=np.uint8)
    server = ForgeModelServer("missing.pt", device="cpu", image_size=8)
    server._loaded = True
    server._model = _Model()
    monkeypatch.setattr(
        server,
        "_preprocess_image",
        lambda image: torch.from_numpy(image.copy()).permute(2, 0, 1).unsqueeze(0),
    )

    result = server.predict({"images": {camera_key: frame}, "task_description": "move"})

    assert result["actions"].shape == (7,)
    assert np.isfinite(result["actions"]).all()


def test_eval_server_rejects_nonfinite_actions(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Model:
        def __call__(self, *_args: object, **_kwargs: object) -> dict[str, np.ndarray]:
            return {"actions": np.array([[np.nan] * 7], dtype=np.float32)}

    server = ForgeModelServer("missing.pt", device="cpu", image_size=8)
    server._loaded = True
    server._model = _Model()
    monkeypatch.setattr(server, "_preprocess_image", lambda _image: torch.zeros(1, 3, 8, 8))

    with pytest.raises(RuntimeError, match="non-finite actions"):
        server.predict(
            {
                "images": {"base_camera": np.zeros((8, 8, 3), dtype=np.uint8)},
                "task_description": "move",
            }
        )


def test_eval_protocol_buffers_chunks_and_resets_per_client(monkeypatch: pytest.MonkeyPatch) -> None:
    import msgpack

    class _WebSocket:
        remote_address = ("127.0.0.1", 1234)

        def __init__(self) -> None:
            self.sent: list[bytes] = []
            self.messages = [
                msgpack.packb({"type": "predict", "payload": {"observation": {}}, "seq": 1}),
                msgpack.packb({"type": "predict", "payload": {"observation": {}}, "seq": 2}),
                msgpack.packb({"type": "reset", "payload": {}, "seq": 3}),
                msgpack.packb({"type": "predict", "payload": {"observation": {}}, "seq": 4}),
            ]

        def __aiter__(self):
            self._iterator = iter(self.messages)
            return self

        async def __anext__(self):
            try:
                return next(self._iterator)
            except StopIteration as exc:
                raise StopAsyncIteration from exc

        async def send(self, payload: bytes) -> None:
            self.sent.append(payload)

    server = ForgeModelServer("missing.pt", device="cpu", chunk_size=3)
    calls: list[int] = []

    def predict(*_args, **_kwargs):
        calls.append(1)
        return {"actions": np.arange(21, dtype=np.float32).reshape(3, 7)}

    monkeypatch.setattr(server, "predict", predict)
    websocket = _WebSocket()

    asyncio.run(server._handle_client(websocket))

    decoded = [msgpack.unpackb(payload, raw=False)["payload"]["actions"] for payload in websocket.sent]
    assert calls == [1, 1]
    assert decoded == [list(range(7)), list(range(7, 14)), list(range(7))]


def test_eval_protocol_action_rejects_invalid_shapes() -> None:
    buffer: deque[np.ndarray] = deque()
    server = ForgeModelServer("missing.pt", device="cpu", chunk_size=3)

    with pytest.raises(RuntimeError, match="checkpoint requires"):
        server._next_protocol_action(np.zeros((1, 2, 3)), buffer)


@pytest.mark.parametrize("shape", [(6,), (3, 6), (2, 7), (3, 8)])
def test_eval_protocol_action_requires_exact_checkpoint_shape(shape: tuple[int, ...]) -> None:
    server = ForgeModelServer("missing.pt", device="cpu", chunk_size=3)

    with pytest.raises(RuntimeError, match="checkpoint requires"):
        server._next_protocol_action(np.zeros(shape, dtype=np.float32), deque())


def test_eval_protocol_accepts_exact_single_action_with_chunked_checkpoint() -> None:
    server = ForgeModelServer("missing.pt", device="cpu", chunk_size=3)
    action = np.arange(7, dtype=np.float32)

    returned = server._next_protocol_action(action, deque())

    np.testing.assert_array_equal(returned, action)
