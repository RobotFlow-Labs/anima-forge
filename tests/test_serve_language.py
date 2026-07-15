"""Truthful language-input contracts for the public inference server."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from forge.serve import _tokenize_instructions


class _Tokenizer:
    def __init__(self) -> None:
        self.instructions: list[str] = []

    def __call__(self, instructions: list[str], **kwargs: object) -> dict[str, torch.Tensor]:
        self.instructions = instructions
        assert kwargs == {
            "return_tensors": "pt",
            "padding": True,
            "truncation": True,
            "max_length": 64,
        }
        values = torch.arange(len(instructions) * 3, dtype=torch.long).reshape(len(instructions), 3)
        return {"input_ids": values}


def test_serve_tokenizes_every_instruction_in_a_batch() -> None:
    tokenizer = _Tokenizer()
    student = SimpleNamespace(tokenizer=tokenizer, component_provenance={"language": "real"})

    ids = _tokenize_instructions(
        student,
        ["pick up the block", "close the drawer"],
        device="cpu",
        allow_mock=False,
    )

    assert tokenizer.instructions == ["pick up the block", "close the drawer"]
    assert ids.tolist() == [[0, 1, 2], [3, 4, 5]]


def test_serve_refuses_missing_real_tokenizer() -> None:
    student = SimpleNamespace(tokenizer=None, component_provenance={"language": "real"})

    with pytest.raises(RuntimeError, match="will not invent language tokens"):
        _tokenize_instructions(student, ["move"], device="cpu", allow_mock=False)


def test_explicit_mock_serve_uses_deterministic_tokens() -> None:
    student = SimpleNamespace(tokenizer=None, component_provenance={"language": "mock"})

    first = _tokenize_instructions(student, ["move", "stop"], device="cpu", allow_mock=True)
    second = _tokenize_instructions(student, ["move", "stop"], device="cpu", allow_mock=True)

    assert first.shape == (2, 1)
    assert torch.equal(first, second)
    assert torch.count_nonzero(first) == 0


@pytest.mark.parametrize("instructions", [[], [""], ["move", "   "]])
def test_serve_rejects_empty_instructions(instructions: list[str]) -> None:
    student = SimpleNamespace(tokenizer=_Tokenizer(), component_provenance={"language": "real"})

    with pytest.raises(ValueError, match="non-empty instruction"):
        _tokenize_instructions(student, instructions, device="cpu", allow_mock=False)
