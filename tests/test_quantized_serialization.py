"""Real packed-checkpoint contracts for INT4/INT8 artifacts."""

from __future__ import annotations

import torch
from torch import nn

from forge.checkpoint_compat import extract_checkpoint_state_dict, load_model_weights_with_compatibility
from forge.quantize.serialization import PACKED_STATE_KEY, pack_state_dict, unpack_state_dict


def test_int4_pack_is_compact_and_roundtrips_shapes() -> None:
    state = {
        "weight": torch.linspace(-2, 2, 1024, dtype=torch.float32).reshape(32, 32),
        "bias": torch.linspace(-1, 1, 32, dtype=torch.bfloat16),
        "counter": torch.tensor(7, dtype=torch.int64),
    }

    packed, metadata = pack_state_dict(state, bits=4)
    restored = unpack_state_dict(packed)

    assert metadata["schema"] == "forge.packed-state.v1"
    assert metadata["compression_ratio"] > 4.0
    assert restored.keys() == state.keys()
    assert restored["weight"].shape == state["weight"].shape
    assert restored["weight"].dtype == torch.float32
    assert restored["bias"].dtype == torch.bfloat16
    assert torch.equal(restored["counter"], state["counter"])
    assert torch.mean((restored["weight"] - state["weight"]) ** 2).item() < 0.01


def test_int8_pack_has_lower_error_than_int4() -> None:
    state = {"weight": torch.randn(64, 64)}
    packed4, _ = pack_state_dict(state, bits=4)
    packed8, _ = pack_state_dict(state, bits=8)

    error4 = torch.mean((unpack_state_dict(packed4)["weight"] - state["weight"]) ** 2)
    error8 = torch.mean((unpack_state_dict(packed8)["weight"] - state["weight"]) ** 2)
    assert error8 < error4


def test_checkpoint_extractor_understands_packed_state() -> None:
    state = {"weight": torch.randn(4, 4)}
    packed, metadata = pack_state_dict(state, bits=4)

    restored, source = extract_checkpoint_state_dict({PACKED_STATE_KEY: packed, "quantization": metadata})

    assert source == PACKED_STATE_KEY
    assert restored is not None
    assert restored["weight"].shape == state["weight"].shape


def test_packed_checkpoint_loads_through_normal_model_boundary() -> None:
    source = nn.Linear(16, 8)
    target = nn.Linear(16, 8)
    packed, metadata = pack_state_dict(source.state_dict(), bits=8)
    restored, source_key = extract_checkpoint_state_dict({PACKED_STATE_KEY: packed, "quantization": metadata})

    assert restored is not None
    _, report = load_model_weights_with_compatibility(
        target,
        restored,
        context="packed-unit",
        minimum_coverage=1.0,
    )
    assert source_key == PACKED_STATE_KEY
    assert report.coverage_fraction == 1.0
    inputs = torch.randn(4, 16)
    assert torch.mean((source(inputs) - target(inputs)) ** 2).item() < 1e-3
