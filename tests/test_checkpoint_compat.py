"""Compatibility regression tests for modern PyTorch/transformers checkpoints."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from forge.checkpoint_compat import (
    apply_checkpoint_structure,
    extract_checkpoint_state_dict,
    filter_state_dict_by_shape,
    load_checkpoint_payload,
    load_model_weights_with_compatibility,
    strip_known_prefixes,
    summarize_checkpoint_report,
)


def test_load_checkpoint_payload_uses_safe_torch_format(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.pt"
    torch.save({"student_state_dict": {"weight": torch.ones(2, 2)}}, path)

    payload = load_checkpoint_payload(str(path))

    assert payload is not None
    assert torch.equal(payload["student_state_dict"]["weight"], torch.ones(2, 2))
    assert load_checkpoint_payload(str(tmp_path / "missing.pt")) is None


@pytest.mark.parametrize("key", ["model_state_dict", "student_state_dict", "state_dict", "model"])
def test_extract_checkpoint_state_dict_supports_current_wrappers(key: str) -> None:
    state = {"weight": torch.ones(1)}

    extracted, source = extract_checkpoint_state_dict({key: state})

    assert extracted is state
    assert source == key


@pytest.mark.parametrize(
    ("prefix", "expected"),
    [
        ("module.", "weight"),
        ("model.", "weight"),
        ("student.", "weight"),
        ("module.student.", "weight"),
        ("model.student.", "weight"),
    ],
)
def test_strip_known_prefixes_prefers_complete_wrapper(prefix: str, expected: str) -> None:
    normalized, stripped = strip_known_prefixes({f"{prefix}weight": torch.ones(1)})

    assert set(normalized) == {expected}
    assert stripped == prefix


def test_shape_filter_counts_non_tensor_and_mismatch() -> None:
    model_state = {"weight": torch.zeros(2, 2), "bias": torch.zeros(2)}
    candidate = {
        "weight": torch.ones(3, 2),
        "bias": torch.ones(2),
        "metadata": "not-a-tensor",
        "extra": torch.ones(1),
    }

    compatible, skipped, mismatched = filter_state_dict_by_shape(model_state, candidate)

    assert set(compatible) == {"bias"}
    assert skipped == 1
    assert mismatched == 2


def test_compatibility_loader_falls_back_to_shape_filtered_weights() -> None:
    model = nn.Linear(2, 2)
    bias = torch.tensor([3.0, 4.0])
    state = {"module.weight": torch.ones(3, 2), "module.bias": bias}

    compatibility, report = load_model_weights_with_compatibility(model, state, context="unit")

    assert report.load_mode == "shape_filtered"
    assert report.normalized_prefix == "module."
    assert report.shape_compatible_count == 1
    assert report.mismatched_shape_count == 1
    assert torch.equal(model.bias, bias)
    assert "weight" in compatibility.missing_keys
    assert "shape_compatible=1" in summarize_checkpoint_report("unit", report)


def test_compatibility_loader_rejects_zero_compatible_tensors() -> None:
    model = nn.Linear(2, 2)
    state = {"weight": torch.ones(3, 3)}

    with pytest.raises(RuntimeError, match="No compatible tensor keys"):
        load_model_weights_with_compatibility(model, state, context="unit")


def test_apply_checkpoint_structure_recreates_pruned_depth() -> None:
    class AttentionLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = nn.Linear(2, 2)

    class ToyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([AttentionLayer() for _ in range(6)])
            self.config = SimpleNamespace(num_hidden_layers=6)

    model = ToyModel()
    apply_checkpoint_structure(
        model,
        {
            "pruning": {
                "removed_layers": [2, 3],
                "pre_prune_layer_count": 6,
                "target_layers": 4,
            }
        },
    )
    assert len(model.layers) == 4
    assert model.config.num_hidden_layers == 4


@pytest.mark.parametrize(
    "pruning",
    [
        {"removed_layers": [1, 1], "pre_prune_layer_count": 6, "target_layers": 4},
        {"removed_layers": [-1], "pre_prune_layer_count": 6, "target_layers": 5},
        {"removed_layers": [6], "pre_prune_layer_count": 6, "target_layers": 5},
        {"removed_layers": [1], "pre_prune_layer_count": 5, "target_layers": 4},
        {"removed_layers": [1], "pre_prune_layer_count": 6, "target_layers": 3},
    ],
)
def test_invalid_pruning_metadata_never_partially_mutates_model(pruning: dict[str, object]) -> None:
    class AttentionLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = nn.Linear(2, 2)

    class ToyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([AttentionLayer() for _ in range(6)])
            self.config = SimpleNamespace(num_hidden_layers=6)

    model = ToyModel()
    original_layers = model.layers
    original_layer_ids = [id(layer) for layer in model.layers]

    with pytest.raises(ValueError, match="Checkpoint pruning"):
        apply_checkpoint_structure(model, {"pruning": pruning})

    assert model.layers is original_layers
    assert [id(layer) for layer in model.layers] == original_layer_ids
    assert model.config.num_hidden_layers == 6


def test_pruned_checkpoint_requires_pre_prune_layer_count() -> None:
    class AttentionLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = nn.Linear(2, 2)

    class ToyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([AttentionLayer() for _ in range(4)])

    with pytest.raises(ValueError, match="pre_prune_layer_count"):
        apply_checkpoint_structure(ToyModel(), {"pruning": {"removed_layers": [1]}})


def test_inconsistent_model_config_is_rejected_before_layer_mutation() -> None:
    class AttentionLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = nn.Linear(2, 2)

    class ToyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([AttentionLayer() for _ in range(6)])
            self.config = SimpleNamespace(num_hidden_layers=5)

    model = ToyModel()
    original_layers = model.layers

    with pytest.raises(ValueError, match="does not match model config"):
        apply_checkpoint_structure(
            model,
            {
                "pruning": {
                    "removed_layers": [1],
                    "pre_prune_layer_count": 6,
                    "target_layers": 5,
                }
            },
        )

    assert model.layers is original_layers
    assert len(model.layers) == 6
    assert model.config.num_hidden_layers == 5


def test_apply_checkpoint_structure_ignores_unrelated_sibling_configs() -> None:
    class AttentionLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = nn.Linear(2, 2)

    class VisionEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([AttentionLayer() for _ in range(6)])

    class ToyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.vision = VisionEncoder()
            self.language = SimpleNamespace(config=SimpleNamespace(num_hidden_layers=28))

    model = ToyModel()
    apply_checkpoint_structure(
        model,
        {
            "pruning": {
                "removed_layers": [2, 3],
                "pre_prune_layer_count": 6,
                "target_layers": 4,
            }
        },
    )

    assert len(model.vision.layers) == 4
    assert model.language.config.num_hidden_layers == 28
