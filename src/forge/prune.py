"""PRD-04: Layer Pruning (Shallow-Pi methodology).

Reduces transformer depth from N layers to N/3 with <1% quality loss.
Key insight: middle layers in VLA transformers contain redundant representations.

Two scoring methods:
1. Angular distance: how much does each layer change its input?
2. Action sensitivity: how much does skipping a layer affect actions?

After pruning, a short recovery fine-tune restores quality.

Usage:
    forge prune run --config configs/forge_nano.yaml --target-layers 8
"""

from __future__ import annotations

import copy
import logging
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.utils.data import RandomSampler

from forge.config import ForgeConfig, PruningConfig
from forge.errors import ForgeDataNotFoundError

logger = logging.getLogger(__name__)


def _forward_with_layer_outputs(
    model: nn.Module,
    layers: list[nn.Module],
    images: torch.Tensor,
) -> tuple[Any, dict[int, Any]]:
    """Run one baseline forward while observing each decoder layer's output contract."""
    observed: dict[int, Any] = {}
    handles = []

    for index, layer in enumerate(layers):

        def capture(_module: nn.Module, _inputs: tuple[Any, ...], output: Any, *, layer_index: int = index) -> None:
            observed[layer_index] = output

        handles.append(layer.register_forward_hook(capture))

    try:
        output = model(images)
    finally:
        for handle in handles:
            handle.remove()

    missing = sorted(set(range(len(layers))) - observed.keys())
    if missing:
        raise RuntimeError(f"Baseline forward did not execute decoder layer(s): {missing}")
    return output, observed


def _identity_forward_for_observed_output(observed_output: Any) -> Callable[..., Any]:
    """Build an identity that preserves an observed decoder layer return container."""
    if torch.is_tensor(observed_output):
        output_kind = "tensor"
        auxiliary: tuple[Any, ...] = ()
    elif isinstance(observed_output, tuple) and observed_output:
        output_kind = "tuple"
        auxiliary = tuple(observed_output[1:])
    elif isinstance(observed_output, list) and observed_output:
        output_kind = "list"
        auxiliary = tuple(observed_output[1:])
    else:
        raise RuntimeError(
            "Unsupported decoder layer output contract while computing pruning importance: "
            f"{type(observed_output).__name__}"
        )

    def identity_forward(*args: Any, **kwargs: Any) -> Any:
        hidden_states = args[0] if args else kwargs.get("hidden_states")
        if not torch.is_tensor(hidden_states):
            raise RuntimeError("Decoder identity did not receive tensor hidden_states")
        if output_kind == "tuple":
            return (hidden_states, *auxiliary)
        if output_kind == "list":
            return [hidden_states, *auxiliary]
        return hidden_states

    return identity_forward


def _validate_importance_scores(
    scores: dict[int, float],
    n_layers: int,
    *,
    context: str,
) -> None:
    """Reject incomplete, non-finite, or all-zero scores before layer selection."""
    expected = set(range(n_layers))
    if set(scores) != expected:
        missing = sorted(expected - set(scores))
        extra = sorted(set(scores) - expected)
        raise RuntimeError(f"{context} layer scores are incomplete (missing={missing}, extra={extra})")
    if any(not math.isfinite(float(score)) for score in scores.values()):
        raise RuntimeError(f"{context} layer scores contain non-finite values")
    if not any(abs(float(score)) > 1e-12 for score in scores.values()):
        raise RuntimeError(f"{context} layer scores are all zero; refusing arbitrary pruning")


def compute_layer_importance(
    model: nn.Module,
    calibration_data: list[dict],
    n_samples: int = 100,
) -> dict[int, float]:
    """Score each transformer layer by contribution to action prediction.

    Combines:
    - Angular distance (60%): How much does this layer transform its input?
    - Action sensitivity (40%): How much does removing this layer affect actions?

    Returns:
        dict mapping layer_index → importance_score (higher = more important)
    """
    transformer_layers = _find_transformer_layers(model)
    if not transformer_layers:
        logger.warning("No transformer layers found in model")
        return {}

    n_layers = len(transformer_layers)
    logger.info(
        "Computing importance for %s transformer layers on %s samples",
        n_layers,
        min(n_samples, len(calibration_data)),
    )

    angular_scores = _compute_angular_distances(model, transformer_layers, calibration_data, n_samples)
    action_scores = _compute_action_sensitivity(model, transformer_layers, calibration_data, n_samples)

    raw_scores = {index: angular_scores[index] + action_scores[index] for index in range(n_layers)}
    _validate_importance_scores(raw_scores, n_layers, context="Computed pruning importance")

    # Normalize to [0, 1]
    angular_scores = _normalize(angular_scores)
    action_scores = _normalize(action_scores)

    # Combined score: 60% angular + 40% action sensitivity
    combined = {}
    for idx in range(n_layers):
        combined[idx] = 0.6 * angular_scores.get(idx, 0) + 0.4 * action_scores.get(idx, 0)

    return combined


def _compute_angular_distances(
    model: nn.Module,
    layers: list[nn.Module],
    data: list[dict],
    n_samples: int,
) -> dict[int, float]:
    """Measure how much each layer changes its input (cosine distance)."""
    scores = {i: 0.0 for i in range(len(layers))}
    hidden_states = {}

    def make_hook(layer_idx: int, position: str):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output
            hidden_states[f"{layer_idx}_{position}"] = h.detach()

        return hook_fn

    # Register hooks before and after each layer
    handles = []
    for i, layer in enumerate(layers):
        # Hook on the layer to capture output
        h = layer.register_forward_hook(make_hook(i, "after"))
        handles.append(h)

    model.eval()
    samples_used = 0
    try:
        with torch.no_grad():
            for sample_index, batch in enumerate(data[:n_samples]):
                images = batch["image"].unsqueeze(0) if batch["image"].dim() == 3 else batch["image"]
                try:
                    model(images)
                except Exception as exc:
                    raise RuntimeError(f"Angular importance baseline failed for sample {sample_index}") from exc

                # Compute angular distance between consecutive layer outputs
                for i in range(len(layers) - 1):
                    key_curr = f"{i}_after"
                    key_next = f"{i + 1}_after"
                    if key_curr not in hidden_states or key_next not in hidden_states:
                        raise RuntimeError(
                            f"Angular importance did not observe consecutive decoder layers {i} and {i + 1}"
                        )
                    h_before = hidden_states[key_curr].flatten(1)
                    h_after = hidden_states[key_next].flatten(1)
                    cos_sim = functional.cosine_similarity(h_before, h_after, dim=-1).mean()
                    angular_dist = 1.0 - cos_sim.item()
                    scores[i + 1] += angular_dist

                hidden_states.clear()
                samples_used += 1
    finally:
        for h in handles:
            h.remove()

    # Average
    if samples_used > 0:
        scores = {k: v / samples_used for k, v in scores.items()}

    return scores


def _compute_action_sensitivity(
    model: nn.Module,
    layers: list[nn.Module],
    data: list[dict],
    n_samples: int,
) -> dict[int, float]:
    """Measure action change when each layer is skipped."""
    scores = {i: 0.0 for i in range(len(layers))}

    model.eval()
    samples_used = 0

    with torch.no_grad():
        for sample_index, batch in enumerate(data[: min(n_samples, 50)]):  # Fewer samples for speed
            images = batch["image"].unsqueeze(0) if batch["image"].dim() == 3 else batch["image"]

            # Baseline action
            try:
                baseline_out, observed_outputs = _forward_with_layer_outputs(model, layers, images)
                baseline_actions = baseline_out["actions"]
            except Exception as exc:
                raise RuntimeError(f"Action-sensitivity baseline failed for sample {sample_index}") from exc

            # Skip each layer and measure delta
            for i, layer in enumerate(layers):
                # Temporarily replace layer with identity
                original_forward = layer.forward

                layer.forward = _identity_forward_for_observed_output(observed_outputs[i])
                try:
                    skip_out = model(images)
                    skip_actions = skip_out["actions"]
                    delta = functional.mse_loss(baseline_actions, skip_actions).item()
                    scores[i] += delta
                except Exception as exc:
                    raise RuntimeError(
                        f"Action-sensitivity scoring failed for sample {sample_index}, layer {i}"
                    ) from exc
                finally:
                    layer.forward = original_forward

            samples_used += 1

    if samples_used > 0:
        scores = {k: v / samples_used for k, v in scores.items()}

    return scores


def prune_layers(
    model: nn.Module,
    layer_scores: dict[int, float],
    config: PruningConfig,
) -> tuple[nn.Module, list[int]]:
    """Remove least important transformer layers.

    Args:
        model: Student model with transformer layers
        layer_scores: importance scores per layer (higher = keep)
        config: Pruning configuration

    Returns:
        (pruned_model, removed_layer_indices)
    """
    transformer_layers = _find_transformer_layers(model)
    n_current = len(transformer_layers)
    n_remove = n_current - config.target_layers

    if n_remove <= 0:
        logger.info(f"Model has {n_current} layers, target is {config.target_layers}. No pruning needed.")
        return model, []

    _validate_importance_scores(layer_scores, n_current, context="Provided pruning importance")

    # Sort by importance (ascending) — remove least important
    ranked = sorted(layer_scores.items(), key=lambda x: x[1])

    # Always keep first N and last N layers
    removable = [idx for idx, score in ranked if idx >= config.keep_first_n and idx < n_current - config.keep_last_n]

    layers_to_remove = removable[:n_remove]

    if len(layers_to_remove) < n_remove:
        # Not enough removable layers with constraints — relax
        logger.warning(f"Only {len(layers_to_remove)} removable layers (need {n_remove}). Relaxing constraints.")
        remaining = [idx for idx, _ in ranked if idx not in layers_to_remove]
        layers_to_remove.extend(remaining[: n_remove - len(layers_to_remove)])

    layers_to_remove = sorted(layers_to_remove)

    logger.info(f"Pruning layers: {layers_to_remove} ({n_current} → {n_current - len(layers_to_remove)})")

    # Execute pruning
    pruned_model = _remove_layers(model, layers_to_remove)

    return pruned_model, layers_to_remove


def compute_activation_layer_importance(
    model: nn.Module,
    calibration_data: list[dict],
    n_samples: int = 8,
) -> dict[int, float]:
    """Score layers from real activation changes with one forward per sample."""
    layers = _find_transformer_layers(model)
    if not layers:
        return {}
    raw_scores = _compute_angular_distances(
        model,
        layers,
        calibration_data,
        min(n_samples, len(calibration_data)),
    )
    _validate_importance_scores(raw_scores, len(layers), context="Computed activation importance")
    return _normalize(raw_scores)


def _remove_layers(model: nn.Module, layer_indices: list[int]) -> nn.Module:
    """Remove specific transformer layers from the model."""
    model = copy.deepcopy(model)
    return apply_pruning_structure(
        model,
        layer_indices,
        pre_prune_layer_count=len(_find_transformer_layers(model)),
    )


def apply_pruning_structure(
    model: nn.Module,
    layer_indices: list[int],
    *,
    pre_prune_layer_count: int,
) -> nn.Module:
    """Remove recorded transformer layers in place for checkpoint reconstruction."""
    transformer_layers = _find_transformer_layers(model)
    if not transformer_layers:
        raise ValueError("Cannot apply pruning metadata: no transformer layers found")
    parent, attr_name = _find_layer_parent(model)
    if parent is None or attr_name is None:
        raise ValueError("Cannot apply pruning metadata: transformer layer parent not found")

    current_layers = getattr(parent, attr_name, None)
    if not isinstance(current_layers, nn.ModuleList) or len(current_layers) != len(transformer_layers):
        raise ValueError("Cannot apply pruning metadata: resolved transformer layer container is inconsistent")
    if isinstance(pre_prune_layer_count, bool) or not isinstance(pre_prune_layer_count, int):
        raise ValueError("Checkpoint pruning.pre_prune_layer_count must be an integer")
    if pre_prune_layer_count < 1 or len(current_layers) != pre_prune_layer_count:
        raise ValueError(
            "Checkpoint pruning pre-prune layer count does not match model structure: "
            f"metadata={pre_prune_layer_count}, model={len(current_layers)}"
        )
    if not isinstance(layer_indices, list) or not all(
        isinstance(index, int) and not isinstance(index, bool) for index in layer_indices
    ):
        raise ValueError("Checkpoint pruning.removed_layers must be a list of integers")
    if len(set(layer_indices)) != len(layer_indices):
        raise ValueError("Checkpoint pruning.removed_layers contains duplicate indices")
    invalid = [index for index in layer_indices if index < 0 or index >= pre_prune_layer_count]
    if invalid:
        raise ValueError(
            f"Checkpoint pruning.removed_layers contains indices outside the pre-prune structure: {invalid}"
        )
    if len(layer_indices) >= pre_prune_layer_count:
        raise ValueError("Checkpoint pruning metadata would remove every transformer layer")

    configs = []
    seen_configs: set[int] = set()
    for candidate in (parent,):
        config = getattr(candidate, "config", None)
        if config is None or not hasattr(config, "num_hidden_layers") or id(config) in seen_configs:
            continue
        seen_configs.add(id(config))
        if config.num_hidden_layers != pre_prune_layer_count:
            raise ValueError(
                "Checkpoint pruning pre-prune layer count does not match model config: "
                f"metadata={pre_prune_layer_count}, config={config.num_hidden_layers}"
            )
        configs.append(config)

    keep_indices = [i for i in range(pre_prune_layer_count) if i not in set(layer_indices)]
    new_layers = nn.ModuleList([current_layers[i] for i in keep_indices])
    setattr(parent, attr_name, new_layers)
    for config in configs:
        config.num_hidden_layers = len(new_layers)

    return model


def recovery_finetune(
    model: nn.Module,
    config: ForgeConfig,
    device: str = "cpu",
    max_steps: int | None = None,
) -> nn.Module:
    """Short recovery fine-tune after pruning.

    Only trains LoRA + action head with task loss (no KD).
    """
    max_steps = max_steps or config.pruning.recovery_steps
    data_dir = Path(config.paths.data_dir) / "teacher_labels"

    if not (data_dir / "metadata.json").is_file():
        if not config.student.allow_mock:
            raise ForgeDataNotFoundError(
                f"Teacher labels not found at {data_dir}. Recovery fine-tuning requires "
                "provenance-verified real labels; run `forge pipeline --stage labels` first."
            )
        from forge.distill import _create_mock_dataset

        logger.warning("Recovery labels are missing; explicit allow_mock permits generated test labels")
        dataset = _create_mock_dataset(data_dir, n_episodes=50)
    else:
        from forge.data.teacher_dataset import TeacherLabelDataset

        dataset = TeacherLabelDataset(str(data_dir))
    if dataset.labels_provenance != "real" and not config.student.allow_mock:
        dataset.close()
        raise ForgeDataNotFoundError(
            f"Teacher labels at {data_dir} are mock-derived or untrusted. Recovery fine-tuning "
            "requires real labels unless student.allow_mock is explicitly enabled."
        )
    if len(dataset) < 1:
        dataset.close()
        raise ForgeDataNotFoundError(f"Recovery teacher-label dataset is empty: {data_dir}")

    sampler = None
    if len(dataset) < config.distill.batch_size:
        sampler = RandomSampler(
            dataset,
            replacement=True,
            num_samples=config.distill.batch_size,
        )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.distill.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        drop_last=True,
    )

    # Only train action head + LoRA
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if "action_head" in name or "lora" in name:
            param.requires_grad = True

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.pruning.recovery_lr)

    model.train()
    model = model.to(device)
    data_iter = iter(dataloader)

    for step in range(max_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        images = batch["image"].to(device)
        gt_actions = batch["ground_truth_actions"].to(device)

        out = model(images, gt_actions=gt_actions)
        loss = functional.mse_loss(out["actions"], gt_actions)
        if "loss" in out:
            loss = loss + out["loss"]

        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

        if step % 100 == 0:
            logger.info(f"Recovery step {step}/{max_steps}: loss={loss.item():.4f}")

    return model


def _find_transformer_layers(model: nn.Module) -> list[nn.Module]:
    """Find the transformer decoder/encoder layers in a model."""
    # Search for common layer container names
    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleList):
            # Check if children look like transformer layers
            children = list(module.children())
            if len(children) >= 4:  # At least 4 layers
                # Check if they have attention-like submodules
                first = children[0]
                has_attn = any("attn" in n.lower() or "attention" in n.lower() for n, _ in first.named_modules())
                has_norm = any("norm" in n.lower() or "layernorm" in n.lower() for n, _ in first.named_modules())
                if has_attn or has_norm:
                    return children

    # Fallback: look in common paths
    for attr_path in [
        "language.model.layers",
        "language.layers",
        "model.layers",
        "transformer.layers",
        "encoder.layers",
        "decoder.layers",
    ]:
        try:
            module = model
            for attr in attr_path.split("."):
                module = getattr(module, attr)
            if isinstance(module, nn.ModuleList) and len(module) >= 4:
                return list(module.children())
        except AttributeError:
            continue

    return []


def _find_layer_parent(model: nn.Module) -> tuple[nn.Module | None, str | None]:
    """Find the parent module and attribute name of the transformer layers."""
    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleList):
            children = list(module.children())
            if len(children) >= 4:
                first = children[0]
                has_attn = any("attn" in n.lower() or "attention" in n.lower() for n, _ in first.named_modules())
                if has_attn:
                    parts = name.rsplit(".", 1)
                    if len(parts) == 2:
                        parent = dict(model.named_modules())[parts[0]]
                        return parent, parts[1]
                    else:
                        return model, name

    return None, None


def _normalize(scores: dict[int, float]) -> dict[int, float]:
    """Normalize scores to [0, 1]."""
    if not scores:
        return scores
    values = list(scores.values())
    min_v, max_v = min(values), max(values)
    range_v = max_v - min_v
    if range_v < 1e-8:
        return {k: 0.5 for k in scores}
    return {k: (v - min_v) / range_v for k, v in scores.items()}
