"""Chunk-Aware Layer Pruning.

Key insight: Layer importance for action chunks isn't just about accuracy —
it's about temporal coherence. A layer might have low single-step importance
but be critical for maintaining smooth multi-step predictions.

Metrics:
1. Standard: LayerDrop-style importance (Sajjad et al.)
2. Temporal: Measure how much removing a layer increases
   the variance between consecutive steps in the chunk
3. Combined: importance = α * standard + (1-α) * temporal_coherence

Based on:
- LayerDrop (Fan et al., 2020)
- Shallow-Pi pruning (PRD-04 v1)
- FORGE v2 temporal coherence design
"""

from __future__ import annotations

import copy
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from forge.prune import (
    _find_layer_parent,
    _find_transformer_layers,
    _forward_with_layer_outputs,
    _identity_forward_for_observed_output,
    _normalize,
    _validate_importance_scores,
)

logger = logging.getLogger(__name__)


def temporal_coherence_score(action_chunk: torch.Tensor) -> float:
    """Measure smoothness of an action chunk.

    Lower is smoother (less jitter between consecutive steps).

    Score = mean(||a_{t+1} - a_t||^2) / mean(||a_t||^2)

    Args:
        action_chunk: (H, D_action) or (B, H, D_action)

    Returns:
        Temporal coherence score (0 = perfectly smooth)
    """
    if action_chunk.dim() == 2:
        action_chunk = action_chunk.unsqueeze(0)

    if action_chunk.shape[1] < 2:
        return 0.0

    # Consecutive differences
    diffs = action_chunk[:, 1:] - action_chunk[:, :-1]  # (B, H-1, D)
    jitter = (diffs**2).mean()

    # Normalize by action magnitude
    magnitude = (action_chunk**2).mean().clamp(min=1e-8)

    return (jitter / magnitude).item()


def compute_chunk_layer_importance(
    model: nn.Module,
    calibration_data: list[torch.Tensor],
    action_horizon: int = 8,
    alpha: float = 0.6,
) -> dict[int, float]:
    """Compute layer importance considering temporal coherence.

    For each layer:
    1. Run model with layer → get chunk predictions
    2. Run model without layer → get chunk predictions
    3. Measure: accuracy drop + temporal coherence drop

    Args:
        model: Student model with transformer layers
        calibration_data: List of input image tensors (B, C, H, W)
        action_horizon: Expected chunk length
        alpha: Weight for standard importance vs temporal (α*standard + (1-α)*temporal)

    Returns:
        {layer_idx: importance_score} — higher means more important
    """
    transformer_layers = _find_transformer_layers(model)
    if not transformer_layers:
        logger.warning("No transformer layers found")
        return {}

    n_layers = len(transformer_layers)
    standard_scores = {i: 0.0 for i in range(n_layers)}
    temporal_scores = {i: 0.0 for i in range(n_layers)}

    model.eval()
    samples_used = 0

    with torch.no_grad():
        for sample_index, images in enumerate(calibration_data):
            if images.dim() == 3:
                images = images.unsqueeze(0)

            # Baseline prediction
            try:
                baseline_out, observed_outputs = _forward_with_layer_outputs(model, transformer_layers, images)
                baseline_actions = baseline_out["actions"]
            except Exception as exc:
                raise RuntimeError(f"Chunk importance baseline failed for sample {sample_index}") from exc

            # Compute baseline temporal coherence if actions are chunked
            if baseline_actions.dim() == 3 and baseline_actions.shape[1] > 1:
                baseline_tc = temporal_coherence_score(baseline_actions)
            else:
                baseline_tc = 0.0

            # Skip each layer and measure both action delta and TC delta
            for i, layer in enumerate(transformer_layers):
                original_forward = layer.forward

                layer.forward = _identity_forward_for_observed_output(observed_outputs[i])
                try:
                    skip_out = model(images)
                    skip_actions = skip_out["actions"]

                    # Standard importance: action MSE when layer is skipped
                    action_delta = F.mse_loss(baseline_actions, skip_actions).item()
                    standard_scores[i] += action_delta

                    # Temporal importance: TC degradation when layer is skipped
                    if skip_actions.dim() == 3 and skip_actions.shape[1] > 1:
                        skip_tc = temporal_coherence_score(skip_actions)
                        tc_delta = max(0.0, skip_tc - baseline_tc)
                    else:
                        tc_delta = 0.0
                    temporal_scores[i] += tc_delta

                except Exception as exc:
                    raise RuntimeError(f"Chunk importance failed for sample {sample_index}, layer {i}") from exc
                finally:
                    layer.forward = original_forward

            samples_used += 1

    if samples_used > 0:
        standard_scores = {k: v / samples_used for k, v in standard_scores.items()}
        temporal_scores = {k: v / samples_used for k, v in temporal_scores.items()}

    raw_scores = {index: standard_scores[index] + temporal_scores[index] for index in range(n_layers)}
    _validate_importance_scores(raw_scores, n_layers, context="Computed chunk importance")

    # Normalize each set of scores
    standard_norm = _normalize(standard_scores)
    temporal_norm = _normalize(temporal_scores)

    # Combined importance
    combined = {}
    for idx in range(n_layers):
        combined[idx] = alpha * standard_norm.get(idx, 0) + (1 - alpha) * temporal_norm.get(idx, 0)

    return combined


def prune_chunk_aware(
    model: nn.Module,
    importance_scores: dict[int, float],
    target_layers: int,
    keep_first_n: int = 2,
    keep_last_n: int = 2,
) -> tuple[nn.Module, list[int]]:
    """Prune layers while preserving temporal coherence.

    Removes the least important layers according to chunk-aware importance
    scores, always keeping the first and last N layers.

    Args:
        model: Student model
        importance_scores: {layer_idx: score} from compute_chunk_layer_importance
        target_layers: Desired number of layers after pruning
        keep_first_n: Always keep first N layers
        keep_last_n: Always keep last N layers

    Returns:
        (pruned_model, removed_layer_indices)
    """
    transformer_layers = _find_transformer_layers(model)
    n_current = len(transformer_layers)
    n_remove = n_current - target_layers

    if n_remove <= 0:
        logger.info(f"Model has {n_current} layers, target is {target_layers}. No pruning needed.")
        return model, []

    _validate_importance_scores(importance_scores, n_current, context="Provided chunk importance")

    # Sort by importance (ascending) — remove least important first
    ranked = sorted(importance_scores.items(), key=lambda x: x[1])

    # Filter to removable layers (respect keep_first_n and keep_last_n)
    removable = [idx for idx, _ in ranked if idx >= keep_first_n and idx < n_current - keep_last_n]

    layers_to_remove = removable[:n_remove]

    if len(layers_to_remove) < n_remove:
        logger.warning(
            f"Only {len(layers_to_remove)} removable layers (need {n_remove}). Relaxing boundary constraints."
        )
        remaining = [idx for idx, _ in ranked if idx not in layers_to_remove]
        layers_to_remove.extend(remaining[: n_remove - len(layers_to_remove)])

    layers_to_remove = sorted(layers_to_remove)
    logger.info(
        "Chunk-aware pruning: removing layers %s (%s → %s)",
        layers_to_remove,
        n_current,
        n_current - len(layers_to_remove),
    )

    # Deep copy and remove layers
    pruned_model = copy.deepcopy(model)
    parent, attr_name = _find_layer_parent(pruned_model)
    if parent is None or attr_name is None:
        raise ValueError("Cannot prune discovered transformer layers: their parent ModuleList could not be resolved")

    current_layers = getattr(parent, attr_name, None)
    if not isinstance(current_layers, nn.ModuleList) or len(current_layers) != n_current:
        raise ValueError("Cannot prune discovered transformer layers: resolved layer container does not match")

    keep_indices = [i for i in range(len(current_layers)) if i not in layers_to_remove]
    new_layers = nn.ModuleList([current_layers[i] for i in keep_indices])
    setattr(parent, attr_name, new_layers)

    if hasattr(pruned_model, "config") and hasattr(pruned_model.config, "num_hidden_layers"):
        pruned_model.config.num_hidden_layers = len(new_layers)

    return pruned_model, layers_to_remove


def attention_head_pruning(
    model: nn.Module,
    calibration_data: list[torch.Tensor],
    target_heads_ratio: float = 0.5,
) -> nn.Module:
    """Prune attention heads that contribute least to action quality.

    Measures each head's contribution to action prediction accuracy,
    then zeros out the least important heads.

    Args:
        model: Student model
        calibration_data: List of input image tensors
        target_heads_ratio: Fraction of heads to keep (0.5 = keep 50%)

    Returns:
        Model with least important heads zeroed out
    """
    model = copy.deepcopy(model)
    model.eval()

    # Find multi-head attention modules
    attn_modules = []
    for name, module in model.named_modules():
        if hasattr(module, "num_heads") and hasattr(module, "head_dim"):
            attn_modules.append((name, module))

    if not attn_modules:
        logger.warning("No attention modules with num_heads found")
        return model

    # Collect baseline actions
    baseline_actions_list = []
    with torch.no_grad():
        for images in calibration_data[:20]:
            if images.dim() == 3:
                images = images.unsqueeze(0)
            try:
                out = model(images)
                baseline_actions_list.append(out["actions"].clone())
            except Exception:
                continue

    if not baseline_actions_list:
        return model

    # For each attention module, score each head
    for name, module in attn_modules:
        if not hasattr(module, "num_heads"):
            continue

        n_heads = module.num_heads
        n_keep = max(1, int(n_heads * target_heads_ratio))
        head_dim = module.head_dim if hasattr(module, "head_dim") else module.embed_dim // n_heads

        # Score each head by masking it and measuring action change
        head_scores = torch.zeros(n_heads)

        # Find the output projection weight
        out_proj = None
        for attr in ["out_proj", "o_proj"]:
            if hasattr(module, attr):
                out_proj = getattr(module, attr)
                break

        if out_proj is None or not isinstance(out_proj, nn.Linear):
            continue

        with torch.no_grad():
            for head_idx in range(n_heads):
                # Temporarily zero out this head's contribution
                start = head_idx * head_dim
                end = start + head_dim
                original_weight = out_proj.weight.data[:, start:end].clone()
                out_proj.weight.data[:, start:end] = 0

                total_delta = 0.0
                for idx, images in enumerate(calibration_data[:10]):
                    if images.dim() == 3:
                        images = images.unsqueeze(0)
                    try:
                        out = model(images)
                        if idx < len(baseline_actions_list):
                            delta = F.mse_loss(out["actions"], baseline_actions_list[idx]).item()
                            total_delta += delta
                    except Exception:
                        pass

                head_scores[head_idx] = total_delta
                out_proj.weight.data[:, start:end] = original_weight

        # Keep top-k heads, zero out the rest
        _, keep_indices = head_scores.topk(n_keep)
        remove_indices = [i for i in range(n_heads) if i not in keep_indices]

        for head_idx in remove_indices:
            start = head_idx * head_dim
            end = start + head_dim
            out_proj.weight.data[:, start:end] = 0
            if out_proj.bias is not None:
                # Don't zero bias — it's shared
                pass

        logger.info(f"{name}: keeping {n_keep}/{n_heads} heads (removed {remove_indices})")

    return model
