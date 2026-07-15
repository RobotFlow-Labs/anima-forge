"""HuggingFace model card generation for FORGE students."""

from __future__ import annotations

import dataclasses

from forge.profiler.dataclasses import ModelProfileCard

_W = 45  # outer box interior width (chars between the two │ delimiters)


def _format_params(n: int) -> str:
    """Return "400.0M", "12.5K", or str(n) for a raw parameter count."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _short_name(path: str) -> str:
    """Return the last segment of a HuggingFace path or "--"-separated dir name."""
    for sep in ("/", "--"):
        if sep in path:
            return path.split(sep)[-1]
    return path


def _outer(content: str) -> str:
    return f"│{content:<{_W}}│"


def _inner(content: str, tag: str = "") -> str:
    """One row of the 22-char inner box with an optional right-side tag."""
    tag_zone = _W - 26  # chars available right of the inner │
    cell = f"{content[:20]:<20}"
    right = f" {tag:<{tag_zone - 1}}" if tag else " " * tag_zone
    return f"│  │  {cell}│{right}│"


def generate_ascii_diagram(card: ModelProfileCard) -> str:
    """Return an ASCII architecture diagram for *card*, all lines 47 chars wide."""
    vs = _short_name(card.vision_encoder)
    lm = _short_name(card.language_model)
    vp = _format_params(card.components[0].param_count) if card.components else "?M"
    lp = _format_params(card.components[2].param_count) if len(card.components) > 2 else "?M"
    nq: int = card.bridge_config.get("n_queries", 64)
    nl: int = card.bridge_config.get("n_layers", 4)
    dm: int = card.bridge_config.get("d_model", 896)
    rank: int = card.recommended_hp.lora_rank if card.recommended_hp is not None else 32

    rows = [
        "┌─────────────────────────────────────────────┐",
        _outer(f"  {card.model_name}"),
        "├─────────────────────────────────────────────┤",
        _outer("  Image (3×384×384)"),
        _outer("       ↓"),
        _outer("  ┌─────────────────────┐"),
        _inner(vs, "◄── Frozen"),
        _inner(f"({vp} params)"),
        _outer("  └────────┬────────────┘"),
        _outer("           ↓ (B, 729, 1152)"),
        _outer("  ┌─────────────────────┐"),
        _inner("Bridge Attention", "◄── Trainable"),
        _inner(f"({nq} queries, {nl} L)"),
        _outer("  └────────┬────────────┘"),
        _outer(f"           ↓ (B, {nq}, {dm})"),
        _outer("  ┌─────────────────────┐"),
        _inner(lm, f"◄── LoRA (r={rank})"),
        _inner(f"({lp} params)"),
        _outer("  └────────┬────────────┘"),
        _outer(f"           ↓ (B, {dm})"),
        _outer("  ┌─────────────────────┐"),
        _inner(card.action_head_type, "◄── Trainable"),
        _inner(f"({card.action_dim}-DOF, H={card.action_horizon})"),
        _outer("  └────────┬────────────┘"),
        _outer("           ↓"),
        _outer(f"  Actions (B, {card.action_dim})"),
        "└─────────────────────────────────────────────┘",
    ]
    return "\n".join(rows)


def generate_markdown(card: ModelProfileCard) -> str:
    """Generate a full HuggingFace README.md for *card*.

    Sections that depend on optional fields (``flops``, ``vram``,
    ``recommended_hp``) are omitted when those fields are ``None``.
    """
    vs = _short_name(card.vision_encoder)
    lm = _short_name(card.language_model)
    rank = card.recommended_hp.lora_rank if card.recommended_hp is not None else 32
    total_m, train_m, frozen_m = (
        _format_params(card.total_params),
        _format_params(card.trainable_params),
        _format_params(card.frozen_params),
    )
    parts: list[str] = []

    diagram = card.architecture_diagram or generate_ascii_diagram(card)
    parts += [
        "---\ntags:\n- robotics\n- vla\n- forge\n---",
        f"# {card.model_name}",
        (
            f"## Model Details\n"
            f"- **Architecture**: {vs} → BridgeAttention → {lm} (LoRA-{rank}) → {card.action_head_type} Head\n"
            f"- **Parameters**: {total_m} total ({train_m} trainable, {frozen_m} frozen)\n"
            f"- **Action Space**: {card.action_dim}-DOF, horizon={card.action_horizon}"
        ),
        f"## Architecture\n```\n{diagram}\n```",
    ]

    if card.components:
        rows = "\n".join(
            f"| {c.name} | {_format_params(c.param_count)} "
            f"| {_format_params(c.trainable_params)} | {_format_params(c.frozen_params)} |"
            for c in card.components
        )
        parts.append(
            "## Parameter Breakdown\n"
            "| Component | Total | Trainable | Frozen |\n"
            "|-----------|-------|-----------|--------|\n" + rows
        )

    perf: list[str] = (
        ([f"| FLOPs | {card.flops.total_gflops:.1f} GFLOPs |"] if card.flops else [])
        + ([f"| FP16 Size | {card.fp16_size_mb:.0f} MB |"] if card.fp16_size_mb else [])
        + ([f"| INT8 Size | {card.int8_size_mb:.0f} MB |"] if card.int8_size_mb else [])
        + ([f"| INT4 Size | {card.int4_size_mb:.0f} MB |"] if card.int4_size_mb else [])
        + (
            [
                f"| Training VRAM (FP16) | {card.vram.training_fp16_mb:.0f} MB |",
                f"| Inference VRAM (FP16) | {card.vram.inference_fp16_mb:.0f} MB |",
            ]
            if card.vram
            else []
        )
    )
    if perf:
        parts.append("## Performance Estimates\n| Metric | Value |\n|--------|-------|\n" + "\n".join(perf))

    if card.recommended_hp is not None:
        hp = {k: v for k, v in dataclasses.asdict(card.recommended_hp).items() if k != "rationale"}
        parts.append(
            "## Recommended Training Config\n```yaml\n" + "\n".join(f"{k}: {v}" for k, v in hp.items()) + "\n```"
        )

    if card.vram is not None and card.vram.fits_gpu:
        parts.append(
            "## GPU Compatibility\n| GPU | Fits? |\n|-----|-------|\n"
            + "\n".join(f"| {g} | {'Yes' if ok else 'No'} |" for g, ok in card.vram.fits_gpu.items())
        )

    return "\n\n".join(parts) + "\n"
