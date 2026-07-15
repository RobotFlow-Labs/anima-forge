"""FORGE Model Profiler — deep introspection & model card generation.

Usage:
    from forge.profiler import FORGEProfiler

    profiler = FORGEProfiler(variant="nano")
    card = profiler.generate_card()
    card.save_json("profiles/nano.json")
    md = profiler.generate_markdown(card)
"""

from forge.profiler.dataclasses import (
    ComponentProfile,
    FLOPsEstimate,
    ModelProfileCard,
    RecommendedHyperparams,
    VRAMEstimate,
)
from forge.profiler.profiler import FORGEProfiler

__all__ = [
    "ComponentProfile",
    "FLOPsEstimate",
    "FORGEProfiler",
    "ModelProfileCard",
    "RecommendedHyperparams",
    "VRAMEstimate",
]
