"""FORGE v2 Embodiment Profiles & Auto-Configuration.

Hardware-specific configurations for robot embodiments.
"""

from forge.embodiments.profiles import EmbodimentProfile
from forge.embodiments.registry import BUILTIN_PROFILES, EmbodimentRegistry

__all__ = [
    "EmbodimentProfile",
    "EmbodimentRegistry",
    "BUILTIN_PROFILES",
]
