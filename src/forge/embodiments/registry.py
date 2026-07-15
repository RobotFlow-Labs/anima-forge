"""Embodiment registry with built-in profiles.

Provides pre-configured profiles for common robot platforms and
supports custom profile registration.

Usage:
    from forge.embodiments.registry import EmbodimentRegistry

    registry = EmbodimentRegistry()
    profile = registry.get("franka")
    yaml_config = registry.generate_yaml_config("franka")
"""

from __future__ import annotations

from forge.embodiments.profiles import EmbodimentProfile

# Built-in robot profiles
BUILTIN_PROFILES: dict[str, EmbodimentProfile] = {
    "franka": EmbodimentProfile(
        name="franka",
        description="Franka Emika Panda — 7-DoF collaborative robot",
        action_dim=7,
        dof=7,
        joint_names=["j1", "j2", "j3", "j4", "j5", "j6", "j7"],
        joint_min=[-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
        joint_max=[2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
        has_gripper=True,
        gripper_type="binary",
        control_frequency_hz=50.0,
        max_latency_ms=20.0,
        safety_torque_limit=0.8,
        recommended_variant="nano",
        recommended_horizon=8,
        recommended_chunk_overlap=2,
        recommended_action_head="flow",
    ),
    "aloha": EmbodimentProfile(
        name="aloha",
        description="ALOHA — Dual-arm 14-DoF bimanual manipulation",
        action_dim=14,
        dof=14,
        joint_names=[f"left_j{i}" for i in range(7)] + [f"right_j{i}" for i in range(7)],
        joint_min=[-3.14] * 14,
        joint_max=[3.14] * 14,
        has_gripper=True,
        gripper_type="binary",
        control_frequency_hz=50.0,
        max_latency_ms=20.0,
        safety_torque_limit=0.7,
        recommended_variant="small",
        recommended_horizon=16,
        recommended_chunk_overlap=4,
        recommended_action_head="chunk",
    ),
    "xarm": EmbodimentProfile(
        name="xarm",
        description="UFactory xArm — 6-DoF industrial robot",
        action_dim=6,
        dof=6,
        joint_names=["j1", "j2", "j3", "j4", "j5", "j6"],
        joint_min=[-6.28] * 6,
        joint_max=[6.28] * 6,
        has_gripper=True,
        gripper_type="continuous",
        control_frequency_hz=100.0,
        max_latency_ms=10.0,
        safety_torque_limit=1.0,
        recommended_variant="nano",
        recommended_horizon=4,
        recommended_chunk_overlap=1,
        recommended_action_head="flow",
    ),
    "ur5e": EmbodimentProfile(
        name="ur5e",
        description="Universal Robots UR5e — 6-DoF collaborative robot",
        action_dim=6,
        dof=6,
        joint_names=["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"],
        joint_min=[-6.28] * 6,
        joint_max=[6.28] * 6,
        has_gripper=False,
        gripper_type="none",
        control_frequency_hz=125.0,
        max_latency_ms=8.0,
        safety_torque_limit=0.9,
        recommended_variant="nano",
        recommended_horizon=4,
        recommended_chunk_overlap=1,
        recommended_action_head="flow",
    ),
}


class EmbodimentRegistry:
    """Registry for robot embodiment profiles.

    Pre-loaded with common platforms (franka, aloha, xarm, ur5e).
    Custom profiles can be registered at runtime.
    """

    def __init__(self):
        self._profiles: dict[str, EmbodimentProfile] = dict(BUILTIN_PROFILES)

    def get(self, name: str) -> EmbodimentProfile:
        """Get a profile by name.

        Raises:
            KeyError: if name not found
        """
        if name not in self._profiles:
            available = ", ".join(sorted(self._profiles.keys()))
            raise KeyError(f"Unknown embodiment '{name}'. Available: {available}")
        return self._profiles[name]

    def register(self, profile: EmbodimentProfile) -> None:
        """Register a custom embodiment profile."""
        self._profiles[profile.name] = profile

    def list_embodiments(self) -> list[str]:
        """List all registered embodiment names."""
        return sorted(self._profiles.keys())

    def generate_yaml_config(self, name: str) -> str:
        """Generate a complete FORGE YAML config for this embodiment.

        Args:
            name: Embodiment name

        Returns:
            YAML string with FORGE config overrides
        """
        import yaml

        profile = self.get(name)
        config = profile.to_forge_config()

        # Add metadata comment
        header = (
            f"# FORGE v3 config for {profile.description}\n"
            f"# Generated from embodiment profile: {profile.name}\n"
            f"# Control: {profile.control_frequency_hz}Hz, "
            f"max latency {profile.max_latency_ms}ms\n\n"
        )
        return header + yaml.dump(config, default_flow_style=False, sort_keys=False)
