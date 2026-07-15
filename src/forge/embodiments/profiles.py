"""Robot embodiment profiles — hardware-specific configurations.

Each profile specifies:
- Action space: dimension, joint limits, gripper type
- Control: frequency, latency budget, safety constraints
- Recommended FORGE config: chunk horizon, student variant, encoder

Usage:
    from forge.embodiments.profiles import EmbodimentProfile

    profile = EmbodimentProfile(
        name="franka",
        description="Franka Emika Panda",
        action_dim=7,
        dof=7,
        joint_names=["j1", ..., "j7"],
        joint_min=[-2.89, ...],
        joint_max=[2.89, ...],
    )
    config_overrides = profile.to_forge_config()
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EmbodimentProfile:
    """Complete robot embodiment specification.

    Describes the robot's action space, control parameters, and
    recommended FORGE configuration for optimal performance.

    Attributes:
        name: Short identifier (e.g., "franka", "aloha")
        description: Human-readable description
        action_dim: Total action dimension (joints + gripper)
        dof: Degrees of freedom (joints only)
        joint_names: Name of each joint
        joint_min: Lower joint limits (radians)
        joint_max: Upper joint limits (radians)
        has_gripper: Whether the robot has a gripper
        gripper_type: "binary", "continuous", or "none"
        control_frequency_hz: Target control frequency
        max_latency_ms: Maximum acceptable inference latency
        safety_torque_limit: Normalized torque limit (0-1)
        recommended_variant: Student model variant ("nano", "small", "micro")
        recommended_horizon: Optimal action chunk horizon
        recommended_chunk_overlap: Overlap for chunk blending
        recommended_action_head: Best action head type
    """

    name: str
    description: str

    # Action space
    action_dim: int
    dof: int
    joint_names: list[str] = field(default_factory=list)
    joint_min: list[float] = field(default_factory=list)
    joint_max: list[float] = field(default_factory=list)
    has_gripper: bool = True
    gripper_type: str = "binary"  # "binary" | "continuous" | "none"

    # Control
    control_frequency_hz: float = 50.0
    max_latency_ms: float = 20.0
    safety_torque_limit: float = 1.0

    # Recommended FORGE config
    recommended_variant: str = "nano"
    recommended_horizon: int = 8
    recommended_chunk_overlap: int = 2
    recommended_action_head: str = "flow"

    def to_forge_config(self) -> dict:
        """Generate FORGE config overrides for this embodiment.

        Returns:
            Dict suitable for merging into ForgeConfig via _apply_overrides
        """
        return {
            "student": {
                "variant": self.recommended_variant,
                "action_dim": self.action_dim,
                "action_horizon": self.recommended_horizon,
                "chunk_overlap": self.recommended_chunk_overlap,
                "action_head_type": self.recommended_action_head,
            },
        }

    def validate(self) -> list[str]:
        """Validate profile consistency.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        if len(self.joint_names) != self.dof:
            errors.append(f"joint_names count ({len(self.joint_names)}) != dof ({self.dof})")
        if len(self.joint_min) != self.dof:
            errors.append(f"joint_min count ({len(self.joint_min)}) != dof ({self.dof})")
        if len(self.joint_max) != self.dof:
            errors.append(f"joint_max count ({len(self.joint_max)}) != dof ({self.dof})")
        for i, (lo, hi) in enumerate(zip(self.joint_min, self.joint_max)):
            if lo >= hi:
                errors.append(f"joint {i}: min ({lo}) >= max ({hi})")
        if self.action_dim < self.dof:
            errors.append(f"action_dim ({self.action_dim}) < dof ({self.dof})")
        if self.control_frequency_hz <= 0:
            errors.append("control_frequency_hz must be positive")
        return errors
