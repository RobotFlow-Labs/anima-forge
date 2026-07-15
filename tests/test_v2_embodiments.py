"""PRD-16: Embodiment Profiles & Auto-Configuration tests."""

import pytest

from forge.embodiments.profiles import EmbodimentProfile
from forge.embodiments.registry import BUILTIN_PROFILES, EmbodimentRegistry


def test_embodiment_profile_dataclass():
    """EmbodimentProfile stores all required fields."""
    profile = EmbodimentProfile(
        name="test_bot",
        description="Test robot",
        action_dim=5,
        dof=5,
        joint_names=["j1", "j2", "j3", "j4", "j5"],
        joint_min=[-1.0] * 5,
        joint_max=[1.0] * 5,
    )
    assert profile.name == "test_bot"
    assert profile.action_dim == 5
    assert profile.dof == 5
    assert len(profile.joint_names) == 5
    assert profile.has_gripper is True  # default
    assert profile.control_frequency_hz == 50.0  # default


def test_franka_profile():
    """Franka profile has correct specs."""
    profile = BUILTIN_PROFILES["franka"]
    assert profile.action_dim == 7
    assert profile.dof == 7
    assert len(profile.joint_names) == 7
    assert len(profile.joint_min) == 7
    assert len(profile.joint_max) == 7
    assert profile.has_gripper is True
    assert profile.recommended_action_head == "flow"
    assert profile.validate() == []


def test_aloha_profile():
    """ALOHA profile has correct specs for bimanual."""
    profile = BUILTIN_PROFILES["aloha"]
    assert profile.action_dim == 14
    assert profile.dof == 14
    assert len(profile.joint_names) == 14
    assert profile.recommended_variant == "small"  # Larger for bimanual
    assert profile.recommended_horizon == 16
    assert profile.recommended_action_head == "chunk"
    assert profile.validate() == []


def test_xarm_profile():
    """xArm profile has correct specs."""
    profile = BUILTIN_PROFILES["xarm"]
    assert profile.action_dim == 6
    assert profile.dof == 6
    assert profile.gripper_type == "continuous"
    assert profile.control_frequency_hz == 100.0
    assert profile.recommended_horizon == 4
    assert profile.validate() == []


def test_embodiment_registry_list():
    """Registry lists all built-in embodiments."""
    registry = EmbodimentRegistry()
    names = registry.list_embodiments()
    assert "franka" in names
    assert "aloha" in names
    assert "xarm" in names
    assert "ur5e" in names
    assert len(names) >= 4


def test_embodiment_registry_get():
    """Registry retrieves profiles by name and raises on unknown."""
    registry = EmbodimentRegistry()
    profile = registry.get("franka")
    assert profile.name == "franka"

    with pytest.raises(KeyError, match="Unknown embodiment"):
        registry.get("nonexistent_robot")


def test_embodiment_to_forge_config():
    """to_forge_config produces correct config overrides."""
    profile = BUILTIN_PROFILES["franka"]
    config = profile.to_forge_config()

    assert "student" in config
    assert config["student"]["action_dim"] == 7
    assert config["student"]["action_horizon"] == 8
    assert config["student"]["action_head_type"] == "flow"
    assert config["student"]["variant"] == "nano"


def test_embodiment_generate_yaml():
    """generate_yaml_config produces valid YAML string."""
    import yaml

    registry = EmbodimentRegistry()
    yaml_str = registry.generate_yaml_config("franka")

    assert isinstance(yaml_str, str)
    assert "franka" in yaml_str.lower()

    # Extract just the YAML part (skip comments)
    yaml_lines = [line for line in yaml_str.split("\n") if not line.startswith("#")]
    yaml_body = "\n".join(yaml_lines)
    parsed = yaml.safe_load(yaml_body)

    assert parsed is not None
    assert "student" in parsed
    assert parsed["student"]["action_dim"] == 7
