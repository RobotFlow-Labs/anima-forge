"""PRD-08: Universal Teacher Registry tests."""

import numpy as np


def test_action_chunk_dataclass():
    """ActionChunk stores actions with correct shapes."""
    from forge.teachers.base import ActionChunk

    actions = np.random.randn(4, 7).astype(np.float32)
    chunk = ActionChunk(
        actions=actions,
        action_mean=actions,
        action_std=np.ones_like(actions) * 0.1,
        confidence=np.ones_like(actions) * 0.9,
    )
    assert chunk.actions.shape == (4, 7)
    assert chunk.vision_features is None
    assert chunk.language_features is None
    assert chunk.metadata == {}


def test_action_chunk_properties():
    """horizon and action_dim properties work."""
    from forge.teachers.base import ActionChunk

    actions = np.random.randn(8, 7).astype(np.float32)
    chunk = ActionChunk(
        actions=actions,
        action_mean=actions,
        action_std=np.ones_like(actions),
        confidence=np.ones_like(actions),
    )
    assert chunk.horizon == 8
    assert chunk.action_dim == 7


def test_teacher_info_dataclass():
    """TeacherInfo stores metadata correctly."""
    from forge.teachers.base import TeacherInfo

    info = TeacherInfo(
        name="test-teacher",
        architecture="token-ar",
        param_count=7.6,
        action_dim=7,
        action_horizon=1,
        vision_encoder="siglip",
        language_model="llama",
        supports_chunking=False,
        supports_features=True,
    )
    assert info.name == "test-teacher"
    assert info.param_count == 7.6
    assert info.supports_features is True


def test_registry_singleton():
    """TeacherRegistry is a singleton."""
    from forge.teachers.registry import TeacherRegistry

    r1 = TeacherRegistry()
    r2 = TeacherRegistry()
    assert r1 is r2
    # Reset for other tests
    r1.reset()


def test_registry_register_and_create():
    """Can register and create adapters."""
    from forge.teachers.base import TeacherAdapter
    from forge.teachers.registry import TeacherRegistry

    registry = TeacherRegistry()
    registry.reset()

    # Create a minimal concrete adapter for testing
    from forge.teachers.openvla_adapter import OpenVLAAdapter

    registry.register("test-openvla", OpenVLAAdapter)
    adapter = registry.create("test-openvla")
    assert isinstance(adapter, TeacherAdapter)
    assert not adapter.is_loaded

    registry.reset()


def test_registry_auto_discover():
    """Auto-discovers *_adapter.py modules."""
    from forge.teachers.registry import TeacherRegistry

    registry = TeacherRegistry()
    registry.reset()
    registry.auto_discover()

    teachers = registry.list_teachers()
    assert "openvla-7b" in teachers
    assert "rdt2-fm" in teachers
    assert "smolvla-base" in teachers
    assert len(teachers) >= 3

    registry.reset()


def test_registry_list_teachers():
    """Lists all registered teachers."""
    from forge.teachers.registry import get_registry

    registry = get_registry()
    registry.reset()
    registry.auto_discover()

    teachers = registry.list_teachers()
    assert isinstance(teachers, list)
    assert teachers == sorted(teachers)  # Should be sorted

    registry.reset()


def test_openvla_adapter_info():
    """OpenVLA adapter returns correct info."""
    from forge.teachers.openvla_adapter import OpenVLAAdapter

    adapter = OpenVLAAdapter()
    info = adapter.info()
    assert info.name == "openvla-7b"
    assert info.architecture == "token-ar"
    assert info.param_count == 7.6
    assert info.action_dim == 7
    assert info.action_horizon == 1
    assert info.supports_chunking is False
    assert info.supports_features is True
    assert not adapter.is_loaded

    # Test action space
    space = adapter.get_action_space()
    assert space["dim"] == 7
    assert len(space["names"]) == 7
