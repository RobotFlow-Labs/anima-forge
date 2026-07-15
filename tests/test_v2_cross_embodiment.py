"""Tests for PRD-30: Cross-Embodiment Transfer Learning."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from forge.cross_embodiment import (
    ActionSpaceMapper,
    EmbodimentProfile,
    EmbodimentTransfer,
    JointNameMapper,
    LearnedActionAdapter,
    TransferConfig,
)

# ── Test Profiles ────────────────────────────────────────

FRANKA = EmbodimentProfile(
    name="franka",
    action_dim=7,
    joint_names=["j1", "j2", "j3", "j4", "j5", "j6", "j7"],
    joint_min=[-2.9, -1.8, -2.9, -3.1, -2.9, -0.02, -2.9],
    joint_max=[2.9, 1.8, 2.9, -0.07, 2.9, 3.75, 2.9],
    has_gripper=True,
)

UR5E = EmbodimentProfile(
    name="ur5e",
    action_dim=6,
    joint_names=["shoulder", "upper_arm", "forearm", "wrist1", "wrist2", "wrist3"],
    joint_min=[-6.28] * 6,
    joint_max=[6.28] * 6,
)

ALOHA = EmbodimentProfile(
    name="aloha",
    action_dim=14,
    joint_names=[f"left_j{i}" for i in range(7)] + [f"right_j{i}" for i in range(7)],
    joint_min=[-3.14] * 14,
    joint_max=[3.14] * 14,
    has_gripper=True,
)


# ── ActionSpaceMapper ────────────────────────────────────


class TestActionSpaceMapper:
    def test_same_dim(self):
        m = ActionSpaceMapper(7, 7)
        actions = np.ones((4, 7))
        result = m.map(actions)
        assert result.shape == (4, 7)

    def test_pad_zero(self):
        m = ActionSpaceMapper(6, 7, TransferConfig(pad_strategy="zero", scale_actions=False))
        actions = np.ones((2, 6))
        result = m.map(actions)
        assert result.shape == (2, 7)
        assert result[0, 6] == 0.0  # Padded with zero

    def test_pad_mirror(self):
        m = ActionSpaceMapper(6, 8, TransferConfig(pad_strategy="mirror", scale_actions=False))
        actions = np.arange(6, dtype=np.float32).reshape(1, 6)
        result = m.map(actions)
        assert result.shape == (1, 8)
        assert result[0, 6] == 4.0  # Mirror last 2
        assert result[0, 7] == 5.0

    def test_pad_repeat(self):
        m = ActionSpaceMapper(6, 8, TransferConfig(pad_strategy="repeat", scale_actions=False))
        actions = np.arange(6, dtype=np.float32).reshape(1, 6)
        result = m.map(actions)
        assert result[0, 6] == 5.0
        assert result[0, 7] == 5.0

    def test_trim_first(self):
        m = ActionSpaceMapper(7, 6, TransferConfig(trim_strategy="first", scale_actions=False))
        actions = np.arange(7, dtype=np.float32).reshape(1, 7)
        result = m.map(actions)
        assert result.shape == (1, 6)
        np.testing.assert_array_equal(result[0], [0, 1, 2, 3, 4, 5])

    def test_trim_last(self):
        m = ActionSpaceMapper(7, 6, TransferConfig(trim_strategy="last", scale_actions=False))
        actions = np.arange(7, dtype=np.float32).reshape(1, 7)
        result = m.map(actions)
        np.testing.assert_array_equal(result[0], [1, 2, 3, 4, 5, 6])

    def test_trim_even(self):
        m = ActionSpaceMapper(7, 3, TransferConfig(trim_strategy="even", scale_actions=False))
        actions = np.arange(7, dtype=np.float32).reshape(1, 7)
        result = m.map(actions)
        assert result.shape == (1, 3)

    def test_joint_limit_scaling(self):
        m = ActionSpaceMapper(2, 2)
        m.set_joint_limits(
            source_min=[0.0, 0.0],
            source_max=[1.0, 1.0],
            target_min=[-1.0, -1.0],
            target_max=[1.0, 1.0],
        )
        actions = np.array([[0.5, 0.5]])
        result = m.map(actions)
        np.testing.assert_allclose(result[0], [0.0, 0.0], atol=0.01)

    def test_batch_shape_preserved(self):
        m = ActionSpaceMapper(7, 6, TransferConfig(scale_actions=False))
        actions = np.ones((3, 5, 7))
        result = m.map(actions)
        assert result.shape == (3, 5, 6)

    def test_1d_input(self):
        m = ActionSpaceMapper(3, 5, TransferConfig(scale_actions=False))
        actions = np.ones(3)
        result = m.map(actions)
        assert result.shape == (5,)


# ── JointNameMapper ──────────────────────────────────────


class TestJointNameMapper:
    def test_exact_match(self):
        mapper = JointNameMapper(
            source_joints=["j1", "j2", "j3"],
            target_joints=["j1", "j2", "j3"],
        )
        assert len(mapper.mapping) == 3

    def test_no_match(self):
        mapper = JointNameMapper(
            source_joints=["alpha", "beta"],
            target_joints=["gamma", "delta"],
        )
        assert len(mapper.mapping) == 0
        with pytest.raises(ValueError, match="unmatched target joints"):
            mapper.map(np.ones((1, 2), dtype=np.float32))

    def test_partial_match(self):
        mapper = JointNameMapper(
            source_joints=["shoulder", "elbow", "wrist"],
            target_joints=["shoulder", "wrist", "finger"],
        )
        # shoulder and wrist should match
        assert 0 in mapper.mapping  # shoulder → shoulder
        assert 1 in mapper.mapping  # wrist → wrist
        assert mapper.unmatched_target_joints == ["finger"]
        with pytest.raises(ValueError, match="finger"):
            mapper.map(np.ones((1, 3), dtype=np.float32))

    def test_map_1d(self):
        mapper = JointNameMapper(
            source_joints=["j1", "j2", "j3"],
            target_joints=["j1", "j3"],
        )
        source = np.array([1.0, 2.0, 3.0])
        result = mapper.map(source)
        assert result.shape == (2,)
        assert result[0] == 1.0  # j1
        assert result[1] == 3.0  # j3

    def test_map_batch(self):
        mapper = JointNameMapper(
            source_joints=["j1", "j2"],
            target_joints=["j1", "j2"],
        )
        source = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = mapper.map(source)
        assert result.shape == (2, 2)

    def test_similarity_identical(self):
        assert JointNameMapper._similarity("hello", "hello") == 1.0

    def test_similarity_case_insensitive(self):
        assert JointNameMapper._similarity("HELLO", "hello") == 0.95


# ── LearnedActionAdapter ─────────────────────────────────


class TestLearnedAdapter:
    def test_forward(self):
        adapter = LearnedActionAdapter(7, 6)
        x = torch.randn(4, 7)
        y = adapter(x)
        assert y.shape == (4, 6)

    def test_gradients(self):
        adapter = LearnedActionAdapter(7, 6)
        x = torch.randn(4, 7)
        y = adapter(x)
        loss = y.pow(2).mean()
        loss.backward()
        has_grad = any(p.grad is not None for p in adapter.parameters())
        assert has_grad

    def test_param_count(self):
        adapter = LearnedActionAdapter(7, 6, hidden_dim=32)
        n = sum(p.numel() for p in adapter.parameters())
        assert n > 0

    def test_custom_hidden(self):
        adapter = LearnedActionAdapter(7, 6, hidden_dim=128)
        x = torch.randn(1, 7)
        y = adapter(x)
        assert y.shape == (1, 6)


# ── EmbodimentTransfer ───────────────────────────────────


class TestEmbodimentTransfer:
    def test_linear_franka_to_ur5e(self):
        transfer = EmbodimentTransfer(FRANKA, UR5E)
        actions = np.random.randn(5, 7).astype(np.float32)
        result = transfer.map_actions(actions)
        assert result.shape == (5, 6)

    def test_linear_ur5e_to_franka(self):
        transfer = EmbodimentTransfer(UR5E, FRANKA)
        actions = np.random.randn(5, 6).astype(np.float32)
        result = transfer.map_actions(actions)
        assert result.shape == (5, 7)

    def test_franka_to_aloha(self):
        transfer = EmbodimentTransfer(FRANKA, ALOHA)
        actions = np.random.randn(3, 7).astype(np.float32)
        result = transfer.map_actions(actions)
        assert result.shape == (3, 14)

    def test_joint_name_strategy(self):
        transfer = EmbodimentTransfer(
            FRANKA,
            UR5E,
            TransferConfig(mapping_strategy="joint_name"),
        )
        assert transfer.joint_mapper is not None
        actions = np.random.randn(2, 7).astype(np.float32)
        with pytest.raises(ValueError, match="unmatched target joints"):
            transfer.map_actions(actions)

    def test_joint_name_strategy_with_correspondence(self):
        xarm = EmbodimentProfile(
            name="xarm",
            action_dim=6,
            joint_names=["j1", "j2", "j3", "j4", "j5", "j6"],
        )
        transfer = EmbodimentTransfer(
            FRANKA,
            xarm,
            TransferConfig(mapping_strategy="joint_name"),
        )
        actions = np.arange(14, dtype=np.float32).reshape(2, 7)
        result = transfer.map_actions(actions)
        assert result.shape == (2, 6)
        np.testing.assert_array_equal(result, actions[:, :6])

    def test_learned_strategy(self):
        transfer = EmbodimentTransfer(
            FRANKA,
            UR5E,
            TransferConfig(mapping_strategy="learned"),
        )
        assert transfer.learned_adapter is not None
        actions = np.random.randn(2, 7).astype(np.float32)
        with pytest.raises(RuntimeError, match="untrained"):
            transfer.map_actions(actions)
        target = actions[:, :6].copy()
        metrics = transfer.fit_learned_adapter(actions, target, steps=20)
        result = transfer.map_actions(actions)
        assert result.shape == (2, 6)
        assert metrics["loss_after"] < metrics["loss_before"]

    def test_map_actions_torch(self):
        transfer = EmbodimentTransfer(
            FRANKA,
            UR5E,
            TransferConfig(mapping_strategy="learned"),
        )
        actions = torch.randn(4, 7)
        result = transfer.map_actions_torch(actions)
        assert result.shape == (4, 6)

    def test_info(self):
        transfer = EmbodimentTransfer(FRANKA, UR5E)
        info = transfer.info()
        assert info["source"] == "franka"
        assert info["target"] == "ur5e"
        assert info["dim_change"] == -1

    def test_info_learned(self):
        transfer = EmbodimentTransfer(
            FRANKA,
            UR5E,
            TransferConfig(mapping_strategy="learned"),
        )
        info = transfer.info()
        assert "adapter_params" in info
        assert info["adapter_params"] > 0
        assert info["learned_fitted"] is False

    def test_same_embodiment(self):
        transfer = EmbodimentTransfer(FRANKA, FRANKA)
        actions = np.random.randn(3, 7).astype(np.float32)
        result = transfer.map_actions(actions)
        assert result.shape == (3, 7)


# ── Strict Edge Cases ────────────────────────────────────


class TestTransferStrict:
    def test_1d_action(self):
        transfer = EmbodimentTransfer(
            FRANKA,
            UR5E,
            TransferConfig(scale_actions=False),
        )
        action = np.ones(7, dtype=np.float32)
        result = transfer.map_actions(action)
        assert result.shape == (6,)

    def test_aloha_to_franka_trim(self):
        transfer = EmbodimentTransfer(
            ALOHA,
            FRANKA,
            TransferConfig(scale_actions=False),
        )
        actions = np.arange(14, dtype=np.float32).reshape(1, 14)
        result = transfer.map_actions(actions)
        assert result.shape == (1, 7)

    def test_zero_dim_profile(self):
        p1 = EmbodimentProfile(name="a", action_dim=3)
        p2 = EmbodimentProfile(name="b", action_dim=3)
        transfer = EmbodimentTransfer(p1, p2, TransferConfig(scale_actions=False))
        actions = np.zeros((1, 3), dtype=np.float32)
        result = transfer.map_actions(actions)
        assert result.shape == (1, 3)

    def test_large_batch(self):
        transfer = EmbodimentTransfer(
            FRANKA,
            UR5E,
            TransferConfig(scale_actions=False),
        )
        actions = np.random.randn(1000, 7).astype(np.float32)
        result = transfer.map_actions(actions)
        assert result.shape == (1000, 6)

    def test_learned_torch_gradients(self):
        transfer = EmbodimentTransfer(
            FRANKA,
            UR5E,
            TransferConfig(mapping_strategy="learned"),
        )
        actions = torch.randn(4, 7, requires_grad=True)
        result = transfer.map_actions_torch(actions)
        loss = result.pow(2).mean()
        loss.backward()
        assert transfer.learned_adapter is not None
        has_grad = any(p.grad is not None for p in transfer.learned_adapter.parameters())
        assert has_grad
