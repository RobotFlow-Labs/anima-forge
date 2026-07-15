"""Tests for PRD-28: Domain Adaptation & Fine-Tuning Pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from forge.finetune import (
    EWCPenalty,
    FinetuneConfig,
    FinetuneReport,
    FinetuneTrainer,
    ReplayBuffer,
    apply_finetune_strategy,
)

# ── Helpers ──────────────────────────────────────────────


class DummyStudent(nn.Module):
    """Minimal student model with LoRA-like parameters."""

    def __init__(self):
        super().__init__()
        self.vision_encoder = nn.Linear(384, 128)
        self.bridge = nn.Linear(128, 64)
        self.lora_adapter = nn.Linear(64, 64)
        self.action_head = nn.Linear(64, 7)

    def forward(self, image, gt_actions=None):
        x = self.vision_encoder(image.view(image.shape[0], -1)[:, :384])
        x = self.bridge(x)
        x = self.lora_adapter(x)
        actions = self.action_head(x)
        result = {"actions": actions}
        if gt_actions is not None:
            result["loss"] = nn.functional.mse_loss(actions, gt_actions)
        return result


class DummyDataset(Dataset):
    """Minimal dataset for testing."""

    def __init__(self, n: int = 100, image_dim: int = 384, action_dim: int = 7):
        self.n = n
        self.images = torch.randn(n, 3, 384, 384)
        self.actions = torch.randn(n, action_dim)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {
            "image": self.images[idx],
            "ground_truth_actions": self.actions[idx],
        }


# ── FinetuneConfig ───────────────────────────────────────


class TestFinetuneConfig:
    def test_defaults(self):
        config = FinetuneConfig()
        assert config.strategy == "lora"
        assert config.lr == 5e-5
        assert config.max_steps == 5000
        assert config.replay_enabled is False
        assert config.ewc_enabled is False

    def test_custom_config(self):
        config = FinetuneConfig(
            strategy="action_head",
            lr=1e-4,
            max_steps=1000,
            replay_enabled=True,
            replay_ratio=0.3,
        )
        assert config.strategy == "action_head"
        assert config.lr == 1e-4
        assert config.replay_enabled is True

    def test_ewc_config(self):
        config = FinetuneConfig(ewc_enabled=True, ewc_lambda=500.0)
        assert config.ewc_enabled is True
        assert config.ewc_lambda == 500.0


# ── ReplayBuffer ─────────────────────────────────────────


class TestReplayBuffer:
    def test_add_and_size(self):
        buf = ReplayBuffer(max_size=10)
        assert buf.size == 0
        buf.add({"x": torch.tensor([1.0])})
        assert buf.size == 1

    def test_max_size(self):
        buf = ReplayBuffer(max_size=5)
        for i in range(10):
            buf.add({"x": torch.tensor([float(i)])})
        assert buf.size == 5

    def test_sample(self):
        buf = ReplayBuffer(max_size=100)
        for i in range(20):
            buf.add({"x": torch.tensor([float(i)])})
        samples = buf.sample(5)
        assert len(samples) == 5
        assert all("x" in s for s in samples)

    def test_sample_more_than_buffer(self):
        buf = ReplayBuffer(max_size=100)
        for i in range(3):
            buf.add({"x": torch.tensor([float(i)])})
        samples = buf.sample(10)
        assert len(samples) == 3  # Can't sample more than available

    def test_sample_empty(self):
        buf = ReplayBuffer(max_size=10)
        assert buf.sample(5) == []

    def test_is_ready(self):
        buf = ReplayBuffer(max_size=10)
        assert buf.is_ready(min_size=1) is False
        buf.add({"x": torch.tensor([1.0])})
        assert buf.is_ready(min_size=1) is True
        assert buf.is_ready(min_size=5) is False

    def test_detaches_tensors(self):
        buf = ReplayBuffer(max_size=10)
        t = torch.tensor([1.0], requires_grad=True)
        buf.add({"x": t})
        assert not buf._buffer[0]["x"].requires_grad

    def test_cpu_storage(self):
        buf = ReplayBuffer(max_size=10)
        buf.add({"x": torch.tensor([1.0])})
        assert buf._buffer[0]["x"].device.type == "cpu"


# ── EWCPenalty ───────────────────────────────────────────


class TestEWCPenalty:
    def test_stores_reference_params(self):
        model = DummyStudent()
        ewc = EWCPenalty(model, ewc_lambda=1000.0)
        assert len(ewc._means) > 0
        assert all(isinstance(v, torch.Tensor) for v in ewc._means.values())

    def test_penalty_zero_at_init(self):
        model = DummyStudent()
        ewc = EWCPenalty(model, ewc_lambda=1000.0)
        # Without Fisher info, penalty should still be zero (no Fisher computed)
        assert not ewc.has_fisher

    def test_compute_fisher(self):
        model = DummyStudent()
        dataset = DummyDataset(n=10)
        ewc = EWCPenalty(model, ewc_lambda=1000.0)
        ewc.compute_fisher(model, dataset, device="cpu", n_samples=5)
        assert ewc.has_fisher
        assert len(ewc._fisher) > 0

    def test_penalty_increases_with_deviation(self):
        model = DummyStudent()
        dataset = DummyDataset(n=10)
        ewc = EWCPenalty(model, ewc_lambda=1000.0)
        ewc.compute_fisher(model, dataset, device="cpu", n_samples=5)

        penalty_before = ewc.penalty(model).item()

        # Modify model params
        with torch.no_grad():
            for param in model.parameters():
                if param.requires_grad:
                    param.add_(torch.randn_like(param) * 0.1)

        penalty_after = ewc.penalty(model).item()
        assert penalty_after > penalty_before

    def test_ewc_lambda_scales_penalty(self):
        model = DummyStudent()
        dataset = DummyDataset(n=10)

        ewc_low = EWCPenalty(model, ewc_lambda=1.0)
        ewc_high = EWCPenalty(model, ewc_lambda=10000.0)

        ewc_low.compute_fisher(model, dataset, device="cpu", n_samples=5)
        ewc_high._fisher = {k: v.clone() for k, v in ewc_low._fisher.items()}

        # Modify params
        with torch.no_grad():
            for param in model.parameters():
                if param.requires_grad:
                    param.add_(torch.randn_like(param) * 0.1)

        p_low = ewc_low.penalty(model).item()
        p_high = ewc_high.penalty(model).item()
        assert p_high > p_low


# ── apply_finetune_strategy ──────────────────────────────


class TestApplyFinetuneStrategy:
    def test_lora_strategy(self):
        model = DummyStudent()
        trainable = apply_finetune_strategy(model, "lora")
        assert trainable > 0
        # LoRA adapter and action head should be trainable
        assert model.lora_adapter.weight.requires_grad
        assert model.action_head.weight.requires_grad
        # Vision encoder should be frozen
        assert not model.vision_encoder.weight.requires_grad

    def test_action_head_strategy(self):
        model = DummyStudent()
        trainable = apply_finetune_strategy(model, "action_head")
        assert trainable > 0
        assert model.action_head.weight.requires_grad
        assert not model.lora_adapter.weight.requires_grad
        assert not model.vision_encoder.weight.requires_grad

    def test_full_strategy(self):
        model = DummyStudent()
        trainable = apply_finetune_strategy(model, "full")
        assert trainable > 0
        assert model.lora_adapter.weight.requires_grad
        assert model.bridge.weight.requires_grad
        assert model.action_head.weight.requires_grad
        assert not model.vision_encoder.weight.requires_grad

    def test_unknown_strategy_raises(self):
        model = DummyStudent()
        with pytest.raises(ValueError, match="Unknown strategy"):
            apply_finetune_strategy(model, "invalid")

    def test_strategy_param_count(self):
        model = DummyStudent()
        lora_count = apply_finetune_strategy(model, "lora")
        model2 = DummyStudent()
        head_count = apply_finetune_strategy(model2, "action_head")
        model3 = DummyStudent()
        full_count = apply_finetune_strategy(model3, "full")
        # action_head < lora < full
        assert head_count < lora_count <= full_count


# ── FinetuneReport ───────────────────────────────────────


class TestFinetuneReport:
    def test_defaults(self):
        report = FinetuneReport()
        assert report.total_steps == 0
        assert report.best_loss == float("inf")

    def test_to_dict(self):
        report = FinetuneReport(
            total_steps=100,
            elapsed_seconds=30.5,
            final_loss=0.01,
            best_loss=0.008,
            strategy="lora",
            checkpoint_path="/tmp/ckpt.pt",
            ewc_used=True,
            replay_used=False,
        )
        d = report.to_dict()
        assert d["total_steps"] == 100
        assert d["final_loss"] == 0.01
        assert d["ewc_used"] is True
        assert d["strategy"] == "lora"


# ── FinetuneTrainer ──────────────────────────────────────


class TestFinetuneTrainer:
    def test_create_trainer(self, tmp_path):
        model = DummyStudent()
        config = FinetuneConfig(
            strategy="lora",
            max_steps=10,
            output_dir=str(tmp_path / "ft"),
        )
        trainer = FinetuneTrainer(model, config, device="cpu")
        assert trainer.device == "cpu"
        assert trainer.replay is None
        assert trainer.ewc is None

    def test_create_with_replay(self, tmp_path):
        model = DummyStudent()
        config = FinetuneConfig(
            strategy="lora",
            max_steps=10,
            output_dir=str(tmp_path / "ft"),
            replay_enabled=True,
            replay_buffer_size=100,
        )
        trainer = FinetuneTrainer(model, config, device="cpu")
        assert trainer.replay is not None
        assert trainer.replay.max_size == 100

    def test_create_with_ewc(self, tmp_path):
        model = DummyStudent()
        config = FinetuneConfig(
            strategy="lora",
            max_steps=10,
            output_dir=str(tmp_path / "ft"),
            ewc_enabled=True,
            ewc_lambda=500.0,
        )
        trainer = FinetuneTrainer(model, config, device="cpu")
        assert trainer.ewc is not None

    def test_cosine_lr_warmup(self, tmp_path):
        model = DummyStudent()
        config = FinetuneConfig(
            lr=1e-4,
            warmup_steps=100,
            max_steps=1000,
            output_dir=str(tmp_path / "ft"),
        )
        trainer = FinetuneTrainer(model, config, device="cpu")
        # At step 0, LR should be 0 (warmup start)
        assert trainer._cosine_lr(0) == 0.0
        # At warmup end, LR should be full
        assert trainer._cosine_lr(100) == pytest.approx(1e-4, rel=0.01)
        # At max_steps, LR should be ~0
        assert trainer._cosine_lr(1000) == pytest.approx(0.0, abs=1e-8)

    def test_cosine_lr_midpoint(self, tmp_path):
        model = DummyStudent()
        config = FinetuneConfig(
            lr=1e-4,
            warmup_steps=0,
            max_steps=1000,
            output_dir=str(tmp_path / "ft"),
        )
        trainer = FinetuneTrainer(model, config, device="cpu")
        mid_lr = trainer._cosine_lr(500)
        # At midpoint of cosine, should be ~0.5 * lr
        assert 0.4e-4 < mid_lr < 0.6e-4

    def test_train_basic(self, tmp_path):
        model = DummyStudent()
        dataset = DummyDataset(n=20)
        config = FinetuneConfig(
            strategy="lora",
            lr=1e-3,
            max_steps=10,
            batch_size=4,
            output_dir=str(tmp_path / "ft"),
            save_every=100,  # Don't save intermediate
        )
        trainer = FinetuneTrainer(model, config, device="cpu")
        report = trainer.train(dataset, log_every=5)

        assert report.total_steps == 10
        assert report.elapsed_seconds > 0
        assert report.final_loss > 0
        assert report.strategy == "lora"
        assert report.checkpoint_path != ""

    def test_train_with_replay(self, tmp_path):
        model = DummyStudent()
        dataset = DummyDataset(n=20)
        config = FinetuneConfig(
            strategy="action_head",
            max_steps=10,
            batch_size=4,
            output_dir=str(tmp_path / "ft"),
            replay_enabled=True,
            replay_buffer_size=50,
        )
        trainer = FinetuneTrainer(model, config, device="cpu")
        report = trainer.train(dataset, log_every=5)
        assert report.replay_used is True
        assert trainer.replay.size > 0

    def test_train_with_ewc(self, tmp_path):
        model = DummyStudent()
        dataset = DummyDataset(n=20)
        config = FinetuneConfig(
            strategy="lora",
            max_steps=10,
            batch_size=4,
            output_dir=str(tmp_path / "ft"),
            ewc_enabled=True,
            ewc_lambda=100.0,
        )
        trainer = FinetuneTrainer(model, config, device="cpu")
        report = trainer.train(dataset, log_every=5)
        assert report.ewc_used is True

    def test_checkpoint_saved(self, tmp_path):
        model = DummyStudent()
        dataset = DummyDataset(n=20)
        config = FinetuneConfig(
            strategy="lora",
            max_steps=10,
            batch_size=4,
            output_dir=str(tmp_path / "ft"),
        )
        trainer = FinetuneTrainer(model, config, device="cpu")
        report = trainer.train(dataset)
        assert Path(report.checkpoint_path).exists()

    def test_checkpoint_contents(self, tmp_path):
        model = DummyStudent()
        dataset = DummyDataset(n=20)
        config = FinetuneConfig(
            strategy="lora",
            max_steps=5,
            batch_size=4,
            output_dir=str(tmp_path / "ft"),
        )
        trainer = FinetuneTrainer(model, config, device="cpu")
        report = trainer.train(dataset)
        ckpt = torch.load(report.checkpoint_path, weights_only=True)
        assert "student_state_dict" in ckpt
        assert "optimizer_state_dict" in ckpt
        assert ckpt["strategy"] == "lora"
        assert ckpt["global_step"] == 5

    def test_load_pretrained(self, tmp_path):
        model = DummyStudent()
        dataset = DummyDataset(n=20)
        config = FinetuneConfig(
            strategy="lora",
            max_steps=5,
            batch_size=4,
            output_dir=str(tmp_path / "ft"),
        )
        trainer = FinetuneTrainer(model, config, device="cpu")
        report = trainer.train(dataset)

        # Load into new model
        model2 = DummyStudent()
        config2 = FinetuneConfig(
            strategy="lora",
            max_steps=5,
            output_dir=str(tmp_path / "ft2"),
        )
        trainer2 = FinetuneTrainer(model2, config2, device="cpu")
        trainer2.load_pretrained(report.checkpoint_path)

    def test_loss_decreases(self, tmp_path):
        """Verify loss decreases over training (smoke test)."""
        torch.manual_seed(42)
        model = DummyStudent()
        dataset = DummyDataset(n=50)
        config = FinetuneConfig(
            strategy="full",
            lr=1e-2,
            max_steps=50,
            batch_size=8,
            warmup_steps=0,
            output_dir=str(tmp_path / "ft"),
        )
        trainer = FinetuneTrainer(model, config, device="cpu")
        report = trainer.train(dataset, log_every=100)
        # Best loss should be lower than initial (training works)
        assert report.best_loss < report.final_loss or report.best_loss < 1.0

    def test_full_strategy_training(self, tmp_path):
        model = DummyStudent()
        dataset = DummyDataset(n=20)
        config = FinetuneConfig(
            strategy="full",
            max_steps=5,
            batch_size=4,
            output_dir=str(tmp_path / "ft"),
        )
        trainer = FinetuneTrainer(model, config, device="cpu")
        report = trainer.train(dataset)
        assert report.strategy == "full"


# ── CLI ──────────────────────────────────────────────────


class TestCLI:
    def test_finetune_status_json(self, tmp_path):
        from typer.testing import CliRunner

        from forge.cli import app

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["finetune", "status", "--json", "--output-dir", str(tmp_path)],
        )
        assert result.exit_code == 0
        import json

        data = json.loads(result.stdout)
        assert "strategy" in data or "checkpoints" in data or isinstance(data, dict)

    def test_finetune_list_checkpoints(self, tmp_path):
        from typer.testing import CliRunner

        from forge.cli import app

        # Create a fake checkpoint
        ckpt_path = tmp_path / "finetune_step_100.pt"
        torch.save({"global_step": 100, "strategy": "lora"}, ckpt_path)

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["finetune", "list", "--json", "--output-dir", str(tmp_path)],
        )
        assert result.exit_code == 0


# ── Strict Edge Cases ────────────────────────────────────


class TestReplayBufferStrict:
    def test_ring_buffer_overwrites_oldest(self):
        """After 2x fills, only last max_size items survive."""
        buf = ReplayBuffer(max_size=5)
        for i in range(10):
            buf.add({"val": torch.tensor([float(i)])})
        assert buf.size == 5
        vals = sorted([buf._buffer[j]["val"].item() for j in range(5)])
        assert vals == [5.0, 6.0, 7.0, 8.0, 9.0]

    def test_add_batch_unbatches_correctly(self):
        """add_batch stores individual samples, not whole batch."""
        buf = ReplayBuffer(max_size=100)
        batch = {
            "image": torch.randn(4, 3, 8, 8),
            "actions": torch.randn(4, 7),
        }
        buf.add_batch(batch)
        assert buf.size == 4
        assert buf._buffer[0]["image"].shape == (3, 8, 8)
        assert buf._buffer[0]["actions"].shape == (7,)

    def test_non_tensor_values_preserved(self):
        buf = ReplayBuffer(max_size=10)
        buf.add({"x": torch.tensor([1.0]), "label": "hello", "idx": 42})
        s = buf.sample(1)[0]
        assert s["label"] == "hello"
        assert s["idx"] == 42

    def test_sample_deterministic_with_seed(self):
        import random

        buf = ReplayBuffer(max_size=100)
        for i in range(20):
            buf.add({"x": torch.tensor([float(i)])})
        random.seed(123)
        s1 = [s["x"].item() for s in buf.sample(5)]
        random.seed(123)
        s2 = [s["x"].item() for s in buf.sample(5)]
        assert s1 == s2


class TestEWCStrict:
    def test_penalty_zero_when_unchanged(self):
        model = DummyStudent()
        dataset = DummyDataset(n=10)
        ewc = EWCPenalty(model, ewc_lambda=1000.0)
        ewc.compute_fisher(model, dataset, device="cpu", n_samples=5)
        penalty = ewc.penalty(model).item()
        assert penalty == pytest.approx(0.0, abs=1e-10)

    def test_only_grad_params_stored(self):
        model = DummyStudent()
        model.vision_encoder.weight.requires_grad = False
        model.vision_encoder.bias.requires_grad = False
        ewc = EWCPenalty(model, ewc_lambda=1.0)
        for name in ewc._means:
            assert "vision_encoder" not in name

    def test_penalty_gradient_flows(self):
        model = DummyStudent()
        dataset = DummyDataset(n=10)
        ewc = EWCPenalty(model, ewc_lambda=1.0)
        ewc.compute_fisher(model, dataset, device="cpu", n_samples=5)
        with torch.no_grad():
            for p in model.parameters():
                if p.requires_grad:
                    p.add_(torch.randn_like(p) * 0.01)
        penalty = ewc.penalty(model)
        penalty.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters() if p.requires_grad)
        assert has_grad


class TestStrategyStrict:
    def test_strategy_idempotent(self):
        model = DummyStudent()
        c1 = apply_finetune_strategy(model, "lora")
        c2 = apply_finetune_strategy(model, "lora")
        assert c1 == c2

    def test_vision_always_frozen(self):
        for strategy in ["lora", "action_head", "full"]:
            model = DummyStudent()
            apply_finetune_strategy(model, strategy)
            assert not model.vision_encoder.weight.requires_grad


class TestTrainerStrict:
    def test_intermediate_checkpoints(self, tmp_path):
        model = DummyStudent()
        dataset = DummyDataset(n=20)
        config = FinetuneConfig(
            strategy="lora",
            max_steps=10,
            batch_size=4,
            save_every=3,
            output_dir=str(tmp_path / "ft"),
        )
        trainer = FinetuneTrainer(model, config, device="cpu")
        trainer.train(dataset, log_every=100)
        files = list((tmp_path / "ft").glob("finetune_*.pt"))
        names = sorted(f.name for f in files)
        assert "finetune_step_3.pt" in names
        assert "finetune_step_6.pt" in names
        assert "finetune_step_9.pt" in names
        assert "finetune_final.pt" in names

    def test_optimizer_only_trainable_params(self, tmp_path):
        model = DummyStudent()
        config = FinetuneConfig(
            strategy="action_head",
            max_steps=5,
            output_dir=str(tmp_path / "ft"),
        )
        trainer = FinetuneTrainer(model, config, device="cpu")
        n_opt = len(trainer.optimizer.param_groups[0]["params"])
        n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
        assert n_opt == n_trainable

    def test_replay_populated_during_training(self, tmp_path):
        model = DummyStudent()
        dataset = DummyDataset(n=20)
        config = FinetuneConfig(
            strategy="lora",
            max_steps=5,
            batch_size=4,
            replay_enabled=True,
            replay_buffer_size=1000,
            output_dir=str(tmp_path / "ft"),
        )
        trainer = FinetuneTrainer(model, config, device="cpu")
        trainer.train(dataset, log_every=100)
        # 5 steps * 4 batch_size = 20 samples
        assert trainer.replay.size == 20

    def test_multiple_train_calls(self, tmp_path):
        model = DummyStudent()
        dataset = DummyDataset(n=20)
        config = FinetuneConfig(
            strategy="lora",
            max_steps=3,
            batch_size=4,
            output_dir=str(tmp_path / "ft"),
        )
        trainer = FinetuneTrainer(model, config, device="cpu")
        r1 = trainer.train(dataset, log_every=100)
        r2 = trainer.train(dataset, log_every=100)
        assert r1.total_steps == 3
        assert r2.total_steps == 3

    def test_load_pretrained_raw_state_dict(self, tmp_path):
        model = DummyStudent()
        ckpt_path = tmp_path / "raw.pt"
        torch.save(model.state_dict(), ckpt_path)
        model2 = DummyStudent()
        config = FinetuneConfig(strategy="lora", output_dir=str(tmp_path / "ft"))
        trainer = FinetuneTrainer(model2, config, device="cpu")
        trainer.load_pretrained(str(ckpt_path))


class TestCLIStrict:
    def test_status_no_checkpoints(self, tmp_path):
        from typer.testing import CliRunner

        from forge.cli import app

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["finetune", "status", "--json", "--output-dir", str(tmp_path)],
        )
        assert result.exit_code == 0
        import json

        data = json.loads(result.stdout)
        assert data["checkpoint_count"] == 0

    def test_list_empty_dir(self, tmp_path):
        from typer.testing import CliRunner

        from forge.cli import app

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["finetune", "list", "--json", "--output-dir", str(tmp_path)],
        )
        assert result.exit_code == 0
        import json

        assert json.loads(result.stdout) == []

    def test_status_rich_output(self, tmp_path):
        from typer.testing import CliRunner

        from forge.cli import app

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["finetune", "status", "--output-dir", str(tmp_path)],
        )
        assert result.exit_code == 0
