"""Tests for PRD-26: Trained Student Model Registry."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
import torch

from forge.config import ForgeConfig
from forge.model_registry import ModelEntry, ModelRegistry, _config_hash, _generate_model_id
from forge.trainer import TrainingReport


@pytest.fixture
def registry_dir(tmp_path: Path) -> Path:
    return tmp_path / "registry"


@pytest.fixture
def registry(registry_dir: Path) -> ModelRegistry:
    return ModelRegistry(registry_dir)


@pytest.fixture
def dummy_checkpoint(tmp_path: Path) -> Path:
    """Create a minimal checkpoint file."""
    ckpt_path = tmp_path / "checkpoints" / "best.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"global_step": 1000, "student_state_dict": {}}, ckpt_path)
    return ckpt_path


@pytest.fixture
def config() -> ForgeConfig:
    return ForgeConfig.default()


# ── ModelEntry ────────────────────────────────────────────


class TestModelEntry:
    def test_create_entry(self):
        entry = ModelEntry(
            model_id="abc123",
            name="test-model",
            variant="nano",
            checkpoint_path="/tmp/test.pt",
            created_at=time.time(),
        )
        assert entry.model_id == "abc123"
        assert entry.variant == "nano"

    def test_to_dict_roundtrip(self):
        entry = ModelEntry(
            model_id="abc123",
            name="test-model",
            variant="nano",
            checkpoint_path="/tmp/test.pt",
            created_at=1234567890.0,
            tags=["production"],
            metrics={"latency_ms": 45.2},
        )
        d = entry.to_dict()
        restored = ModelEntry.from_dict(d)
        assert restored.model_id == entry.model_id
        assert restored.tags == ["production"]
        assert restored.metrics["latency_ms"] == 45.2

    def test_summary(self):
        entry = ModelEntry(
            model_id="abcdef1234567890",
            name="nano-test",
            variant="nano",
            checkpoint_path="/tmp/test.pt",
            created_at=time.time(),
            total_steps=5000,
            final_loss=0.023,
            best_loss=0.021,
            tags=["best"],
        )
        s = entry.summary()
        assert "abcdef12" in s
        assert "nano-test" in s
        assert "0.0230" in s

    def test_age_hours(self):
        entry = ModelEntry(
            model_id="abc",
            name="test",
            variant="nano",
            checkpoint_path="/tmp/test.pt",
            created_at=time.time() - 7200,  # 2 hours ago
        )
        assert 1.9 < entry.age_hours < 2.1


# ── Helper Functions ──────────────────────────────────────


class TestHelpers:
    def test_generate_model_id_unique(self):
        id1 = _generate_model_id("nano", "abc", 1.0)
        id2 = _generate_model_id("nano", "abc", 2.0)
        assert id1 != id2
        assert len(id1) == 16

    def test_config_hash_deterministic(self):
        config = ForgeConfig.default()
        h1 = _config_hash(config)
        h2 = _config_hash(config)
        assert h1 == h2

    def test_config_hash_changes_with_variant(self):
        c1 = ForgeConfig.default()
        c2 = ForgeConfig.default()
        c2.student.variant = "small"
        assert _config_hash(c1) != _config_hash(c2)


# ── ModelRegistry ─────────────────────────────────────────


class TestModelRegistry:
    def test_create_empty_registry(self, registry: ModelRegistry):
        assert registry.count == 0

    def test_register_model(self, registry: ModelRegistry, dummy_checkpoint: Path, config: ForgeConfig):
        entry = registry.register(
            checkpoint_path=dummy_checkpoint,
            variant="nano",
            config=config,
        )
        assert entry.model_id
        assert entry.variant == "nano"
        assert registry.count == 1

    def test_register_with_metrics(self, registry: ModelRegistry, dummy_checkpoint: Path, config: ForgeConfig):
        entry = registry.register(
            checkpoint_path=dummy_checkpoint,
            variant="nano",
            config=config,
            metrics={"latency_ms": 45.2, "throughput_fps": 22.1},
        )
        assert entry.metrics["latency_ms"] == 45.2
        assert entry.metrics["throughput_fps"] == 22.1

    def test_register_with_training_report(self, registry: ModelRegistry, dummy_checkpoint: Path, config: ForgeConfig):
        report = TrainingReport(
            total_steps=10000,
            final_loss=0.023,
            best_loss=0.021,
            device="cuda",
        )
        entry = registry.register(
            checkpoint_path=dummy_checkpoint,
            variant="nano",
            config=config,
            training_report=report,
        )
        assert entry.total_steps == 10000
        assert entry.final_loss == 0.023
        assert entry.best_loss == 0.021
        assert entry.training_device == "cuda"

    def test_register_extracts_config(self, registry: ModelRegistry, dummy_checkpoint: Path, config: ForgeConfig):
        entry = registry.register(
            checkpoint_path=dummy_checkpoint,
            variant="nano",
            config=config,
        )
        assert entry.vision_encoder == config.student.vision_encoder
        assert entry.language_model == config.student.language_model
        assert entry.bridge_d_vision == config.student.bridge_d_vision
        assert entry.bridge_d_model == config.student.bridge_d_model
        assert entry.parent_teacher == config.paths.teacher

    def test_register_auto_names(self, registry: ModelRegistry, dummy_checkpoint: Path, config: ForgeConfig):
        entry = registry.register(
            checkpoint_path=dummy_checkpoint,
            variant="nano",
            config=config,
        )
        assert "nano" in entry.name

    def test_register_custom_name(self, registry: ModelRegistry, dummy_checkpoint: Path):
        entry = registry.register(
            checkpoint_path=dummy_checkpoint,
            variant="nano",
            name="my-special-model",
        )
        assert entry.name == "my-special-model"

    def test_get_by_full_id(self, registry: ModelRegistry, dummy_checkpoint: Path):
        entry = registry.register(checkpoint_path=dummy_checkpoint, variant="nano")
        found = registry.get(entry.model_id)
        assert found is not None
        assert found.model_id == entry.model_id

    def test_get_by_prefix(self, registry: ModelRegistry, dummy_checkpoint: Path):
        entry = registry.register(checkpoint_path=dummy_checkpoint, variant="nano")
        found = registry.get(entry.model_id[:8])
        assert found is not None
        assert found.model_id == entry.model_id

    def test_get_not_found(self, registry: ModelRegistry):
        assert registry.get("nonexistent") is None

    def test_list_all(self, registry: ModelRegistry, dummy_checkpoint: Path):
        registry.register(checkpoint_path=dummy_checkpoint, variant="nano")
        time.sleep(0.01)  # Ensure different timestamps
        registry.register(checkpoint_path=dummy_checkpoint, variant="small")
        entries = registry.list_models()
        assert len(entries) == 2

    def test_list_filter_variant(self, registry: ModelRegistry, dummy_checkpoint: Path):
        registry.register(checkpoint_path=dummy_checkpoint, variant="nano")
        registry.register(checkpoint_path=dummy_checkpoint, variant="small")
        entries = registry.list_models(variant="nano")
        assert len(entries) == 1
        assert entries[0].variant == "nano"

    def test_list_filter_tag(self, registry: ModelRegistry, dummy_checkpoint: Path):
        e1 = registry.register(checkpoint_path=dummy_checkpoint, variant="nano", tags=["production"])
        registry.register(checkpoint_path=dummy_checkpoint, variant="small")
        entries = registry.list_models(tag="production")
        assert len(entries) == 1
        assert entries[0].model_id == e1.model_id


# ── Best / Promote / Compare ─────────────────────────────


class TestBestPromoteCompare:
    def test_best_by_loss(self, registry: ModelRegistry, dummy_checkpoint: Path, config: ForgeConfig):
        r1 = TrainingReport(total_steps=100, best_loss=0.05)
        r2 = TrainingReport(total_steps=200, best_loss=0.02)
        registry.register(checkpoint_path=dummy_checkpoint, variant="nano", config=config, training_report=r1)
        time.sleep(0.01)
        registry.register(checkpoint_path=dummy_checkpoint, variant="nano", config=config, training_report=r2)
        best = registry.best(by="best_loss")
        assert best is not None
        assert best.best_loss == 0.02

    def test_best_by_metric(self, registry: ModelRegistry, dummy_checkpoint: Path):
        registry.register(checkpoint_path=dummy_checkpoint, variant="nano", metrics={"latency_ms": 50.0})
        time.sleep(0.01)
        registry.register(checkpoint_path=dummy_checkpoint, variant="nano", metrics={"latency_ms": 30.0})
        best = registry.best(by="latency_ms")
        assert best is not None
        assert best.metrics["latency_ms"] == 30.0

    def test_best_higher_is_better(self, registry: ModelRegistry, dummy_checkpoint: Path):
        registry.register(checkpoint_path=dummy_checkpoint, variant="nano", metrics={"throughput_fps": 10.0})
        time.sleep(0.01)
        registry.register(checkpoint_path=dummy_checkpoint, variant="nano", metrics={"throughput_fps": 25.0})
        best = registry.best(by="throughput_fps", lower_is_better=False)
        assert best is not None
        assert best.metrics["throughput_fps"] == 25.0

    def test_best_empty_registry(self, registry: ModelRegistry):
        assert registry.best() is None

    def test_promote(self, registry: ModelRegistry, dummy_checkpoint: Path):
        e1 = registry.register(checkpoint_path=dummy_checkpoint, variant="nano")
        promoted = registry.promote(e1.model_id, tag="production")
        assert promoted is not None
        assert "production" in promoted.tags

    def test_promote_removes_tag_from_others(self, registry: ModelRegistry, dummy_checkpoint: Path):
        e1 = registry.register(checkpoint_path=dummy_checkpoint, variant="nano", tags=["production"])
        time.sleep(0.01)
        e2 = registry.register(checkpoint_path=dummy_checkpoint, variant="small")
        registry.promote(e2.model_id, tag="production")
        # e1 should no longer have "production"
        e1_refreshed = registry.get(e1.model_id)
        e2_refreshed = registry.get(e2.model_id)
        assert "production" not in e1_refreshed.tags
        assert "production" in e2_refreshed.tags

    def test_promote_not_found(self, registry: ModelRegistry):
        assert registry.promote("nonexistent") is None

    def test_compare(self, registry: ModelRegistry, dummy_checkpoint: Path):
        c1 = ForgeConfig.default()
        c2 = ForgeConfig.default()
        c2.student.bridge_d_model = 1536
        e1 = registry.register(checkpoint_path=dummy_checkpoint, variant="nano", config=c1)
        time.sleep(0.01)
        e2 = registry.register(checkpoint_path=dummy_checkpoint, variant="nano", config=c2)
        result = registry.compare(e1.model_id, e2.model_id)
        assert "differences" in result
        assert "bridge_d_model" in result["differences"]
        assert result["differences"]["bridge_d_model"]["model_1"] == 1024
        assert result["differences"]["bridge_d_model"]["model_2"] == 1536

    def test_compare_not_found(self, registry: ModelRegistry):
        result = registry.compare("a", "b")
        assert "error" in result


# ── Delete ────────────────────────────────────────────────


class TestDelete:
    def test_delete(self, registry: ModelRegistry, dummy_checkpoint: Path):
        entry = registry.register(checkpoint_path=dummy_checkpoint, variant="nano")
        assert registry.count == 1
        registry.delete(entry.model_id)
        assert registry.count == 0

    def test_delete_not_found(self, registry: ModelRegistry):
        assert registry.delete("nonexistent") is False

    def test_delete_with_checkpoint(self, registry: ModelRegistry, tmp_path: Path):
        ckpt = tmp_path / "model.pt"
        torch.save({}, ckpt)
        entry = registry.register(checkpoint_path=ckpt, variant="nano")
        registry.delete(entry.model_id, delete_checkpoint=True)
        assert not ckpt.exists()


# ── Persistence ───────────────────────────────────────────


class TestPersistence:
    def test_persists_to_disk(self, registry_dir: Path, dummy_checkpoint: Path, config: ForgeConfig):
        reg1 = ModelRegistry(registry_dir)
        reg1.register(checkpoint_path=dummy_checkpoint, variant="nano", config=config)

        # Load fresh
        reg2 = ModelRegistry(registry_dir)
        assert reg2.count == 1

    def test_registry_json_format(self, registry_dir: Path, dummy_checkpoint: Path):
        reg = ModelRegistry(registry_dir)
        reg.register(checkpoint_path=dummy_checkpoint, variant="nano")

        data = json.loads((registry_dir / "registry.json").read_text())
        assert data["version"] == 1
        assert data["count"] == 1
        assert len(data["models"]) == 1

    def test_copy_checkpoint(self, registry_dir: Path, dummy_checkpoint: Path):
        reg = ModelRegistry(registry_dir)
        entry = reg.register(
            checkpoint_path=dummy_checkpoint,
            variant="nano",
            copy_checkpoint=True,
        )
        # Checkpoint should be copied to models dir
        copied_path = Path(entry.checkpoint_path)
        assert copied_path.exists()
        assert "models" in str(copied_path)


# ── CLI Integration ───────────────────────────────────────


class TestCLI:
    def test_models_list_json(self, registry_dir: Path, dummy_checkpoint: Path):
        """Test forge models list --json."""
        from typer.testing import CliRunner

        from forge.cli import app

        # Register a model first
        reg = ModelRegistry(registry_dir)
        reg.register(checkpoint_path=dummy_checkpoint, variant="nano", name="test-cli")

        runner = CliRunner()
        result = runner.invoke(app, ["models", "list", "--json", "--registry-dir", str(registry_dir)])
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert len(data) == 1
        assert data[0]["name"] == "test-cli"

    def test_models_best_json(self, registry_dir: Path, dummy_checkpoint: Path):
        """Test forge models best --json."""
        from typer.testing import CliRunner

        from forge.cli import app

        reg = ModelRegistry(registry_dir)
        report = TrainingReport(total_steps=100, best_loss=0.05)
        reg.register(checkpoint_path=dummy_checkpoint, variant="nano", training_report=report)

        runner = CliRunner()
        result = runner.invoke(app, ["models", "best", "--json", "--registry-dir", str(registry_dir)])
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert data["best_loss"] == 0.05


# ── Integration with TrainingReport ───────────────────────


class TestIntegration:
    def test_full_workflow(self, registry_dir: Path, dummy_checkpoint: Path):
        """Full workflow: register → list → best → promote → compare."""
        config = ForgeConfig.default()
        reg = ModelRegistry(registry_dir)

        # Register two models
        r1 = TrainingReport(total_steps=5000, final_loss=0.05, best_loss=0.04)
        e1 = reg.register(
            checkpoint_path=dummy_checkpoint,
            variant="nano",
            config=config,
            training_report=r1,
            metrics={"latency_ms": 50.0},
        )

        time.sleep(0.01)
        config2 = ForgeConfig.default()
        config2.student.bridge_d_model = 1536
        r2 = TrainingReport(total_steps=10000, final_loss=0.02, best_loss=0.018)
        e2 = reg.register(
            checkpoint_path=dummy_checkpoint,
            variant="nano",
            config=config2,
            training_report=r2,
            metrics={"latency_ms": 35.0},
        )

        # List
        assert reg.count == 2
        all_models = reg.list_models()
        assert len(all_models) == 2

        # Best
        best = reg.best(by="best_loss")
        assert best.model_id == e2.model_id

        best_latency = reg.best(by="latency_ms")
        assert best_latency.model_id == e2.model_id  # 35 < 50

        # Promote
        reg.promote(e2.model_id, tag="production")
        prod = reg.list_models(tag="production")
        assert len(prod) == 1
        assert prod[0].model_id == e2.model_id

        # Compare
        comparison = reg.compare(e1.model_id, e2.model_id)
        assert "bridge_d_model" in comparison["differences"]
