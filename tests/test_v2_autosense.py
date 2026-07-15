"""Tests for PRD-25: AutoSense — Dynamic Model Config Detection."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pytest

from forge.autosense import (
    apply_autosense,
    autosense_config,
    sense_language_model,
    sense_model_roles,
    sense_teacher,
    sense_vision_encoder,
)
from forge.config import ForgeConfig


@pytest.fixture
def tmp_model_dir(tmp_path: Path) -> Path:
    """Create a temporary model directory with mock config.json files."""
    return tmp_path


def _write_config(path: Path, data: dict) -> None:
    """Helper to write a config.json."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps(data))


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (
            {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"], "hidden_size": 1024, "vocab_size": 151936},
            frozenset({"language"}),
        ),
        (
            {
                "model_type": "vjepa2",
                "architectures": ["VJEPA2Model"],
                "hidden_size": 1024,
                "image_size": 256,
                "patch_size": 16,
            },
            frozenset({"vision"}),
        ),
        (
            {
                "model_type": "qwen3_vl",
                "architectures": ["Qwen3VLForConditionalGeneration"],
                "vision_config": {"hidden_size": 1024},
                "text_config": {"hidden_size": 2048, "vocab_size": 151936},
            },
            frozenset({"vision", "language"}),
        ),
        (
            {
                "model_type": "siglip",
                "vision_config": {"hidden_size": 1152},
                "text_config": {"hidden_size": 1152, "vocab_size": 256000},
            },
            frozenset({"vision"}),
        ),
    ],
)
def test_model_roles_distinguish_language_vision_and_multimodal(
    tmp_model_dir: Path,
    config: dict,
    expected: frozenset[str],
) -> None:
    model_path = tmp_model_dir / "model"
    _write_config(model_path, config)

    assert sense_model_roles(model_path) == expected


def test_benchmark_scan_counts_unique_configs_and_preserves_roles(tmp_model_dir: Path) -> None:
    from forge.benchmark.suites.bench_06_autosense import scan_model_configs

    _write_config(
        tmp_model_dir / "language",
        {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"], "hidden_size": 1024, "vocab_size": 151936},
    )
    _write_config(
        tmp_model_dir / "vision",
        {"model_type": "vjepa2", "hidden_size": 1024, "image_size": 256, "patch_size": 16},
    )
    _write_config(
        tmp_model_dir / "multimodal",
        {
            "model_type": "qwen3_vl",
            "architectures": ["Qwen3VLForConditionalGeneration"],
            "vision_config": {"hidden_size": 1024},
            "text_config": {"hidden_size": 2048, "vocab_size": 151936},
        },
    )

    scanned, vision, language, _timings = scan_model_configs(tmp_model_dir)

    assert scanned == ["language", "multimodal", "vision"]
    assert set(vision) == {"multimodal", "vision"}
    assert set(language) == {"language", "multimodal"}
    assert len(set(vision) | set(language)) == len(scanned)


# ── sense_vision_encoder ──────────────────────────────────


class TestSenseVisionEncoder:
    def test_siglip_config(self, tmp_model_dir: Path):
        """SigLIP: reads vision_config.hidden_size."""
        model_path = tmp_model_dir / "google--siglip-so400m-patch14-384"
        _write_config(
            model_path,
            {
                "model_type": "siglip",
                "vision_config": {
                    "hidden_size": 1152,
                    "image_size": 384,
                    "patch_size": 14,
                    "num_hidden_layers": 27,
                },
            },
        )
        result = sense_vision_encoder(model_path)
        assert result is not None
        assert result["d_output"] == 1152
        assert result["image_size"] == 384
        assert result["patch_size"] == 14
        assert result["n_tokens"] == (384 // 14) ** 2  # 729

    def test_dinov2_config(self, tmp_model_dir: Path):
        """DINOv2: reads top-level hidden_size."""
        model_path = tmp_model_dir / "facebook--dinov2-small"
        _write_config(
            model_path,
            {
                "model_type": "dinov2",
                "hidden_size": 384,
                "image_size": 518,
                "patch_size": 14,
                "num_hidden_layers": 12,
            },
        )
        result = sense_vision_encoder(model_path)
        assert result is not None
        assert result["d_output"] == 384
        assert result["n_tokens"] == (518 // 14) ** 2  # 1369

    def test_dinov2_base_config(self, tmp_model_dir: Path):
        """DINOv2-base: d=768."""
        model_path = tmp_model_dir / "facebook--dinov2-base"
        _write_config(
            model_path,
            {
                "model_type": "dinov2",
                "hidden_size": 768,
                "image_size": 518,
                "patch_size": 14,
            },
        )
        result = sense_vision_encoder(model_path)
        assert result is not None
        assert result["d_output"] == 768

    def test_theia_config(self, tmp_model_dir: Path):
        """Theia: reads top-level hidden_size."""
        model_path = tmp_model_dir / "theia--theia-tiny"
        _write_config(
            model_path,
            {
                "hidden_size": 384,
                "image_size": 384,
                "patch_size": 16,
            },
        )
        result = sense_vision_encoder(model_path)
        assert result is not None
        assert result["d_output"] == 384
        assert result["n_tokens"] == (384 // 16) ** 2  # 576

    def test_missing_config(self, tmp_model_dir: Path):
        """Returns None when config.json is missing."""
        model_path = tmp_model_dir / "nonexistent-model"
        model_path.mkdir(parents=True, exist_ok=True)
        result = sense_vision_encoder(model_path)
        assert result is None

    def test_missing_dir(self, tmp_model_dir: Path):
        """Returns None when directory doesn't exist."""
        result = sense_vision_encoder(tmp_model_dir / "does-not-exist")
        assert result is None

    def test_invalid_json(self, tmp_model_dir: Path):
        """Returns None for invalid JSON."""
        model_path = tmp_model_dir / "bad-model"
        model_path.mkdir(parents=True, exist_ok=True)
        (model_path / "config.json").write_text("not json {{{")
        result = sense_vision_encoder(model_path)
        assert result is None

    def test_no_hidden_size(self, tmp_model_dir: Path):
        """Returns None when config has no hidden_size."""
        model_path = tmp_model_dir / "empty-model"
        _write_config(model_path, {"model_type": "something", "foo": "bar"})
        result = sense_vision_encoder(model_path)
        assert result is None

    def test_n_tokens_without_patch_info(self, tmp_model_dir: Path):
        """n_tokens not computed when patch_size or image_size missing."""
        model_path = tmp_model_dir / "no-patch"
        _write_config(model_path, {"hidden_size": 384})
        result = sense_vision_encoder(model_path)
        assert result is not None
        assert result["d_output"] == 384
        assert "n_tokens" not in result


# ── sense_language_model ──────────────────────────────────


class TestSenseLanguageModel:
    def test_qwen25_05b(self, tmp_model_dir: Path):
        """Qwen2.5-0.5B: d=896, vocab=151936."""
        model_path = tmp_model_dir / "Qwen--Qwen2.5-0.5B"
        _write_config(
            model_path,
            {
                "model_type": "qwen2",
                "hidden_size": 896,
                "vocab_size": 151936,
                "num_hidden_layers": 24,
                "num_attention_heads": 14,
            },
        )
        result = sense_language_model(model_path)
        assert result is not None
        assert result["d_model"] == 896
        assert result["vocab_size"] == 151936
        assert result["n_layers"] == 24
        assert result["n_heads"] == 14

    def test_qwen25_15b(self, tmp_model_dir: Path):
        """Qwen2.5-1.5B: d=1536."""
        model_path = tmp_model_dir / "Qwen--Qwen2.5-1.5B"
        _write_config(
            model_path,
            {
                "model_type": "qwen2",
                "hidden_size": 1536,
                "vocab_size": 151936,
                "num_hidden_layers": 28,
                "num_attention_heads": 12,
            },
        )
        result = sense_language_model(model_path)
        assert result is not None
        assert result["d_model"] == 1536

    def test_qwen35_nested_text_config(self, tmp_model_dir: Path):
        """Qwen3.5 dimensions live under its multimodal text_config wrapper."""
        model_path = tmp_model_dir / "Qwen--Qwen3.5-0.8B"
        _write_config(
            model_path,
            {
                "model_type": "qwen3_5",
                "text_config": {
                    "hidden_size": 1024,
                    "vocab_size": 248320,
                    "num_hidden_layers": 24,
                    "num_attention_heads": 8,
                },
            },
        )
        result = sense_language_model(model_path)
        assert result == {
            "d_model": 1024,
            "vocab_size": 248320,
            "n_layers": 24,
            "n_heads": 8,
        }

    def test_missing_config(self, tmp_model_dir: Path):
        """Returns None when config.json is missing."""
        model_path = tmp_model_dir / "no-lm"
        model_path.mkdir(parents=True, exist_ok=True)
        result = sense_language_model(model_path)
        assert result is None


# ── sense_teacher ─────────────────────────────────────────


class TestSenseTeacher:
    def test_teacher_with_action_dim(self, tmp_model_dir: Path):
        """Reads action_dim from teacher config."""
        model_path = tmp_model_dir / "teacher-model"
        _write_config(model_path, {"action_dim": 7, "action_horizon": 8})
        result = sense_teacher(model_path)
        assert result is not None
        assert result["action_dim"] == 7
        assert result["action_horizon"] == 8

    def test_missing_teacher(self, tmp_model_dir: Path):
        """Returns None for missing teacher dir."""
        result = sense_teacher(tmp_model_dir / "nope")
        assert result is None


# ── autosense_config ──────────────────────────────────────


class TestAutosenseConfig:
    def test_full_pipeline(self, tmp_model_dir: Path):
        """Combines vision + LM into override dict."""
        _write_config(
            tmp_model_dir / "vision-enc",
            {
                "vision_config": {"hidden_size": 1152, "image_size": 384, "patch_size": 14},
            },
        )
        _write_config(
            tmp_model_dir / "lang-model",
            {
                "hidden_size": 896,
                "vocab_size": 151936,
            },
        )
        overrides = autosense_config(tmp_model_dir, "vision-enc", "lang-model")
        assert overrides["bridge_d_vision"] == 1152
        assert overrides["bridge_d_model"] == 896
        assert overrides["n_tokens"] == 729

    def test_partial_vision_only(self, tmp_model_dir: Path):
        """Returns only vision overrides when LM missing."""
        _write_config(
            tmp_model_dir / "vision-enc",
            {
                "vision_config": {"hidden_size": 384},
            },
        )
        overrides = autosense_config(tmp_model_dir, "vision-enc", "missing-lm")
        assert overrides["bridge_d_vision"] == 384
        assert "bridge_d_model" not in overrides

    def test_empty_when_nothing_found(self, tmp_model_dir: Path):
        """Returns empty dict when nothing detected."""
        overrides = autosense_config(tmp_model_dir, "nope", "nope")
        assert overrides == {}


# ── apply_autosense ───────────────────────────────────────


class TestApplyAutosense:
    def test_mutates_config(self, tmp_model_dir: Path):
        """apply_autosense updates config fields in-place."""
        _write_config(
            tmp_model_dir / "google--siglip-so400m-patch14-384",
            {
                "vision_config": {"hidden_size": 1152, "image_size": 384, "patch_size": 14},
            },
        )
        _write_config(
            tmp_model_dir / "Qwen--Qwen2.5-1.5B",
            {
                "hidden_size": 1536,
                "vocab_size": 151936,
            },
        )

        config = ForgeConfig.default()
        config.paths.vision_encoder = "google--siglip-so400m-patch14-384"
        config.paths.language_model = "Qwen--Qwen2.5-1.5B"

        apply_autosense(config, tmp_model_dir)
        assert config.student.bridge_d_model == 1536

    def test_logs_overrides(self, tmp_model_dir: Path, monkeypatch):
        """apply_autosense logs what it changed."""
        _write_config(
            tmp_model_dir / "Qwen--Qwen2.5-1.5B",
            {
                "hidden_size": 1536,
            },
        )

        config = ForgeConfig.default()
        config.paths.language_model = "Qwen--Qwen2.5-1.5B"
        messages = []
        monkeypatch.setattr("forge.autosense.logger.info", lambda message: messages.append(message))

        apply_autosense(config, tmp_model_dir)

        assert any("bridge_d_model" in msg for msg in messages)

    def test_disabled_via_config(self, tmp_model_dir: Path):
        """Skips when autosense=False."""
        _write_config(
            tmp_model_dir / "Qwen--Qwen2.5-1.5B",
            {
                "hidden_size": 1536,
            },
        )

        config = ForgeConfig.default()
        config.student.autosense = False
        config.paths.language_model = "Qwen--Qwen2.5-1.5B"

        apply_autosense(config, tmp_model_dir)
        assert config.student.bridge_d_model == 1024  # Canonical Qwen3 nano default is unchanged

    def test_missing_model_dir(self):
        """No-op when model_dir doesn't exist."""
        config = ForgeConfig.default()
        apply_autosense(config, Path("/nonexistent/path"))
        assert config.student.bridge_d_vision == 1152  # Unchanged

    def test_no_change_when_matching(self, tmp_model_dir: Path, caplog):
        """No log output when values already match defaults."""
        _write_config(
            tmp_model_dir / "google--siglip-so400m-patch14-384",
            {
                "vision_config": {"hidden_size": 1152, "image_size": 384, "patch_size": 14},
            },
        )
        _write_config(
            tmp_model_dir / "Qwen--Qwen2.5-0.5B",
            {
                "hidden_size": 896,
            },
        )

        config = ForgeConfig.default()
        with caplog.at_level(logging.INFO, logger="forge.autosense"):
            apply_autosense(config, tmp_model_dir)

        # Should not log any overrides since values match defaults
        assert not any("override" in msg.lower() for msg in caplog.messages)


# ── n_tokens calculation ──────────────────────────────────


class TestNTokensCalculation:
    @pytest.mark.parametrize(
        "image_size,patch_size,expected",
        [
            (384, 14, 729),  # SigLIP default: (384/14)² = 27² = 729
            (384, 16, 576),  # Theia: (384/16)² = 24² = 576
            (518, 14, 1369),  # DINOv2 default: (518/14)² = 37² = 1369
            (224, 16, 196),  # ViT-base: (224/16)² = 14² = 196
        ],
    )
    def test_n_tokens_from_image_patch(self, tmp_model_dir: Path, image_size, patch_size, expected):
        model_path = tmp_model_dir / f"test-{image_size}-{patch_size}"
        _write_config(
            model_path,
            {
                "hidden_size": 384,
                "image_size": image_size,
                "patch_size": patch_size,
            },
        )
        result = sense_vision_encoder(model_path)
        assert result["n_tokens"] == expected


# ── Real model configs (if available) ─────────────────────


DATASETS_DIR = Path(os.environ.get("FORGE_MODEL_DIR", "./models"))


@pytest.mark.skipif(
    not (DATASETS_DIR / "google--siglip-so400m-patch14-384" / "config.json").exists(),
    reason="Real SigLIP model not available",
)
class TestRealModels:
    def test_real_siglip(self):
        result = sense_vision_encoder(DATASETS_DIR / "google--siglip-so400m-patch14-384")
        assert result is not None
        assert result["d_output"] == 1152
        assert result["n_tokens"] == 729

    def test_real_qwen_05b(self):
        result = sense_language_model(DATASETS_DIR / "Qwen--Qwen2.5-0.5B")
        assert result is not None
        assert result["d_model"] == 896
        assert result["vocab_size"] == 151936

    def test_real_qwen_15b(self):
        result = sense_language_model(DATASETS_DIR / "Qwen--Qwen2.5-1.5B")
        assert result is not None
        assert result["d_model"] == 1536

    def test_real_autosense_config(self):
        overrides = autosense_config(
            DATASETS_DIR,
            "google--siglip-so400m-patch14-384",
            "Qwen--Qwen2.5-0.5B",
        )
        assert overrides["bridge_d_vision"] == 1152
        assert overrides["bridge_d_model"] == 896
