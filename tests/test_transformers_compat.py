"""Transformers 5.x OpenVLA compatibility tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import transformers

import forge.transformers_compat as compat


def test_legacy_vision2seq_auto_map_is_migrated(monkeypatch: pytest.MonkeyPatch) -> None:
    config = SimpleNamespace(
        auto_map={
            "AutoConfig": "configuration_prismatic.OpenVLAConfig",
            "AutoModelForVision2Seq": "modeling_prismatic.OpenVLAForActionPrediction",
        }
    )
    seen: dict[str, object] = {}

    def fake_from_pretrained(path: str, **kwargs):
        seen.update(path=path, **kwargs)
        return config

    monkeypatch.setattr(compat, "configure_transformers_module_cache", lambda path: Path("/tmp/modules"))
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", fake_from_pretrained)

    result = compat.load_image_text_config("models/openvla", local_files_only=True)

    assert result is config
    assert config.auto_map["AutoModelForImageTextToText"] == config.auto_map["AutoModelForVision2Seq"]
    assert seen == {
        "path": "models/openvla",
        "trust_remote_code": True,
        "local_files_only": True,
    }


def test_existing_modern_auto_map_is_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    config = SimpleNamespace(
        auto_map={
            "AutoModelForVision2Seq": "legacy.Target",
            "AutoModelForImageTextToText": "modern.Target",
        }
    )
    monkeypatch.setattr(compat, "configure_transformers_module_cache", lambda path: Path("/tmp/modules"))
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *args, **kwargs: config)

    result = compat.load_image_text_config("models/openvla", local_files_only=True)

    assert result.auto_map["AutoModelForImageTextToText"] == "modern.Target"


def test_real_openvla_config_loads_under_transformers_5() -> None:
    model_path = Path("models/openvla--openvla-7b")
    if not model_path.is_dir():
        pytest.skip("real OpenVLA weights are not installed")

    config = compat.load_image_text_config(model_path, local_files_only=True)

    assert transformers.__version__.startswith("5.")
    assert config.auto_map["AutoModelForImageTextToText"].endswith("OpenVLAForActionPrediction")
