"""Transformers 5.x cache relocation regression tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import transformers.dynamic_module_utils as dynamic_module_utils
import transformers.utils as transformers_utils
import transformers.utils.hub as transformers_hub

import forge.hf_compat as hf_compat


def _clear_cache_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("HF_MODULES_CACHE", "HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        monkeypatch.delenv(name, raising=False)


def test_configure_rewrites_already_imported_transformers_constants(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_cache_environment(monkeypatch)
    hub = tmp_path / ".hf-cache" / "hub"
    hub.mkdir(parents=True)
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    stale = "/retired/cache/modules"
    monkeypatch.setattr(dynamic_module_utils, "HF_MODULES_CACHE", stale)
    monkeypatch.setattr(transformers_hub, "HF_MODULES_CACHE", stale)
    monkeypatch.setattr(transformers_utils, "HF_MODULES_CACHE", stale)

    selected = hf_compat.configure_transformers_module_cache()

    expected = (tmp_path / ".hf-cache" / "modules").resolve()
    assert selected == expected
    assert selected.is_dir()
    assert dynamic_module_utils.HF_MODULES_CACHE == str(expected)
    assert transformers_hub.HF_MODULES_CACHE == str(expected)
    assert transformers_utils.HF_MODULES_CACHE == str(expected)


def test_broken_explicit_cache_falls_back_to_hub_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_cache_environment(monkeypatch)
    blocked_parent = tmp_path / "blocked"
    blocked_parent.write_text("not a directory", encoding="utf-8")
    monkeypatch.setenv("HF_MODULES_CACHE", str(blocked_parent / "modules"))
    hub = tmp_path / "healthy" / "hub"
    hub.mkdir(parents=True)
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))

    selected = hf_compat.configure_transformers_module_cache()

    assert selected == (tmp_path / "healthy" / "modules").resolve()


def test_model_snapshot_path_infers_sibling_modules_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_cache_environment(monkeypatch)
    snapshot = tmp_path / ".hf-cache" / "hub" / "models--org--name" / "snapshots" / "sha"
    snapshot.mkdir(parents=True)
    monkeypatch.setattr(hf_compat, "_checkout_root", lambda: None)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))

    selected = hf_compat.configure_transformers_module_cache(snapshot)

    assert selected == (tmp_path / ".hf-cache" / "modules").resolve()
