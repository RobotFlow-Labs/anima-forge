"""Trusted processor compatibility tests."""

from __future__ import annotations

import transformers.tokenization_utils as legacy_module
import transformers.tokenization_utils_base as base_module

from forge.processor_compat import LEGACY_TOKENIZATION_EXPORTS, install_legacy_tokenization_exports


def test_legacy_tokenization_exports_are_installed(monkeypatch) -> None:
    for name in LEGACY_TOKENIZATION_EXPORTS:
        monkeypatch.delattr(legacy_module, name, raising=False)

    install_legacy_tokenization_exports()

    for name in LEGACY_TOKENIZATION_EXPORTS:
        assert getattr(legacy_module, name) is getattr(base_module, name)


def test_install_is_idempotent() -> None:
    install_legacy_tokenization_exports()
    first = {name: getattr(legacy_module, name) for name in LEGACY_TOKENIZATION_EXPORTS}
    install_legacy_tokenization_exports()

    assert first == {name: getattr(legacy_module, name) for name in LEGACY_TOKENIZATION_EXPORTS}
