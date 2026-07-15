"""Unit coverage for legacy OpenVLA remote-class shims."""

from __future__ import annotations

import sys
from types import ModuleType

from forge.openvla_loader import _patch_legacy_openvla_class


def test_legacy_class_patch_accepts_transformers5_tie_weights_kwargs(monkeypatch) -> None:
    module = ModuleType("fake_openvla_remote")
    module.timm = ModuleType("timm")
    module.timm.__version__ = "1.0.26"
    monkeypatch.setitem(sys.modules, module.__name__, module)

    class LegacyOpenVLA:
        __module__ = module.__name__

        def __init__(self):
            self.tie_calls = 0

        @property
        def _supports_sdpa(self):
            raise AttributeError("language_model is not initialized")

        def tie_weights(self):
            self.tie_calls += 1

    timm_module, original_version = _patch_legacy_openvla_class(LegacyOpenVLA)
    instance = LegacyOpenVLA()
    instance.tie_weights(recompute_mapping=False)

    assert LegacyOpenVLA._supports_sdpa is False
    assert instance.tie_calls == 1
    assert timm_module.__version__ == "0.9.16"
    assert original_version == "1.0.26"


def test_legacy_class_patch_is_idempotent(monkeypatch) -> None:
    module = ModuleType("fake_openvla_idempotent")
    module.timm = ModuleType("timm")
    module.timm.__version__ = "0.9.16"
    monkeypatch.setitem(sys.modules, module.__name__, module)

    class LegacyOpenVLA:
        __module__ = module.__name__

        def tie_weights(self):
            return "tied"

    _patch_legacy_openvla_class(LegacyOpenVLA)
    first = LegacyOpenVLA.tie_weights
    _patch_legacy_openvla_class(LegacyOpenVLA)

    assert LegacyOpenVLA.tie_weights is first
