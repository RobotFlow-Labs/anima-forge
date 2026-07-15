#!/usr/bin/env python3
"""Verify that an installed FORGE wheel exposes every mandatory runtime."""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from importlib.resources import files
from typing import Any


def verify_installed_runtime(*, expected_version: str, require_cpu: bool) -> dict[str, Any]:
    import torch
    from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy  # type: ignore[import-untyped]
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # type: ignore[import-untyped]
    from lerobot.policies.vla_jepa.modeling_vla_jepa import VLAJEPAPolicy  # type: ignore[import-untyped]

    import forge
    from forge.teachers.molmoact2_adapter import MolmoAct2Adapter
    from forge.teachers.rdt2_adapter import RDT2Adapter
    from forge.teachers.registry import get_registry
    from forge.teachers.smolvla_adapter import SmolVLAAdapter
    from forge.teachers.vla_jepa_adapter import VLAJEPAAdapter
    from forge.vendor.rdt2 import RDTRunner

    installed_version = version("anima-forge")
    if installed_version != expected_version or forge.__version__ != expected_version:
        raise RuntimeError(
            f"Installed FORGE version mismatch: metadata={installed_version}, package={forge.__version__}, "
            f"expected={expected_version}"
        )
    if require_cpu and torch.version.cuda is not None:
        raise RuntimeError(f"CPU installation resolved a CUDA-enabled torch build ({torch.version.cuda})")
    if not files("forge.web").joinpath("dashboard.html").is_file():
        raise RuntimeError("Installed wheel is missing forge.web/dashboard.html")

    expected_adapters = {
        "molmoact2-libero": MolmoAct2Adapter,
        "rdt2-fm": RDT2Adapter,
        "smolvla-base": SmolVLAAdapter,
        "vla-jepa-3b": VLAJEPAAdapter,
    }
    registry = get_registry()
    available = set(registry.list_teachers())
    missing = sorted(expected_adapters.keys() - available)
    if missing:
        raise RuntimeError(f"Installed teacher registry is missing mandatory adapters: {', '.join(missing)}")
    for name, adapter_class in expected_adapters.items():
        adapter = registry.create(name)
        if type(adapter) is not adapter_class:
            raise RuntimeError(
                f"Installed teacher registry mapped {name!r} to {type(adapter).__name__}, "
                f"expected {adapter_class.__name__}"
            )

    mandatory_classes = (RDTRunner, MolmoAct2Policy, SmolVLAPolicy, VLAJEPAPolicy)
    if not all(isinstance(runtime, type) for runtime in mandatory_classes):
        raise RuntimeError("One or more mandatory teacher runtime classes could not be imported")

    return {
        "schema": "forge.installed-runtime-smoke.v1",
        "status": "passed",
        "version": installed_version,
        "torch_cuda": torch.version.cuda,
        "teachers": sorted(expected_adapters),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-version", required=True)
    parser.add_argument("--require-cpu", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            verify_installed_runtime(
                expected_version=args.expected_version,
                require_cpu=args.require_cpu,
            ),
            allow_nan=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
