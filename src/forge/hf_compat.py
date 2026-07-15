"""Hugging Face runtime compatibility helpers.

Transformers snapshots may contain trusted remote Python modules. Transformers
binds the module-cache path at import time, so changing only ``HF_HOME`` after an
import does not repair a relocated or broken cache mount.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable
from pathlib import Path


def _checkout_root() -> Path | None:
    candidates = [Path.cwd(), *Path(__file__).resolve().parents]
    return next((path for path in candidates if (path / "pyproject.toml").is_file()), None)


def _module_cache_candidates(model_path: str | Path | None = None) -> Iterable[Path]:
    if explicit := os.environ.get("HF_MODULES_CACHE"):
        yield Path(explicit).expanduser()
    if hf_home := os.environ.get("HF_HOME"):
        yield Path(hf_home).expanduser() / "modules"
    for variable in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if hub_cache := os.environ.get(variable):
            yield Path(hub_cache).expanduser().parent / "modules"

    if model_path is not None:
        try:
            resolved = Path(model_path).expanduser().resolve()
        except OSError:
            resolved = Path(model_path).expanduser().absolute()
        for parent in (resolved, *resolved.parents):
            if parent.name == "hub":
                yield parent.parent / "modules"
                break
            if parent.name == ".hf-cache":
                yield parent / "modules"
                break

    if root := _checkout_root():
        yield root.parent.parent / ".hf-cache" / "modules"
    yield Path.home() / ".cache" / "forge" / "huggingface" / "modules"


def _prepare_writable_directory(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix=".forge-write-", dir=path):
            pass
    except OSError:
        return False
    return True


def configure_transformers_module_cache(model_path: str | Path | None = None) -> Path:
    """Select a writable module cache and update already-imported Transformers.

    The returned directory is created and write-tested. No model data is read or
    downloaded by this function.
    """
    candidates: list[Path] = []
    seen: set[str] = set()
    for candidate in _module_cache_candidates(model_path):
        key = str(candidate)
        if key not in seen:
            candidates.append(candidate)
            seen.add(key)

    selected = next((path for path in candidates if _prepare_writable_directory(path)), None)
    if selected is None:
        attempted = ", ".join(str(path) for path in candidates)
        raise RuntimeError(f"No writable Hugging Face module cache found. Tried: {attempted}")

    value = str(selected.resolve())
    os.environ["HF_MODULES_CACHE"] = value

    # Transformers copies this constant into multiple modules at import time.
    # Patch each binding so this also works when Transformers was imported before
    # FORGE discovered a dead cache symlink.
    import transformers.dynamic_module_utils as dynamic_module_utils
    import transformers.utils as transformers_utils
    import transformers.utils.hub as transformers_hub

    dynamic_module_utils.HF_MODULES_CACHE = value
    transformers_hub.HF_MODULES_CACHE = value
    transformers_utils.HF_MODULES_CACHE = value
    return Path(value)


__all__ = ["configure_transformers_module_cache"]
