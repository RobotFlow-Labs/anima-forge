"""Embed an immutable source revision in FORGE distribution artifacts."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

SHA_PATTERN = re.compile(r"[0-9a-f]{40}")


def _valid_sha(value: str) -> str | None:
    candidate = value.strip().lower()
    return candidate if SHA_PATTERN.fullmatch(candidate) else None


def _source_revision(root: Path) -> str:
    for key in ("FORGE_GIT_SHA", "GITHUB_SHA", "GIT_COMMIT"):
        candidate = _valid_sha(os.environ.get(key, ""))
        if candidate is not None:
            return candidate

    existing = root / "src" / "forge" / "_build_info.py"
    if existing.is_file():
        match = re.search(r'^SOURCE_GIT_SHA\s*=\s*"([0-9a-f]{40})"$', existing.read_text(encoding="utf-8"), re.M)
        if match is not None:
            return match.group(1)

    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(
            "FORGE distributions require a 40-character source revision. Build from a Git checkout "
            "or set FORGE_GIT_SHA."
        ) from exc
    candidate = _valid_sha(result.stdout)
    if candidate is None:
        raise RuntimeError("Git returned an invalid FORGE source revision")
    return candidate


class CustomBuildHook(BuildHookInterface):
    """Generate build info for both direct wheels and sdist-derived wheels."""

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        revision = _source_revision(Path(self.root))
        generated_dir = Path(self.directory) / "forge-build-info"
        generated_dir.mkdir(parents=True, exist_ok=True)
        generated = generated_dir / f"{self.target_name}-{version}.py"
        generated.write_text(
            f'"""Generated distribution source revision; do not edit."""\n\nSOURCE_GIT_SHA = "{revision}"\n',
            encoding="utf-8",
        )
        self._generated = generated
        destination = "forge/_build_info.py" if self.target_name == "wheel" else "src/forge/_build_info.py"
        build_data["force_include"][str(generated)] = destination
        build_data.setdefault("force_include_editable", {})[str(generated)] = destination

    def finalize(self, version: str, build_data: dict[str, Any], artifact_path: str) -> None:
        generated = getattr(self, "_generated", None)
        if not isinstance(generated, Path):
            return
        generated.unlink(missing_ok=True)
        try:
            generated.parent.rmdir()
        except OSError:
            pass
