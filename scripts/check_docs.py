#!/usr/bin/env python3
"""Validate public documentation and optional private FORGE skill commands."""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
import textwrap
from pathlib import Path

from typer.main import get_command

from forge.cli_v2 import app

SKILL_DIR = Path(".claude/skills/forge")
CLAUDE_DIR = Path(".claude")
PUBLIC_DOCS = Path("docs")
DEAD_REFERENCE = re.compile(
    "|".join(
        (
            "datai_" + "srv",
            "/" + "home/datai",
            "/" + "mnt/forge-data",
            "/" + "mnt/development",
            "v2/multi-path-distillation",
            "REPORT_GPU" + ".md",
            "huggingface" + "-cli",
            "--extra " + "dev",
            "--extra " + "cuda",
        )
    )
)
MARKDOWN_LINK = re.compile(r"(?<!!)\[[^]]+\]\(([^)]+)\)")


def _forge_command_paths(markdown: str) -> set[tuple[str, ...]]:
    root = get_command(app)
    paths: set[tuple[str, ...]] = set()
    for block in re.findall(r"```(?:bash|shell)\n(.*?)```", markdown, flags=re.DOTALL):
        logical = block.replace("\\\n", " ")
        for raw_line in logical.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            tokens = shlex.split(line)
            if "forge" not in tokens:
                continue
            remaining = tokens[tokens.index("forge") + 1 :]
            command = root
            path: list[str] = []
            for token in remaining:
                children = getattr(command, "commands", {})
                if token.startswith("-") or token not in children:
                    break
                path.append(token)
                command = children[token]
            paths.add(tuple(path))
    return paths


def check_skill() -> list[str]:
    errors: list[str] = []
    files = sorted(SKILL_DIR.glob("*.md"))
    if not files and not (SKILL_DIR / "SKILL.md").is_file():
        if not list(CLAUDE_DIR.rglob("*.md")):
            return []
        return [f"incomplete private skill at {SKILL_DIR}"]
    if not files or not (SKILL_DIR / "SKILL.md").is_file():
        return [f"incomplete private skill at {SKILL_DIR}"]

    combined = ""
    for path in files:
        text = path.read_text(encoding="utf-8")
        combined += f"\n{text}"
        if match := DEAD_REFERENCE.search(text):
            errors.append(f"{path}: dead/private reference {match.group(0)!r}")

    for path in sorted(CLAUDE_DIR.rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        if match := DEAD_REFERENCE.search(text):
            errors.append(f"{path}: dead/private reference {match.group(0)!r}")

    main = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    if len(main.splitlines()) > 150:
        errors.append("SKILL.md exceeds the 150-line operational limit")
    required_description_terms = ("forge", "src/forge", "VLA", "distillation", "quantization")
    if any(term not in main for term in required_description_terms):
        errors.append("SKILL.md frontmatter description is missing required trigger terms")
    if "uv sync --locked --group dev" not in main:
        errors.append("SKILL.md does not use the locked development-group install command")
    if "four NVIDIA L4" not in main:
        errors.append("SKILL.md state of the world lacks the current GPU count")
    for target in re.findall(r"\[[^]]+\]\(([^)]+\.md)\)", main):
        if not (SKILL_DIR / target).is_file():
            errors.append(f"SKILL.md references missing file: {target}")

    for command_path in sorted(_forge_command_paths(combined)):
        command = [sys.executable, "-m", "forge.cli_v2", *command_path, "--help"]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=20, check=False)
        if completed.returncode != 0:
            errors.append(f"command help failed ({completed.returncode}): {' '.join(('forge', *command_path))}")
    return errors


def _public_doc_files() -> list[Path]:
    """Return launch-facing guides; versioned PRDs are planning records."""
    return sorted(PUBLIC_DOCS.glob("*.md"))


def _check_local_links(path: Path, markdown: str) -> list[str]:
    errors: list[str] = []
    for raw_target in MARKDOWN_LINK.findall(markdown):
        target = raw_target.strip().split("#", 1)[0]
        if not target or re.match(r"^[a-z][a-z0-9+.-]*:", target, re.IGNORECASE):
            continue
        resolved = (path.parent / target).resolve()
        if not resolved.exists():
            errors.append(f"{path}: missing local link target {raw_target!r}")
    return errors


def check_public_docs() -> list[str]:
    """Check public links, Python syntax, safe CLI examples, and generated help."""
    errors: list[str] = []
    files = _public_doc_files()
    if not files:
        return [f"no public markdown files found under {PUBLIC_DOCS}"]

    safe_help_paths: set[tuple[str, ...]] = set()
    for path in files:
        markdown = path.read_text(encoding="utf-8")
        errors.extend(_check_local_links(path, markdown))
        if match := DEAD_REFERENCE.search(markdown):
            errors.append(f"{path}: dead/private reference {match.group(0)!r}")

        for block in re.findall(r"```python\n(.*?)```", markdown, flags=re.DOTALL):
            try:
                compile(textwrap.dedent(block), f"{path}:python-block", "exec")
            except SyntaxError as exc:
                errors.append(f"{path}: invalid Python block: {exc.msg} (line {exc.lineno})")

        for block in re.findall(r"```(?:bash|shell)\n(.*?)```", markdown, flags=re.DOTALL):
            logical = block.replace("\\\n", " ")
            for raw_line in logical.splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    tokens = shlex.split(line)
                except ValueError as exc:
                    errors.append(f"{path}: invalid shell example {line!r}: {exc}")
                    continue
                if "forge" in tokens and "--help" in tokens:
                    index = tokens.index("forge")
                    safe_help_paths.add(tuple(tokens[index + 1 : tokens.index("--help")]))

    for command_path in sorted(safe_help_paths):
        command = [sys.executable, "-m", "forge.cli_v2", *command_path, "--help"]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=20, check=False)
        if completed.returncode != 0:
            errors.append(f"public doc help failed: {' '.join(('forge', *command_path))}")

    generated = subprocess.run(
        [sys.executable, "scripts/generate_cli_reference.py", "--check"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if generated.returncode != 0:
        errors.append(generated.stderr.strip() or generated.stdout.strip() or "CLI reference check failed")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skills", action="store_true", help="Validate the optional private FORGE skill")
    parser.add_argument("--docs", action="store_true", help="Validate launch-facing documentation")
    args = parser.parse_args()
    if not args.skills and not args.docs:
        parser.error("select --skills and/or --docs")
    errors: list[str] = []
    if args.skills:
        errors.extend(check_skill())
    if args.docs:
        errors.extend(check_public_docs())
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("FORGE documentation checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
