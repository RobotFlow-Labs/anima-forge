"""Audit the tracked Git snapshot for private operator material and machine data."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

DISALLOWED_TRACKED_FILES = frozenset(
    {
        ".claude/settings.json",
        "BUILDING_PLAN.md",
        "CLAUDE.md",
        "FORGE_PIPELINE.md",
        "NEXT_STEPS.md",
        "PRD.md",
        "REPORT_GPU.md",
        "add-turboquant.md",
        "anima_module.yaml",
        "forge_upgrade_prod_plan.md",
        "manual_test_eval_plan.md",
        "setup-claude-skills.sh",
        "tmux.md",
    }
)
DISALLOWED_TRACKED_PREFIXES = (
    ".claude/",
    "docs/internal/",
    "docs/v2/",
    "docs/v3/",
    "marketing/benchmarks/",
    "marketing/metrics/",
    "old_docs/",
    "prds/",
    "reports/",
)
INLINE_SUPPRESSION_RULES: dict[str, frozenset[str]] = {
    "scripts/check_public_snapshot.py": frozenset({"private-unc-path"}),
    "scripts/run_full_gate_matrix.sh": frozenset({"private-unc-path"}),
    "src/forge/hub_package.py": frozenset({"private-unc-path"}),
}
INLINE_SUPPRESSION_PATTERN = re.compile(r"forge-public-audit:\s*allow\[([a-z0-9,-]+)\]")
PRIVATE_CONTENT_PATTERNS = (
    ("private-mount", re.compile("/" + "mnt/")),
    ("private-home", re.compile("/" + "home/")),
    ("private-user-home", re.compile("/" + "Users/")),
    (
        "private-windows-home",
        re.compile(r"\b[A-Za-z]:[\\/]Users[\\/]", re.IGNORECASE),  # forge-public-audit: allow[private-unc-path]
    ),
    ("private-unc-path", re.compile(r"\\\\[^\\\s]+[\\/]")),  # forge-public-audit: allow[private-unc-path]
    ("internal-ssh-alias", re.compile(r"\b" + "datai_" + r"srv\w*\b", re.IGNORECASE)),
    ("hugging-face-token", re.compile(r"\b" + "hf" + r"_[A-Za-z0-9]{20,}\b")),
    (
        "github-token",
        re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
    ),
    ("aws-access-key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    (
        "private-key",
        re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"),
    ),
    (
        "internal-operator-instruction",
        re.compile(
            "|".join(
                (
                    "RobotFlow-Labs/" + r"claude\.git",
                    "Restart " + "Claude Code",
                    "sync from " + "Mac",
                    "server-specific " + "settings",
                )
            ),
            re.IGNORECASE,
        ),
    ),
    (
        "private-ipv4",
        re.compile(
            r"(?<![\d.])(?:"
            r"10(?:\.\d{1,3}){3}|"
            r"192\.168(?:\.\d{1,3}){2}|"
            r"172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2}"
            r")(?![\d.])"
        ),
    ),
)


@dataclass(frozen=True, slots=True)
class AuditFinding:
    """One public-snapshot policy violation."""

    path: str
    rule: str
    line: int | None = None


def tracked_paths(root: Path) -> list[str]:
    """Return every path in the Git index, including tracked working-tree deletions."""
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return sorted(path.decode("utf-8") for path in result.stdout.split(b"\0") if path)


def untracked_paths(root: Path) -> list[str]:
    """Return nonignored worktree files that would enter a broad add."""
    result = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return sorted(path.decode("utf-8") for path in result.stdout.split(b"\0") if path)


def intended_production_paths(root: Path) -> list[str]:
    """Return the current publishable worktree after explicit archival removals."""
    candidates = set(tracked_paths(root)) | set(untracked_paths(root))
    return sorted(path for path in candidates if not _disallowed_path(path) and os.path.lexists(root / path))


def _disallowed_path(path: str) -> bool:
    return path in DISALLOWED_TRACKED_FILES or path.startswith(DISALLOWED_TRACKED_PREFIXES)


def _decode_text_lines(raw: bytes) -> list[str] | None:
    if b"\0" in raw:
        return None
    try:
        return raw.decode("utf-8").splitlines()
    except UnicodeDecodeError:
        return None


def _worktree_text_lines(root: Path, relative_path: str) -> list[str] | None:
    path = root / relative_path
    if path.is_symlink():
        try:
            return [os.readlink(path)]
        except OSError:
            return None
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    return _decode_text_lines(raw)


def _index_text_lines(root: Path, relative_path: str) -> list[str] | None:
    result = subprocess.run(
        ["git", "show", f":{relative_path}"],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        return None
    return _decode_text_lines(result.stdout)


def _inline_suppressed_rules(relative_path: str, line: str) -> frozenset[str]:
    """Return narrowly approved detector-source suppressions for one line."""
    approved = INLINE_SUPPRESSION_RULES.get(relative_path, frozenset())
    match = INLINE_SUPPRESSION_PATTERN.search(line)
    if match is None:
        return frozenset()
    requested = frozenset(part.strip() for part in match.group(1).split(",") if part.strip())
    return requested & approved


def _looks_like_package_version(line: str, match: re.Match[str]) -> bool:
    """Distinguish known dotted package versions from an address-shaped value."""
    prefix = line[: match.start()]
    if re.search(r"(?:\bversion(?:\s*=\s*['\"]?|\s+)|(?:==|>=|<=|~=|!=)\s*)$", prefix, re.IGNORECASE):
        return True
    if re.search(r"(?:tensorrt(?:[_-][A-Za-z0-9_]+)?|nvidia[_-][A-Za-z0-9_-]+)[_-]$", prefix, re.IGNORECASE):
        return True
    return bool(re.match(r"\s*\|\s*(?:TensorRT|CUDA|cuDNN|NVIDIA\b[^|]*)\s*\|", line, re.IGNORECASE))


def audit_paths(
    root: Path,
    paths: Iterable[str],
    *,
    load_lines: Callable[[Path, str], list[str] | None] = _worktree_text_lines,
) -> list[AuditFinding]:
    """Audit a supplied tracked-path inventory without consulting ignore rules."""
    findings: list[AuditFinding] = []
    for relative_path in sorted(set(paths)):
        if _disallowed_path(relative_path):
            findings.append(AuditFinding(relative_path, "nonproduction-tracked-file"))

        lines = load_lines(root, relative_path)
        if lines is None:
            continue
        for line_number, line in enumerate(lines, start=1):
            suppressed_rules = _inline_suppressed_rules(relative_path, line)
            for rule, pattern in PRIVATE_CONTENT_PATTERNS:
                if rule in suppressed_rules:
                    continue
                matches = list(pattern.finditer(line))
                if rule == "private-ipv4":
                    matches = [match for match in matches if not _looks_like_package_version(line, match)]
                if matches:
                    findings.append(AuditFinding(relative_path, rule, line_number))
    return findings


def audit_repository(root: Path) -> list[AuditFinding]:
    """Audit the paths and exact content currently staged in the Git index."""
    return audit_paths(root, tracked_paths(root), load_lines=_index_text_lines)


def audit_intended_repository(root: Path) -> list[AuditFinding]:
    """Audit current production files while final archival deletions remain unstaged."""
    return audit_paths(root, intended_production_paths(root))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Repository root")
    parser.add_argument("--json", action="store_true", help="Emit a machine-readable report")
    parser.add_argument(
        "--intended",
        action="store_true",
        help="Audit current production worktree files after explicit archival exclusions",
    )
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    findings = audit_intended_repository(root) if args.intended else audit_repository(root)
    if args.json:
        print(
            json.dumps(
                {
                    "schema": "forge.public-snapshot-audit.v1",
                    "scope": "intended-production" if args.intended else "git-index",
                    "status": "clean" if not findings else "failed",
                    "findings": [asdict(finding) for finding in findings],
                },
                indent=2,
                allow_nan=False,
            )
        )
    elif findings:
        for finding in findings:
            location = f"{finding.path}:{finding.line}" if finding.line is not None else finding.path
            print(f"{location}: {finding.rule}")
    else:
        print("Public snapshot audit passed")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
