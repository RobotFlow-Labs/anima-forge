#!/usr/bin/env python3
"""Fail unless every dependency advisory has a current, package-scoped exception."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any


def _finding_key(package: str, advisory: str) -> tuple[str, str]:
    return package.lower().replace("_", "-"), advisory.upper()


def validate_audit(
    audit: dict[str, Any],
    allowlist: dict[str, Any],
    *,
    as_of: date,
) -> list[str]:
    """Return validation errors for unexpected, expired, or stale exceptions."""
    findings = {
        _finding_key(str(dependency["name"]), str(vulnerability["id"]))
        for dependency in audit.get("dependencies", [])
        for vulnerability in dependency.get("vulns", [])
    }
    allowed: dict[tuple[str, str], tuple[date, str]] = {}
    errors: list[str] = []
    for exception in allowlist.get("exceptions", []):
        key = _finding_key(str(exception["package"]), str(exception["advisory"]))
        if key in allowed:
            errors.append(f"duplicate exception: {key[0]} {key[1]}")
            continue
        try:
            expiry = date.fromisoformat(str(exception["expires"]))
        except ValueError:
            errors.append(f"invalid expiry: {key[0]} {key[1]}")
            continue
        rationale = str(exception.get("rationale", "")).strip()
        if not rationale:
            errors.append(f"missing rationale: {key[0]} {key[1]}")
        allowed[key] = expiry, rationale

    for package, advisory in sorted(findings - allowed.keys()):
        errors.append(f"unapproved advisory: {package} {advisory}")
    for package, advisory in sorted(allowed.keys() - findings):
        errors.append(f"stale exception: {package} {advisory}")
    for key in sorted(findings & allowed.keys()):
        expiry, _rationale = allowed[key]
        if expiry < as_of:
            errors.append(f"expired exception: {key[0]} {key[1]} (expired {expiry.isoformat()})")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True, help="pip-audit JSON output")
    parser.add_argument("--allowlist", type=Path, required=True, help="package-scoped exception JSON")
    parser.add_argument("--as-of", type=date.fromisoformat, default=date.today())
    args = parser.parse_args()

    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    allowlist = json.loads(args.allowlist.read_text(encoding="utf-8"))
    errors = validate_audit(audit, allowlist, as_of=args.as_of)
    if errors:
        for error in errors:
            print(f"dependency audit error: {error}")
        return 1
    finding_count = len(
        {
            _finding_key(str(dependency["name"]), str(vulnerability["id"]))
            for dependency in audit.get("dependencies", [])
            for vulnerability in dependency.get("vulns", [])
        }
    )
    print(f"Dependency audit passed with {finding_count} current, package-scoped exceptions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
