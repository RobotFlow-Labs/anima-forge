"""Release-gate tests for the dependency advisory exception policy."""

from __future__ import annotations

from datetime import date

from scripts.check_dependency_audit import validate_audit


def _audit(*findings: tuple[str, str]) -> dict:
    dependencies: dict[str, list[dict[str, str]]] = {}
    for package, advisory in findings:
        dependencies.setdefault(package, []).append({"id": advisory})
    return {
        "dependencies": [
            {"name": package, "version": "1.0", "vulns": vulnerabilities}
            for package, vulnerabilities in dependencies.items()
        ]
    }


def _allowlist(*exceptions: tuple[str, str, str]) -> dict:
    return {
        "exceptions": [
            {
                "package": package,
                "advisory": advisory,
                "expires": expiry,
                "rationale": "The affected entrypoint is unreachable pending an upstream-compatible fix.",
            }
            for package, advisory, expiry in exceptions
        ]
    }


def test_dependency_audit_accepts_current_exact_exception() -> None:
    errors = validate_audit(
        _audit(("torch", "CVE-2025-3000")),
        _allowlist(("torch", "CVE-2025-3000", "2026-08-13")),
        as_of=date(2026, 7, 13),
    )

    assert errors == []


def test_dependency_audit_rejects_unapproved_expired_and_stale_entries() -> None:
    errors = validate_audit(
        _audit(("torch", "NEW-ADVISORY"), ("diffusers", "KNOWN")),
        _allowlist(
            ("diffusers", "KNOWN", "2026-07-12"),
            ("unused", "OLD", "2026-08-13"),
        ),
        as_of=date(2026, 7, 13),
    )

    assert errors == [
        "unapproved advisory: torch NEW-ADVISORY",
        "stale exception: unused OLD",
        "expired exception: diffusers KNOWN (expired 2026-07-12)",
    ]
