# Security Policy

## Supported versions

Security fixes are provided for the latest 3.x release line.

## Reporting a vulnerability

Use GitHub's **Report a vulnerability** private advisory flow for this repository. Do not
open a public issue and do not include live credentials in logs or reproductions.

Include the affected version/commit, impact, reproduction steps, and any proposed
mitigation. We aim to acknowledge reports within three business days and will coordinate
disclosure after a fix is available.

Model quality, benchmark discrepancies, and unsupported hardware are regular bug reports
unless they create a concrete security boundary failure.

## Dependency audit policy

CI and the publishing workflow audit the locked production dependency graph. Findings are
release-blocking unless they have an exact package-and-advisory entry in
`.github/dependency-audit-allowlist.json`. Every exception documents reachability and an
expiry date; expired or stale entries fail the gate. This keeps temporary upstream
constraints visible instead of silently suppressing scanner output.
