"""Public repository community-file contracts."""

from __future__ import annotations

from pathlib import Path

import yaml


def test_community_and_github_templates_are_present_and_parseable() -> None:
    for filename in (
        "LICENSE",
        "CONTRIBUTING.md",
        "CODE_OF_CONDUCT.md",
        "SECURITY.md",
        "CITATION.cff",
        ".github/pull_request_template.md",
    ):
        assert Path(filename).is_file()

    license_text = Path("LICENSE").read_text(encoding="utf-8")
    assert "Apache License" in license_text
    assert "Version 2.0, January 2004" in license_text
    assert "APPENDIX: How to apply the Apache License" in license_text

    for filename in (
        "CITATION.cff",
        ".github/dependabot.yml",
        ".github/ISSUE_TEMPLATE/bug.yml",
        ".github/ISSUE_TEMPLATE/feature.yml",
    ):
        value = yaml.safe_load(Path(filename).read_text(encoding="utf-8"))
        assert isinstance(value, dict) and value
