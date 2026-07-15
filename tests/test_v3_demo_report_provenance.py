"""Visible provenance contracts for generated demo reports."""

from __future__ import annotations

from forge.demo.report import generate_html_report


def _report_data(provenance: dict[str, str] | None) -> dict:
    data = {
        "benchmark": {},
        "teachers": [],
        "embodiments": [],
        "architecture": {},
    }
    if provenance is not None:
        data["provenance"] = provenance
    return data


def test_demo_report_marks_missing_provenance_untrusted() -> None:
    html = generate_html_report(_report_data(None))

    assert "[MOCK — not a real model]" in html
    assert "Missing provenance" in html


def test_demo_report_names_mock_components() -> None:
    html = generate_html_report(_report_data({"vision": "mock", "language": "real", "labels": "mock"}))

    assert "[MOCK — not a real model]" in html
    assert "Mock provenance: vision, labels" in html


def test_demo_report_marks_all_real_provenance() -> None:
    html = generate_html_report(_report_data({"vision": "real", "language": "real", "labels": "real"}))

    assert "REAL provenance verified" in html
    assert "[MOCK — not a real model]" not in html
