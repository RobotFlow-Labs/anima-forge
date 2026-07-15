"""Public documentation and optional private FORGE skill checks."""

from pathlib import Path

import pytest
from scripts.check_docs import check_public_docs, check_skill


def test_optional_private_forge_skill_references_and_commands_are_current() -> None:
    assert check_skill() == []


def test_public_documentation_links_examples_and_cli_reference_are_current() -> None:
    assert check_public_docs() == []


def test_cli_reference_is_independent_of_terminal_width(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts.generate_cli_reference import OUTPUT, render

    monkeypatch.setenv("COLUMNS", "240")
    assert render() == OUTPUT.read_text(encoding="utf-8")


def test_quickstart_compression_and_export_keep_real_data_contract() -> None:
    quickstart = Path("docs/QUICKSTART.md").read_text(encoding="utf-8")

    assert quickstart.count("--data-dir /path/to/real-teacher-labels") >= 3
    assert "outputs/quickstart-compressed/compressed/qvla_4bit.pt" in quickstart
    assert "outputs/quickstart-compressed/compressed/pruned.pt" not in quickstart
    assert "teacher_labels/metadata.json" in quickstart


def test_pipeline_quantization_matrix_uses_one_pruned_source() -> None:
    pipeline = Path("docs/PIPELINE.md").read_text(encoding="utf-8")

    assert pipeline.count("forge quantize run --checkpoint outputs/compressed/pruned.pt") == 3
    assert "--method qvla --bits 8 --device cuda" in pipeline
    assert "--method turboquant-mse --bits 4 --device cuda" in pipeline
    assert "--method turboquant-mse --bits 8 --device cuda" in pipeline
    assert "requested_device: cuda" in pipeline
    assert "an empty\n`fallbacks` list" in pipeline
