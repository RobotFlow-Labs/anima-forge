"""Curated marketing release-kit contracts."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path

from PIL import Image


def _module():
    spec = importlib.util.spec_from_file_location("forge_release_kit", "scripts/build_release_kit.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _configure_claim_fixture(module, monkeypatch, tmp_path: Path, readme_text: str, claim_texts: list[str]) -> Path:
    root = tmp_path / "repository"
    docs = root / "docs"
    evidence = root / "evidence" / "benchmark.json"
    docs.mkdir(parents=True)
    evidence.parent.mkdir(parents=True)
    readme = root / "README.md"
    readme.write_text(readme_text, encoding="utf-8")
    evidence.write_text('{"status": "completed"}\n', encoding="utf-8")
    digest = hashlib.sha256(evidence.read_bytes()).hexdigest()
    registry = docs / "public_claims.json"
    registry.write_text(
        json.dumps(
            {
                "schema": "forge.public-claim-registry.v1",
                "claims": [
                    {
                        "claim_id": f"fixture-claim-{index}",
                        "document": "README.md",
                        "claim_text": claim_text,
                        "source_artifact": "evidence/benchmark.json",
                        "source_artifact_sha256": digest,
                        "source_revision": "a" * 40,
                    }
                    for index, claim_text in enumerate(claim_texts, start=1)
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "ROOT", root)
    monkeypatch.setattr(module, "README", readme)
    monkeypatch.setattr(module, "PUBLIC_DOCS", docs)
    monkeypatch.setattr(module, "CLAIM_REGISTRY", registry)
    return evidence


def test_release_kit_withholds_invalidated_public_claims() -> None:
    module = _module()
    outputs = module.render_release_kit()
    manifest = json.loads(outputs["launch_manifest.json"])

    assert manifest["schema"] == "forge.release-kit.v1"
    assert manifest["version"] == "3.0.1"
    assert manifest["completed_variants"] == []
    assert "corrected-preprocessing" in manifest["release_validation"]
    assert "Every variant" in manifest["pending_claim_policy"]
    assert "no unvalidated performance claim" in outputs["release_notes.md"]


def test_release_kit_contains_no_private_machine_patterns() -> None:
    module = _module()

    for content in module.render_release_kit().values():
        assert module.PRIVATE_PATTERN.search(content) is None


def test_release_kit_files_are_current() -> None:
    module = _module()

    assert module.write_release_kit(check=True) == []
    assert (Path("marketing") / "release" / "release_notes.md").is_file()


def test_release_kit_renders_verified_readme_measurements(monkeypatch, tmp_path: Path) -> None:
    """Final validated rows flow into every public launch asset without hand editing."""
    module = _module()
    readme = tmp_path / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# FORGE",
                "",
                "| Completed variant | Real training loss reduction | Packed INT4 | ONNX CUDA | TensorRT fp16 |",
                "|---|---:|---:|---:|---:|",
                "| nano | 41.2% | 4.84x | 21.3 FPS | 56.1 FPS |",
                "| micro | 38.0% | 4.10x | 17.0 FPS | 44.2 FPS |",
                "",
            ]
        ),
        encoding="utf-8",
    )
    readme_text = readme.read_text(encoding="utf-8")
    _configure_claim_fixture(
        module,
        monkeypatch,
        tmp_path,
        readme_text,
        ["41.2%", "4.84x", "21.3 FPS", "56.1 FPS", "38.0%", "4.10x", "17.0 FPS", "44.2 FPS"],
    )

    outputs = module.render_release_kit()
    manifest = json.loads(outputs["launch_manifest.json"])

    assert manifest["release_validation"] == "launch-week NVIDIA L4 measurements"
    assert [row["variant"] for row in manifest["completed_variants"]] == ["nano", "micro"]
    assert manifest["completed_variants"][0]["packed_int4"] == "4.84x"
    assert "**nano** — 41.2% loss reduction" in outputs["release_notes.md"]
    assert "FORGE v3.0.1 is ready" in outputs["social_copy.md"]


def test_release_kit_rejects_malformed_measurement_rows(monkeypatch, tmp_path: Path) -> None:
    """A partial README row cannot silently become a public launch claim."""
    module = _module()
    readme = tmp_path / "README.md"
    readme.write_text(
        "\n".join(
            [
                "| Completed variant | Real training loss reduction | Packed INT4 | ONNX CUDA | TensorRT fp16 |",
                "|---|---:|---:|---:|---:|",
                "| nano | 41.2% | 4.84x | missing TensorRT |",
            ]
        ),
        encoding="utf-8",
    )
    readme_text = readme.read_text(encoding="utf-8")
    _configure_claim_fixture(module, monkeypatch, tmp_path, readme_text, ["41.2%", "4.84x"])

    try:
        module.render_release_kit()
    except ValueError as exc:
        assert "Malformed launch benchmark row" in str(exc)
    else:
        raise AssertionError("Malformed launch benchmark row was accepted")


def test_public_claim_gate_scans_readme_and_every_curated_doc(monkeypatch, tmp_path: Path) -> None:
    module = _module()
    root = tmp_path / "repository"
    docs = root / "docs"
    docs.mkdir(parents=True)
    readme = root / "README.md"
    readme.write_text("Measured throughput is 10 FPS.\n", encoding="utf-8")
    (docs / "A.md").write_text("The optimized runtime is 2x faster.\n", encoding="utf-8")
    (docs / "B.md").write_text("Measured latency is 20 ms.\n", encoding="utf-8")
    registry = docs / "public_claims.json"
    registry.write_text('{"schema":"forge.public-claim-registry.v1","claims":[]}\n', encoding="utf-8")
    monkeypatch.setattr(module, "ROOT", root)
    monkeypatch.setattr(module, "README", readme)
    monkeypatch.setattr(module, "PUBLIC_DOCS", docs)
    monkeypatch.setattr(module, "CLAIM_REGISTRY", registry)

    findings = module.public_claim_findings()

    assert any(finding.startswith("README.md:1:") for finding in findings)
    assert any(finding.startswith("docs/A.md:1:") for finding in findings)
    assert any(finding.startswith("docs/B.md:1:") for finding in findings)
    assert any("faster" in finding for finding in findings)


def test_public_claim_gate_catches_qualitative_superlatives_and_comparisons(monkeypatch, tmp_path: Path) -> None:
    module = _module()
    root = tmp_path / "repository"
    docs = root / "docs"
    docs.mkdir(parents=True)
    readme = root / "README.md"
    readme.write_text("# Product\n", encoding="utf-8")
    (docs / "CLAIMS.md").write_text(
        "\n".join(
            (
                "The learned adapter has best accuracy.",
                "Use the optimized inference engine.",
                "This produces higher quality and lower latency.",
                "The new runtime is more efficient and more accurate.",
                "It recommends optimal configs from the benchmark.",
            )
        ),
        encoding="utf-8",
    )
    registry = docs / "public_claims.json"
    registry.write_text('{"schema":"forge.public-claim-registry.v1","claims":[]}\n', encoding="utf-8")
    monkeypatch.setattr(module, "ROOT", root)
    monkeypatch.setattr(module, "README", readme)
    monkeypatch.setattr(module, "PUBLIC_DOCS", docs)
    monkeypatch.setattr(module, "CLAIM_REGISTRY", registry)

    findings = "\n".join(module.public_claim_findings()).lower()

    for phrase in (
        "best accuracy",
        "optimized inference",
        "higher quality",
        "lower latency",
        "more efficient",
        "more accurate",
        "optimal configs",
    ):
        assert phrase in findings


def test_public_claim_gate_ignores_identifiers_and_negated_governance_prose(monkeypatch, tmp_path: Path) -> None:
    module = _module()
    root = tmp_path / "repository"
    docs = root / "docs"
    docs.mkdir(parents=True)
    readme = root / "README.md"
    readme.write_text("# Product\n", encoding="utf-8")
    (docs / "SAFE.md").write_text(
        "\n".join(
            (
                "`optimized_inference_engine` is the API identifier.",
                "`higher_throughput` is a result field.",
                "Best accuracy is not claimed pending validation.",
                "No performance claim is made that this path is more efficient.",
            )
        ),
        encoding="utf-8",
    )
    registry = docs / "public_claims.json"
    registry.write_text('{"schema":"forge.public-claim-registry.v1","claims":[]}\n', encoding="utf-8")
    monkeypatch.setattr(module, "ROOT", root)
    monkeypatch.setattr(module, "README", readme)
    monkeypatch.setattr(module, "PUBLIC_DOCS", docs)
    monkeypatch.setattr(module, "CLAIM_REGISTRY", registry)

    assert module.public_claim_findings() == []


def test_public_claim_registry_rejects_tampered_source_artifact(monkeypatch, tmp_path: Path) -> None:
    module = _module()
    evidence = _configure_claim_fixture(
        module,
        monkeypatch,
        tmp_path,
        "Measured throughput is 10 FPS.\n",
        ["10 FPS"],
    )
    evidence.write_text('{"status": "tampered"}\n', encoding="utf-8")

    try:
        module.public_claim_findings()
    except ValueError as exc:
        assert "source artifact SHA-256 does not match" in str(exc)
    else:
        raise AssertionError("A tampered public-claim source artifact was accepted")


def test_public_claim_registry_requires_full_source_revision(monkeypatch, tmp_path: Path) -> None:
    module = _module()
    _configure_claim_fixture(
        module,
        monkeypatch,
        tmp_path,
        "Measured throughput is 10 FPS.\n",
        ["10 FPS"],
    )
    registry = json.loads(module.CLAIM_REGISTRY.read_text(encoding="utf-8"))
    registry["claims"][0]["source_revision"] = "short"
    module.CLAIM_REGISTRY.write_text(json.dumps(registry), encoding="utf-8")

    try:
        module.public_claim_findings()
    except ValueError as exc:
        assert "source_revision must be a full Git SHA" in str(exc)
    else:
        raise AssertionError("A truncated public-claim source revision was accepted")


def test_public_hero_and_intro_withhold_unvalidated_absolute_claims() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    public_intro = "\n".join(readme.splitlines()[:60]).lower()
    project = Path("pyproject.toml").read_text(encoding="utf-8").lower()
    pipeline = Path("docs/PIPELINE.md").read_text(encoding="utf-8").lower()

    for unsupported in ("distill any vla", "7b+ vision-language-action", "<2gb real-time", "8x weight"):
        assert unsupported not in public_intro
        assert unsupported not in project
        assert unsupported not in pipeline
    assert "keeps the behavior" not in public_intro
    assert "assets/hero-v2.png" in readme
    assert 'src="assets/hero.png"' not in readme
    assert re.findall(r"\]\((?!https://|#|mailto:)([^)]+)\)", readme) == []
    assert re.findall(r'(?:src|href)="(?!https://|#|mailto:)([^"]+)"', readme) == []

    with Image.open("assets/hero-v2.png") as hero:
        hero.verify()
        assert hero.width >= 1280
        assert hero.width / hero.height >= 1.9
        assert hero.mode == "RGB"
