"""Build the curated public launch kit from verified repository claims."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).parents[1]
README = ROOT / "README.md"
PUBLIC_DOCS = ROOT / "docs"
CLAIM_REGISTRY = PUBLIC_DOCS / "public_claims.json"
VERSION_FILE = ROOT / "src" / "forge" / "__init__.py"
OUTPUT_DIR = ROOT / "marketing" / "release"
PRIVATE_PATTERN = re.compile(
    "|".join(("/" + "mnt/", "/" + "home/", "datai_" + "srv", "ss" + r"h\s", "hf" + r"_[A-Za-z0-9]"))
)
CLAIM_SCHEMA = "forge.public-claim-registry.v1"
CLAIM_FIELDS = frozenset(
    {
        "claim_id",
        "document",
        "claim_text",
        "source_artifact",
        "source_artifact_sha256",
        "source_revision",
    }
)
NUMERIC_PERFORMANCE = re.compile(
    r"(?<![\w.])(?:[<>~≈]\s*)?\d+(?:\.\d+)?\s*"
    r"(?P<unit>milliseconds?|msecs?|ms|seconds?|secs?|minutes?|mins?|hours?|hrs?|"
    r"GiB|GB|MiB|MB|frames?/s|FPS|steps?/s|tokens?/s|[x×]|%)(?!\w)",
    re.IGNORECASE,
)
COMPARATIVE_PERFORMANCE = re.compile(
    r"\b(?:faster|slower|outperform(?:s|ed|ing)?|speedup|real[- ]time|"
    r"(?:higher|lower|better)\s+(?:quality|accuracy|latency|throughput)|"
    r"more\s+(?:efficient|accurate)|less\s+(?:efficient|accurate)|"
    r"best\s+(?:accuracy|quality|latency|throughput)|most\s+(?:accurate|efficient)|fastest|"
    r"optimized\s+(?:inference|runtime|execution|performance)|optimal\s+configs?|"
    r"less memory|memory[- ]efficient|memory efficiency|superior performance)\b",
    re.IGNORECASE,
)
PERCENT_CONTEXT = re.compile(
    r"\b(?:loss|reduction|improv(?:ement|ed)?|success|accuracy|quality|latency|throughput|"
    r"memory|compression|utilization|faster|slower|speedup)\b",
    re.IGNORECASE,
)
NON_CLAIM_SIZE_CONTEXT = re.compile(
    r"\b(?:workspace|rotate logs?|log files?|GPU compatibility|A100|RTX|Jetson|L4|T4)\b",
    re.IGNORECASE,
)
NON_CLAIM_DURATION_CONTEXT = re.compile(r"\b(?:timeout|default|duration|warmup|window|interval)\b", re.IGNORECASE)
NEGATED_COMPARATIVE_CONTEXT = re.compile(
    r"\b(?:not a performance benchmark|not comparable|claims? remain unpublished|"
    r"no (?:public |performance )?claim|does not claim|not (?:claimed|validated|measured|supported)|"
    r"pending validation|validation is in progress|(?:claims?|measurements?|results?) (?:are )?withheld)\b",
    re.IGNORECASE,
)


def _production_markdown_files() -> list[tuple[str, Path]]:
    files = [("README.md", README)]
    files.extend((path.relative_to(ROOT).as_posix(), path) for path in sorted(PUBLIC_DOCS.glob("*.md")))
    return files


def _canonical_relative_path(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError(f"claim registry {field} must be a canonical relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"claim registry {field} must be a canonical relative path")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_claim_registry(documents: dict[str, str]) -> list[dict[str, str]]:
    try:
        registry = json.loads(CLAIM_REGISTRY.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read public claim registry {CLAIM_REGISTRY}: {exc}") from exc
    if not isinstance(registry, dict) or registry.get("schema") != CLAIM_SCHEMA:
        raise ValueError(f"Public claim registry must use schema {CLAIM_SCHEMA}")
    claims = registry.get("claims")
    if not isinstance(claims, list):
        raise ValueError("Public claim registry must contain a claims list")

    validated: list[dict[str, str]] = []
    claim_ids: set[str] = set()
    locations: set[tuple[str, str]] = set()
    for index, raw_claim in enumerate(claims):
        if not isinstance(raw_claim, dict) or set(raw_claim) != CLAIM_FIELDS:
            raise ValueError(f"Public claim registry entry {index} must contain exactly {sorted(CLAIM_FIELDS)}")
        if not all(isinstance(raw_claim[field], str) and raw_claim[field] for field in CLAIM_FIELDS):
            raise ValueError(f"Public claim registry entry {index} contains an empty or non-string field")
        claim = {field: raw_claim[field] for field in CLAIM_FIELDS}
        claim_id = claim["claim_id"]
        if re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", claim_id) is None or claim_id in claim_ids:
            raise ValueError(f"Public claim registry entry {index} has an invalid or duplicate claim_id")
        claim_ids.add(claim_id)

        document = _canonical_relative_path(claim["document"], field="document")
        if document not in documents or claim["claim_text"] not in documents[document]:
            raise ValueError(f"Registered claim {claim_id} does not match retained document {document}")
        location = (document, claim["claim_text"])
        if location in locations:
            raise ValueError(f"Duplicate registered claim text for {document}: {claim['claim_text']!r}")
        locations.add(location)

        artifact_name = _canonical_relative_path(claim["source_artifact"], field="source_artifact")
        artifact = (ROOT / artifact_name).resolve()
        try:
            artifact.relative_to(ROOT.resolve())
        except ValueError as exc:
            raise ValueError(f"Registered claim {claim_id} source artifact escapes the repository") from exc
        if not artifact.is_file():
            raise ValueError(f"Registered claim {claim_id} source artifact is missing: {artifact_name}")
        artifact_sha256 = claim["source_artifact_sha256"]
        if re.fullmatch(r"[0-9a-f]{64}", artifact_sha256) is None or _sha256(artifact) != artifact_sha256:
            raise ValueError(f"Registered claim {claim_id} source artifact SHA-256 does not match")
        if re.fullmatch(r"[0-9a-f]{40}", claim["source_revision"]) is None:
            raise ValueError(f"Registered claim {claim_id} source_revision must be a full Git SHA")
        validated.append(claim)
    return validated


def _numeric_match_is_claim(match: re.Match[str], context: str) -> bool:
    unit = match.group("unit").lower()
    if unit == "%":
        return PERCENT_CONTEXT.search(context) is not None
    if unit in {"gb", "gib", "mb", "mib"}:
        return NON_CLAIM_SIZE_CONTEXT.search(context) is None
    if unit in {"seconds", "second", "secs", "sec", "minutes", "minute", "mins", "min", "hours", "hour", "hrs", "hr"}:
        return NON_CLAIM_DURATION_CONTEXT.search(context) is None
    return True


def _comparative_match_is_claim(match: re.Match[str], context: str) -> bool:
    candidate = match.group(0).lower().replace(" ", "-")
    if "real-time" in candidate and re.search(
        r"\b(?:monitor|metrics?|stream(?:ing)?|WebSocket|manager)\b", context, re.IGNORECASE
    ):
        return False
    return NEGATED_COMPARATIVE_CONTEXT.search(context) is None


def public_claim_findings() -> list[str]:
    """Return unregistered performance claims across every retained public guide."""
    files = _production_markdown_files()
    documents = {name: path.read_text(encoding="utf-8") for name, path in files}
    registered = _load_claim_registry(documents)
    registrations = {(claim["document"], claim["claim_text"]) for claim in registered}
    findings: list[str] = []

    for document, markdown in documents.items():
        pending_table_header = ""
        active_table_header = ""
        for line_number, line in enumerate(markdown.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                pending_table_header = ""
                active_table_header = ""
            elif stripped.startswith("|"):
                if re.fullmatch(r"\|?[\s:|-]+", stripped):
                    active_table_header = pending_table_header
                elif not active_table_header:
                    pending_table_header = line
            else:
                pending_table_header = ""
                active_table_header = ""
            context = f"{active_table_header} {line}" if active_table_header else line
            candidates = [
                match.group(0).strip()
                for match in NUMERIC_PERFORMANCE.finditer(line)
                if _numeric_match_is_claim(match, context)
            ]
            candidates.extend(
                match.group(0)
                for match in COMPARATIVE_PERFORMANCE.finditer(line)
                if _comparative_match_is_claim(match, context)
            )
            for candidate in dict.fromkeys(candidates):
                if (document, candidate) not in registrations:
                    findings.append(f"{document}:{line_number}: unregistered public performance claim {candidate!r}")
    return findings


def _version() -> str:
    match = re.search(r'^__version__\s*=\s*"([^"]+)"', VERSION_FILE.read_text(encoding="utf-8"), re.MULTILINE)
    if match is None:
        raise ValueError("Could not read the FORGE version")
    return match.group(1)


def _launch_metrics() -> list[dict[str, str]]:
    markdown = README.read_text(encoding="utf-8")
    marker = "| Completed variant | Real training loss reduction | Packed INT4 | ONNX CUDA | TensorRT fp16 |"
    try:
        table = markdown.split(marker, 1)[1]
    except IndexError as exc:
        raise ValueError("README launch benchmark table was not found") from exc
    rows: list[dict[str, str]] = []
    for line in table.splitlines()[2:]:
        if not line.startswith("|"):
            break
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != 5:
            raise ValueError(f"Malformed launch benchmark row: {line}")
        rows.append(
            {
                "variant": cells[0],
                "training_loss_reduction": cells[1],
                "packed_int4": cells[2],
                "onnx_cuda": cells[3],
                "tensorrt_fp16": cells[4],
            }
        )
    return rows


def render_release_kit() -> dict[str, str]:
    """Return every generated release-kit file keyed by relative output name."""
    findings = public_claim_findings()
    if findings:
        raise ValueError("Public claim governance failed:\n" + "\n".join(findings))
    version = _version()
    metrics = _launch_metrics()
    manifest = {
        "schema": "forge.release-kit.v1",
        "version": version,
        "claim_source": "README.md#how-it-works",
        "release_validation": (
            "launch-week NVIDIA L4 measurements" if metrics else "pending corrected-preprocessing validation"
        ),
        "completed_variants": metrics,
        "pending_claim_policy": "Every variant is omitted until corrected-preprocessing validation completes.",
    }
    metric_lines = [
        (
            f"- **{row['variant']}** — {row['training_loss_reduction']} loss reduction; "
            f"{row['packed_int4']} packed INT4; {row['onnx_cuda']} ONNX CUDA; "
            f"{row['tensorrt_fp16']} TensorRT fp16."
        )
        for row in metrics
    ]
    measurement_lines = (
        metric_lines
        if metric_lines
        else [
            "No launch measurements are published yet. Corrected-preprocessing training and",
            "artifact validation must complete before this section gains any result rows.",
        ]
    )
    release_notes = "\n".join(
        [
            f"# FORGE v{version} release notes",
            "",
            "FORGE distills large vision-language-action teachers into compact students for edge robotics.",
            "This launch ships the truthful CLI, mandatory teacher runtimes, trained-checkpoint provenance,",
            "chunk-aware compression, ONNX/TensorRT/MLX export, and clean Python 3.12 packaging.",
            "",
            "## Verified launch measurements",
            "",
            *measurement_lines,
            "",
            (
                "All numbers above are launch-week NVIDIA L4 measurements copied from the public README."
                if metric_lines
                else "The release kit deliberately carries no unvalidated performance claim."
            ),
            "Unfinished variants are deliberately omitted.",
            "",
            "## Start",
            "",
            "```sh",
            "curl -fsSL https://raw.githubusercontent.com/RobotFlow-Labs/"
            "anima-forge/main/install.sh | sh",
            "forge doctor",
            "forge quickstart --yes",
            "```",
            "",
        ]
    )
    social = "\n".join(
        [
            "# Launch copy",
            "",
            "## Short",
            "",
            (
                f"FORGE v{version} is ready: distill 7B+ VLA teachers into compact edge students with one truthful CLI."
                if metric_lines
                else f"FORGE v{version} release validation is in progress; no performance claim is public yet."
            ),
            (
                "Fresh NVIDIA L4 measurements, mandatory teacher integrations, "
                "ONNX/TensorRT/MLX export, and a one-line installer."
                if metric_lines
                else "Mandatory teacher, export, installer, and corrected NVIDIA L4 gates must all pass before launch."
            ),
            "",
            "## Technical",
            "",
            f"FORGE v{version} packages multi-teacher VLA distillation, provenance-enforced "
            "training, chunk-aware pruning/quantization,",
            "and runtime export for edge robotics. The public launch table contains only "
            "freshly completed NVIDIA L4 runs;",
            "pending variants stay unpublished until their artifacts pass validation.",
            "",
            "## Links",
            "",
            "- Repository: https://github.com/RobotFlow-Labs/anima-forge",
            "- Install: `curl -fsSL https://raw.githubusercontent.com/RobotFlow-Labs/"
            "anima-forge/main/install.sh | sh`",
            "",
        ]
    )
    outputs = {
        "launch_manifest.json": json.dumps(manifest, indent=2, allow_nan=False) + "\n",
        "release_notes.md": release_notes,
        "social_copy.md": social,
    }
    for name, content in outputs.items():
        if PRIVATE_PATTERN.search(content):
            raise ValueError(f"Generated release asset {name} contains a private-path or token pattern")
    return outputs


def write_release_kit(*, check: bool) -> list[str]:
    """Write the release kit, or return drift errors in check mode."""
    outputs = render_release_kit()
    errors: list[str] = []
    for name, content in outputs.items():
        path = OUTPUT_DIR / name
        if check:
            if not path.is_file() or path.read_text(encoding="utf-8") != content:
                errors.append(f"{path.relative_to(ROOT)} is stale; run scripts/build_release_kit.py")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail when generated release assets are stale")
    args = parser.parse_args()
    try:
        errors = write_release_kit(check=args.check)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("FORGE release kit is current" if args.check else f"wrote release kit to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
