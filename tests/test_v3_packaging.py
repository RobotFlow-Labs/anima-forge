"""PRD-40 package metadata and resource contracts."""

from __future__ import annotations

import os
import re
import subprocess
import tarfile
import tomllib
from importlib.resources import files
from pathlib import Path

from fastapi.testclient import TestClient

from forge import __version__
from forge.web.api import create_app


def test_v3_version_and_dashboard_package_resource() -> None:
    assert __version__ == "3.0.1"
    dashboard = files("forge.web").joinpath("dashboard.html")
    assert dashboard.is_file()
    assert "FORGE" in dashboard.read_text(encoding="utf-8")

    response = TestClient(create_app()).get("/")
    assert response.status_code == 200
    assert "FORGE" in response.text


def test_complete_runtime_is_part_of_the_base_python_312_install() -> None:
    document = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    metadata = document["project"]
    dependencies = metadata["dependencies"]

    assert metadata["requires-python"] == ">=3.12,<3.13"
    assert metadata["description"] == (
        "Provenance-enforced VLA teacher labeling, student distillation, compression, export, "
        "and evaluation for edge robotics."
    )
    assert "torch>=2.10,<2.11" in dependencies
    assert "torchvision>=0.25,<0.26" in dependencies
    assert "typer>=0.26.8,<0.27" in dependencies
    assert any(dependency.startswith("lerobot[") for dependency in dependencies)
    assert any(dependency.startswith("av>=") for dependency in dependencies)
    assert any(dependency.startswith("onnxruntime-gpu>=") for dependency in dependencies)
    assert any(dependency.startswith("onnxruntime>=") for dependency in dependencies)
    assert "optional-dependencies" not in metadata
    assert "websockets>=16,<17" in dependencies
    assert "msgpack>=1.2.1,<2" in dependencies
    assert "wandb>=0.25,<1" in dependencies
    assert "bitsandbytes>=0.49,<1; platform_system == 'Linux'" in dependencies
    assert "tensorrt==10.16.0.72; platform_system == 'Linux'" in dependencies
    assert "mlx>=0.16.0; platform_system == 'Darwin' and platform_machine == 'arm64'" in dependencies
    assert "mlx-lm>=0.18.0; platform_system == 'Darwin' and platform_machine == 'arm64'" in dependencies
    assert "coremltools>=7.2; platform_system == 'Darwin'" in dependencies
    assert {
        "aiohttp>=3.14,<4",
        "click>=8.4.2,<8.5",
        "fastapi>=0.139,<1",
        "gitpython>=3.1.50,<4",
        "idna>=3.15,<4",
        "mako>=1.3.12,<2",
        "msgpack>=1.2.1,<2",
        "onnx>=1.22,<2",
        "pillow>=12.3,<13",
        "pygments>=2.20,<3",
        "python-multipart>=0.0.31,<1",
        "requests>=2.33,<3",
        "starlette>=1.3.1,<2",
        "urllib3>=2.7,<3",
    }.issubset(dependencies)
    assert "dev" in document["dependency-groups"]


def test_hosted_release_jobs_reclaim_space_for_complete_runtime() -> None:
    for workflow in (".github/workflows/ci.yaml", ".github/workflows/release.yml"):
        source = Path(workflow).read_text(encoding="utf-8")
        assert "Reclaim runner disk for the complete ML runtime" in source
        assert "/usr/local/lib/android" in source
        assert "/usr/share/dotnet" in source
        assert "scripts/generate_cli_reference.py --check" in source


def test_platform_installers_are_executed_on_real_hosted_runners() -> None:
    workflow = Path(".github/workflows/ci.yaml").read_text(encoding="utf-8")
    assert "runs-on: macos-14" in workflow
    assert "./install.sh --cpu --from-wheel" in workflow
    assert "import coremltools" in workflow
    assert "import mlx" in workflow
    assert "MLXTurboQuantizer" in workflow
    assert "np.isfinite(np.asarray(quantized)).all()" in workflow
    assert "runs-on: windows-2022" in workflow
    assert ".\\install.ps1 -Device cpu -FromWheel" in workflow
    assert "import onnxruntime" in workflow


def test_sdist_contains_only_package_source_and_public_metadata(tmp_path: Path) -> None:
    subprocess.run(
        ["uv", "build", "--sdist", "--out-dir", str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    archive = next(tmp_path.glob("anima_forge-*.tar.gz"))
    with tarfile.open(archive, "r:gz") as distribution:
        relative_paths = [Path(*Path(member.name).parts[1:]) for member in distribution.getmembers()]
        build_info_member = next(
            member for member in distribution.getmembers() if member.name.endswith("src/forge/_build_info.py")
        )
        build_info_file = distribution.extractfile(build_info_member)
        assert build_info_file is not None
        build_info = build_info_file.read().decode("utf-8")

    top_level = {path.parts[0] for path in relative_paths if path.parts}
    assert top_level == {
        ".gitignore",
        "CHANGELOG.md",
        "CITATION.cff",
        "COMPATIBILITY.md",
        "hatch_build.py",
        "LICENSE",
        "PKG-INFO",
        "README.md",
        "pyproject.toml",
        "src",
    }
    assert any(path == Path("src/forge/__init__.py") for path in relative_paths)
    expected_revision = os.environ.get("FORGE_GIT_SHA", "").strip().lower()
    if re.fullmatch(r"[0-9a-f]{40}", expected_revision) is None:
        expected_revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    assert re.search(rf'^SOURCE_GIT_SHA = "{expected_revision}"$', build_info, re.M)


def test_docker_targets_use_supported_runtime_and_private_context_boundary() -> None:
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")
    dockerignore = Path(".dockerignore").read_text(encoding="utf-8")

    assert "--extra dev" not in dockerfile
    assert "--extra cuda" not in dockerfile
    assert dockerfile.count("uv sync --locked --group dev") == 4
    assert dockerfile.count("--no-install-project") == 2
    assert dockerfile.count("--mount=type=cache,target=/root/.cache/uv") == 2
    assert dockerfile.count("ARG FORGE_GIT_SHA") == 2
    assert dockerfile.count("ENV FORGE_GIT_SHA=${FORGE_GIT_SHA}") == 2
    assert dockerfile.count("^[0-9a-f]{40}$") == 2
    assert "COPY Dockerfile Dockerfile" in dockerfile
    assert dockerfile.count("COPY pyproject.toml README.md uv.lock hatch_build.py ./") == 2
    assert compose.count("FORGE_GIT_SHA: ${FORGE_GIT_SHA:-unknown}") == 2
    assert "nvidia/cuda:12.8.1-devel-ubuntu22.04 AS cuda" in dockerfile
    assert 'ENTRYPOINT ["uv", "run", "forge"]' in dockerfile
    assert "forge.runtime" not in dockerfile
    assert " AS jetson" not in dockerfile
    assert "target: jetson" not in compose

    assert "!docs/*.md" in dockerignore
    assert "!hatch_build.py" in dockerignore
    assert "!docs/**" not in dockerignore
    assert "!scripts/**" not in dockerignore
    assert "scripts/gpu_real_ops.sh" not in dockerignore
    assert "/docs/**" in dockerignore
    assert "/scripts/**" in dockerignore
    assert "/.claude/**" in dockerignore
    assert "**/__pycache__/" in dockerignore
    assert "!/.github/workflows/release.yml" in dockerignore
    assert "!/marketing/release/launch_manifest.json" in dockerignore
    assert "!/benchmarks/.gitkeep" in dockerignore
