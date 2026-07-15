"""Strict TestPyPI release-artifact verification contracts."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest


def _load_module():
    spec = importlib.util.spec_from_file_location("forge_verify_testpypi", "scripts/verify_testpypi_artifact.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _distributions(tmp_path: Path, *, sdist_version: str = "3.0.0") -> tuple[Path, bytes, Path, bytes]:
    dist = tmp_path / "dist"
    dist.mkdir()
    wheel = dist / "anima_forge-3.0.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("anima_forge-3.0.0.dist-info/METADATA", "Name: anima-forge\nVersion: 3.0.0\n")
        archive.writestr("forge/__init__.py", '__version__ = "3.0.0"\n')

    sdist = dist / "anima_forge-3.0.0.tar.gz"
    metadata = f"Name: anima-forge\nVersion: {sdist_version}\n".encode()
    with tarfile.open(sdist, "w:gz") as archive:
        info = tarfile.TarInfo("anima_forge-3.0.0/PKG-INFO")
        info.size = len(metadata)
        archive.addfile(info, io.BytesIO(metadata))
    return wheel, wheel.read_bytes(), sdist, sdist.read_bytes()


def _record(path: Path, content: bytes, package_type: str) -> dict[str, object]:
    return {
        "filename": path.name,
        "packagetype": package_type,
        "yanked": False,
        "digests": {"sha256": hashlib.sha256(content).hexdigest()},
        "url": f"https://test-files.pythonhosted.org/packages/{path.name}",
    }


def _payload(wheel: Path, wheel_bytes: bytes, sdist: Path, sdist_bytes: bytes) -> dict[str, object]:
    return {
        "info": {"name": "anima-forge", "version": "3.0.0"},
        "urls": [
            _record(wheel, wheel_bytes, "bdist_wheel"),
            _record(sdist, sdist_bytes, "sdist"),
        ],
    }


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        self.close()


def test_exact_testpypi_distribution_pair_is_downloaded_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    wheel, wheel_bytes, sdist, sdist_bytes = _distributions(tmp_path)
    payload = _payload(wheel, wheel_bytes, sdist, sdist_bytes)
    responses = iter(
        (
            _Response(json.dumps(payload).encode()),
            _Response(wheel_bytes),
            _Response(sdist_bytes),
        )
    )
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *_args, **_kwargs: next(responses))

    result = module.verify_testpypi_artifacts(
        dist_dir=wheel.parent,
        download_dir=tmp_path / "download",
        package="anima_forge",
    )

    downloaded = tmp_path / "download"
    assert result["status"] == "verified"
    assert result["wheel_sha256"] == hashlib.sha256(wheel_bytes).hexdigest()
    assert result["sdist_sha256"] == hashlib.sha256(sdist_bytes).hexdigest()
    assert {item["kind"] for item in result["artifacts"]} == {"wheel", "sdist"}
    assert (downloaded / wheel.name).read_bytes() == wheel_bytes
    assert (downloaded / sdist.name).read_bytes() == sdist_bytes
    assert sorted(path.name for path in downloaded.iterdir()) == sorted((wheel.name, sdist.name))


def test_downloaded_sdist_mismatch_preserves_existing_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    wheel, wheel_bytes, sdist, sdist_bytes = _distributions(tmp_path)
    download = tmp_path / "download"
    download.mkdir()
    (download / wheel.name).write_bytes(b"previous wheel")
    (download / sdist.name).write_bytes(b"previous sdist")
    responses = iter(
        (
            _Response(json.dumps(_payload(wheel, wheel_bytes, sdist, sdist_bytes)).encode()),
            _Response(wheel_bytes),
            _Response(b"corrupt sdist download"),
        )
    )
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *_args, **_kwargs: next(responses))

    with pytest.raises(module.ArtifactVerificationError, match="Downloaded TestPyPI sdist SHA-256 mismatch"):
        module.verify_testpypi_artifacts(
            dist_dir=wheel.parent,
            download_dir=download,
            package="anima-forge",
        )

    assert (download / wheel.name).read_bytes() == b"previous wheel"
    assert (download / sdist.name).read_bytes() == b"previous sdist"
    assert sorted(path.name for path in download.iterdir()) == sorted((wheel.name, sdist.name))


@pytest.mark.parametrize("case", ["duplicate", "yanked", "untrusted", "wrong-type"])
def test_invalid_remote_sdist_record_is_rejected(
    case: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    wheel, wheel_bytes, sdist, sdist_bytes = _distributions(tmp_path)
    payload = _payload(wheel, wheel_bytes, sdist, sdist_bytes)
    records = payload["urls"]
    assert isinstance(records, list)
    sdist_record = records[1]
    assert isinstance(sdist_record, dict)
    if case == "duplicate":
        records.append(dict(sdist_record))
    elif case == "yanked":
        sdist_record["yanked"] = True
    elif case == "untrusted":
        sdist_record["url"] = f"https://example.invalid/{sdist.name}"
    else:
        sdist_record["packagetype"] = "bdist_wheel"
    monkeypatch.setattr(
        module.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _Response(json.dumps(payload).encode()),
    )

    with pytest.raises(module.ArtifactVerificationError):
        module.verify_testpypi_artifacts(
            dist_dir=wheel.parent,
            download_dir=tmp_path / "download",
            package="anima-forge",
        )

    assert not (tmp_path / "download").exists()


@pytest.mark.parametrize("case", ["missing-sdist", "extra-distribution"])
def test_missing_or_extra_local_distribution_is_rejected(case: str, tmp_path: Path) -> None:
    module = _load_module()
    wheel, _wheel_bytes, sdist, _sdist_bytes = _distributions(tmp_path)
    if case == "missing-sdist":
        sdist.unlink()
    else:
        (wheel.parent / "unexpected.zip").write_bytes(b"not part of the release")

    with pytest.raises(module.ArtifactVerificationError, match="exactly one wheel and one .tar.gz sdist"):
        module.verify_testpypi_artifacts(
            dist_dir=wheel.parent,
            download_dir=tmp_path / "download",
            package="anima-forge",
        )


def test_uv_build_gitignore_is_not_treated_as_a_distribution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    wheel, wheel_bytes, sdist, sdist_bytes = _distributions(tmp_path)
    (wheel.parent / ".gitignore").write_text("*", encoding="utf-8")
    payload = _payload(wheel, wheel_bytes, sdist, sdist_bytes)
    responses = iter(
        (
            _Response(json.dumps(payload).encode()),
            _Response(wheel_bytes),
            _Response(sdist_bytes),
        )
    )
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *_args, **_kwargs: next(responses))

    result = module.verify_testpypi_artifacts(
        dist_dir=wheel.parent,
        download_dir=tmp_path / "download",
        package="anima-forge",
    )

    assert result["status"] == "verified"


def test_wheel_and_sdist_versions_must_match(tmp_path: Path) -> None:
    module = _load_module()
    wheel, _wheel_bytes, _sdist, _sdist_bytes = _distributions(tmp_path, sdist_version="3.0.1")

    with pytest.raises(module.ArtifactVerificationError, match="Distribution version mismatch"):
        module.verify_testpypi_artifacts(
            dist_dir=wheel.parent,
            download_dir=tmp_path / "download",
            package="anima-forge",
        )
