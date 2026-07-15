"""Verify and download the exact release distributions published to TestPyPI."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from email.message import Message
from email.parser import BytesParser
from pathlib import Path
from typing import Any

TRUSTED_ARTIFACT_HOSTS = frozenset({"test-files.pythonhosted.org"})
TRUSTED_INDEX_HOSTS = frozenset({"test.pypi.org"})


class ArtifactVerificationError(RuntimeError):
    """The published artifact violates the exact-release contract."""


class ArtifactPendingError(ArtifactVerificationError):
    """The expected immutable artifacts are not visible on TestPyPI yet."""


@dataclass(frozen=True, slots=True)
class LocalArtifact:
    """One locally built distribution and its immutable identity."""

    kind: str
    path: Path
    package: str
    version: str
    sha256: str
    package_type: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_package(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _metadata_identity(metadata: Message, *, artifact: Path, kind: str) -> tuple[str, str]:
    package = _normalize_package(metadata.get("Name", "").strip())
    version = metadata.get("Version", "").strip()
    if not package:
        raise ArtifactVerificationError(f"{kind} metadata has no Name: {artifact}")
    if not version:
        raise ArtifactVerificationError(f"{kind} metadata has no Version: {artifact}")
    return package, version


def _wheel_artifact(path: Path) -> LocalArtifact:
    try:
        with zipfile.ZipFile(path) as archive:
            metadata_names = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
            if len(metadata_names) != 1:
                raise ArtifactVerificationError(
                    f"Expected exactly one METADATA file in {path.name}, found {len(metadata_names)}"
                )
            metadata = BytesParser().parsebytes(archive.read(metadata_names[0]))
    except zipfile.BadZipFile as exc:
        raise ArtifactVerificationError(f"Invalid wheel archive: {path}") from exc
    package, version = _metadata_identity(metadata, artifact=path, kind="Wheel")
    return LocalArtifact("wheel", path, package, version, _sha256(path), "bdist_wheel")


def _sdist_artifact(path: Path) -> LocalArtifact:
    try:
        with tarfile.open(path, mode="r:gz") as archive:
            metadata_members = [member for member in archive.getmembers() if member.name.endswith("/PKG-INFO")]
            if len(metadata_members) != 1:
                raise ArtifactVerificationError(
                    f"Expected exactly one PKG-INFO file in {path.name}, found {len(metadata_members)}"
                )
            metadata_file = archive.extractfile(metadata_members[0])
            if metadata_file is None:
                raise ArtifactVerificationError(f"Could not read PKG-INFO from {path.name}")
            metadata = BytesParser().parsebytes(metadata_file.read())
    except tarfile.TarError as exc:
        raise ArtifactVerificationError(f"Invalid source distribution archive: {path}") from exc
    package, version = _metadata_identity(metadata, artifact=path, kind="Source distribution")
    return LocalArtifact("sdist", path, package, version, _sha256(path), "sdist")


def _distribution_pair(dist_dir: Path, *, expected_package: str) -> tuple[LocalArtifact, LocalArtifact]:
    # ``uv build``/Hatch creates ``dist/.gitignore`` containing ``*`` in a
    # fresh output directory. It is build-tool bookkeeping, not a release
    # artifact; continue to reject every other unexpected file fail-closed.
    files = sorted(path for path in dist_dir.iterdir() if path.is_file() and path.name != ".gitignore")
    wheels = [path for path in files if path.suffix == ".whl"]
    sdists = [path for path in files if path.name.endswith(".tar.gz")]
    if len(wheels) != 1 or len(sdists) != 1 or len(files) != 2:
        raise ArtifactVerificationError(
            f"Expected exactly one wheel and one .tar.gz sdist in {dist_dir}; "
            f"found {len(wheels)} wheel(s), {len(sdists)} sdist(s), and {len(files)} total file(s)"
        )
    wheel = _wheel_artifact(wheels[0])
    sdist = _sdist_artifact(sdists[0])
    for artifact in (wheel, sdist):
        if artifact.package != expected_package:
            raise ArtifactVerificationError(
                f"{artifact.kind} package mismatch: expected {expected_package}, observed {artifact.package}"
            )
    if wheel.version != sdist.version:
        raise ArtifactVerificationError(f"Distribution version mismatch: wheel {wheel.version}, sdist {sdist.version}")
    return wheel, sdist


def _load_payload(url: str, *, timeout: float) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise ArtifactPendingError(f"TestPyPI release is not visible yet: {url}") from exc
        raise
    if not isinstance(payload, dict):
        raise ArtifactVerificationError("TestPyPI returned a non-object JSON payload")
    return payload


def _remote_artifacts(
    payload: dict[str, Any],
    *,
    package: str,
    version: str,
    local_artifacts: tuple[LocalArtifact, LocalArtifact],
) -> dict[str, tuple[str, str]]:
    info = payload.get("info")
    if not isinstance(info, dict):
        raise ArtifactVerificationError("TestPyPI metadata has no package information")
    remote_package = _normalize_package(str(info.get("name", "")))
    if remote_package != package or info.get("version") != version:
        raise ArtifactVerificationError(f"TestPyPI metadata does not identify {package} {version}")
    urls = payload.get("urls")
    if not isinstance(urls, list) or any(not isinstance(item, dict) for item in urls):
        raise ArtifactVerificationError("TestPyPI metadata has an invalid artifact list")
    expected_names = {artifact.path.name for artifact in local_artifacts}
    remote_names = [str(item.get("filename", "")) for item in urls]
    if set(remote_names) != expected_names or len(remote_names) != len(expected_names):
        missing = sorted(expected_names - set(remote_names))
        extra = sorted(set(remote_names) - expected_names)
        if missing and not extra:
            raise ArtifactPendingError(f"TestPyPI does not serve expected distribution(s) yet: {', '.join(missing)}")
        raise ArtifactVerificationError(
            f"TestPyPI distribution set mismatch; missing={missing or []}, extra={extra or []}, "
            f"records={len(remote_names)}"
        )

    remote: dict[str, tuple[str, str]] = {}
    for artifact in local_artifacts:
        match = next(item for item in urls if item.get("filename") == artifact.path.name)
        if match.get("packagetype") != artifact.package_type:
            raise ArtifactVerificationError(
                f"TestPyPI record for {artifact.path.name} has wrong package type: {match.get('packagetype')}"
            )
        if match.get("yanked") is True:
            raise ArtifactVerificationError(f"TestPyPI record for {artifact.path.name} is yanked")
        digests = match.get("digests")
        remote_url = match.get("url")
        if not isinstance(digests, dict) or not isinstance(digests.get("sha256"), str):
            raise ArtifactVerificationError(f"TestPyPI record for {artifact.path.name} has no SHA-256 digest")
        if not isinstance(remote_url, str):
            raise ArtifactVerificationError(f"TestPyPI record for {artifact.path.name} has no download URL")
        parsed = urllib.parse.urlsplit(remote_url)
        if parsed.scheme != "https" or parsed.hostname not in TRUSTED_ARTIFACT_HOSTS:
            raise ArtifactVerificationError(f"Refusing untrusted TestPyPI artifact URL: {remote_url}")
        published_sha256 = digests["sha256"].lower()
        if published_sha256 != artifact.sha256:
            raise ArtifactVerificationError(
                f"TestPyPI {artifact.kind} SHA-256 does not match the local release artifact: "
                f"local {artifact.sha256}, published {published_sha256}"
            )
        remote[artifact.kind] = remote_url, published_sha256
    return remote


def _stage_exact_download(
    url: str,
    destination: Path,
    *,
    expected_sha256: str,
    kind: str,
    timeout: float,
) -> Path:
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.{kind}.tmp")
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response, temporary.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
                digest.update(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        observed = digest.hexdigest()
        if observed != expected_sha256:
            raise ArtifactVerificationError(
                f"Downloaded TestPyPI {kind} SHA-256 mismatch: expected {expected_sha256}, observed {observed}"
            )
        return temporary
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def verify_testpypi_artifacts(
    *,
    dist_dir: Path,
    download_dir: Path,
    package: str,
    index_url: str = "https://test.pypi.org/pypi",
    timeout: float = 20.0,
) -> dict[str, Any]:
    """Verify and publish the byte-identical TestPyPI wheel/sdist pair locally."""
    normalized_package = _normalize_package(package)
    local_artifacts = _distribution_pair(dist_dir, expected_package=normalized_package)
    version = local_artifacts[0].version
    parsed_index = urllib.parse.urlsplit(index_url)
    if parsed_index.scheme != "https" or parsed_index.hostname not in TRUSTED_INDEX_HOSTS:
        raise ArtifactVerificationError(f"Refusing untrusted TestPyPI index URL: {index_url}")
    package_path = urllib.parse.quote(normalized_package, safe="")
    version_path = urllib.parse.quote(version, safe="")
    metadata_url = f"{index_url.rstrip('/')}/{package_path}/{version_path}/json"
    payload = _load_payload(metadata_url, timeout=timeout)
    remote = _remote_artifacts(
        payload,
        package=normalized_package,
        version=version,
        local_artifacts=local_artifacts,
    )

    download_dir.mkdir(parents=True, exist_ok=True)
    staged: dict[str, Path] = {}
    try:
        for artifact in local_artifacts:
            remote_url, _published_sha256 = remote[artifact.kind]
            destination = download_dir / artifact.path.name
            staged[artifact.kind] = _stage_exact_download(
                remote_url,
                destination,
                expected_sha256=artifact.sha256,
                kind=artifact.kind,
                timeout=timeout,
            )
        for artifact in local_artifacts:
            os.replace(staged[artifact.kind], download_dir / artifact.path.name)
    finally:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)

    artifacts = [
        {
            "kind": artifact.kind,
            "filename": artifact.path.name,
            "sha256": artifact.sha256,
            "downloaded_path": str(download_dir / artifact.path.name),
        }
        for artifact in local_artifacts
    ]
    return {
        "schema": "forge.testpypi-artifacts.v2",
        "status": "verified",
        "package": normalized_package,
        "version": version,
        "wheel_sha256": local_artifacts[0].sha256,
        "sdist_sha256": local_artifacts[1].sha256,
        "artifacts": artifacts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", type=Path, default=Path("dist"))
    parser.add_argument("--download-dir", type=Path, default=Path("testpypi-dist"))
    parser.add_argument("--package", default="anima-forge")
    parser.add_argument("--index-url", default="https://test.pypi.org/pypi")
    parser.add_argument("--attempts", type=int, default=12)
    parser.add_argument("--delay", type=float, default=10.0)
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args()
    if args.attempts < 1:
        parser.error("--attempts must be at least 1")
    if args.delay < 0 or args.timeout <= 0:
        parser.error("--delay must be non-negative and --timeout must be positive")

    for attempt in range(1, args.attempts + 1):
        try:
            result = verify_testpypi_artifacts(
                dist_dir=args.dist_dir,
                download_dir=args.download_dir,
                package=args.package,
                index_url=args.index_url,
                timeout=args.timeout,
            )
        except (ArtifactPendingError, urllib.error.URLError, TimeoutError) as exc:
            if attempt == args.attempts:
                print(f"TestPyPI artifact verification failed after {attempt} attempts: {exc}", file=sys.stderr)
                return 1
            print(f"TestPyPI artifacts pending (attempt {attempt}/{args.attempts}): {exc}", file=sys.stderr)
            time.sleep(args.delay)
        except (ArtifactVerificationError, OSError, json.JSONDecodeError) as exc:
            print(f"TestPyPI artifact verification failed: {exc}", file=sys.stderr)
            return 1
        else:
            print(json.dumps(result, indent=2, allow_nan=False))
            return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
