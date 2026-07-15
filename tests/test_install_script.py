"""Behavioral contracts for the PRD-45 user-space installers."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parents[1]


def _fake_uv(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    home = tmp_path / "home"
    bin_dir = tmp_path / "tool-bin"
    tool_dir = tmp_path / "tools"
    fake = tmp_path / "uv"
    log = tmp_path / "uv.log"
    fake.write_text(
        f'''#!{sys.executable}
import os
import pathlib
import sys

args = sys.argv[1:]
log = pathlib.Path(os.environ["FORGE_TEST_UV_LOG"])
with log.open("a", encoding="utf-8") as handle:
    handle.write(" ".join(args) + "\\n")
bin_dir = pathlib.Path(os.environ["UV_TOOL_BIN_DIR"])
tool_dir = pathlib.Path(os.environ["UV_TOOL_DIR"])
if args[:2] == ["tool", "install"]:
    bin_dir.mkdir(parents=True, exist_ok=True)
    env_python = tool_dir / "anima-forge" / "bin" / "python"
    env_python.parent.mkdir(parents=True, exist_ok=True)
    if not env_python.exists():
        env_python.symlink_to(os.environ["FORGE_TEST_PYTHON"])
    forge = bin_dir / "forge"
    forge.write_text("""#!/bin/sh
case "$1" in
  --version) printf '3.0.1\\n' ;;
  doctor) printf '{{"status":"error","checks":[]}}\\n'; exit 2 ;;
  *) exit 0 ;;
esac
""", encoding="utf-8")
    forge.chmod(0o755)
elif args[:3] == ["tool", "dir", "--bin"]:
    print(bin_dir)
elif args[:2] == ["tool", "dir"]:
    print(tool_dir)
elif args[:2] in (["tool", "update-shell"], ["tool", "uninstall"]):
    pass
else:
    raise SystemExit(f"unexpected fake uv arguments: {{args}}")
''',
        encoding="utf-8",
    )
    fake.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "HOME": str(home),
            "UV_TOOL_BIN_DIR": str(bin_dir),
            "UV_TOOL_DIR": str(tool_dir),
            "FORGE_INSTALL_UV_BIN": str(fake),
            "FORGE_TEST_UV_LOG": str(log),
            "FORGE_TEST_PYTHON": sys.executable,
        }
    )
    return log, env


def _run_installer(tmp_path: Path, *args: str) -> tuple[subprocess.CompletedProcess[str], Path]:
    log, env = _fake_uv(tmp_path)
    completed = subprocess.run(
        ["sh", str(REPO_ROOT / "install.sh"), *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed, log


def test_install_sh_cpu_local_wheel_and_doctor_json(tmp_path: Path) -> None:
    wheel = tmp_path / "anima_forge-3.0.1-py3-none-any.whl"
    wheel.touch()

    completed, log = _run_installer(tmp_path, "--cpu", "--from-wheel", str(wheel), "--no-modify-path")

    assert completed.returncode == 0, completed.stderr
    assert "Installed FORGE 3.0.1" in completed.stdout
    assert "readiness warnings (exit 2)" in completed.stdout
    invocations = log.read_text(encoding="utf-8")
    assert "anima-forge @ file://" in invocations
    assert "--index https://download.pytorch.org/whl/cpu" in invocations
    assert "--index-strategy unsafe-best-match" in invocations
    assert "tool update-shell" not in invocations


def test_install_sh_cuda_uses_complete_package_and_updates_path(tmp_path: Path) -> None:
    wheel = tmp_path / "anima_forge-3.0.1-py3-none-any.whl"
    wheel.touch()

    completed, log = _run_installer(tmp_path, "--cuda", "--from-wheel", str(wheel))

    assert completed.returncode == 0, completed.stderr
    invocations = log.read_text(encoding="utf-8")
    assert "anima-forge @ file://" in invocations
    assert "anima-forge[cuda]" not in invocations
    assert "download.pytorch.org/whl/cpu" not in invocations
    assert "tool update-shell" in invocations


def test_install_sh_uninstall_uses_selected_backend(tmp_path: Path) -> None:
    completed, log = _run_installer(tmp_path, "--uninstall")

    assert completed.returncode == 0, completed.stderr
    assert "tool uninstall anima-forge" in log.read_text(encoding="utf-8")
    assert "FORGE uninstalled" in completed.stdout


def test_install_sh_rejects_unknown_option(tmp_path: Path) -> None:
    completed, _log = _run_installer(tmp_path, "--not-real")

    assert completed.returncode != 0
    assert "unknown option" in completed.stderr
    assert "Manual fallback" in completed.stderr


def test_install_sh_rejects_insecure_remote_wheel(tmp_path: Path) -> None:
    completed, log = _run_installer(tmp_path, "--from-wheel", "http://example.invalid/anima-forge.whl")

    assert completed.returncode != 0
    assert "remote --from-wheel URLs must use HTTPS" in completed.stderr
    assert not log.exists() or "tool install" not in log.read_text(encoding="utf-8")


def test_installer_files_have_release_safety_contracts() -> None:
    shell = (REPO_ROOT / "install.sh").read_text(encoding="utf-8")
    powershell = (REPO_ROOT / "install.ps1").read_text(encoding="utf-8")

    assert shell.startswith("#!/bin/sh\n")
    assert "set -eu" in shell
    assert "sudo" not in shell
    assert "--from-wheel" in shell
    assert 'http://*)\n            die "remote --from-wheel URLs must use HTTPS"' in shell
    assert "tool install --force" in shell
    assert "https://download.pytorch.org/whl/cpu" in shell
    assert "https://download.pytorch.org/whl/cpu" in powershell
    assert "pipx install --force --python 3.12" in shell
    assert "pipx --version" in shell
    assert "--upgrade --force-reinstall pipx" in shell
    assert "UV_INDEX_STRATEGY=unsafe-best-match" in shell
    assert "pipx environment --value PIPX_BIN_DIR" in shell
    assert "forge doctor --json" in shell
    assert "ConvertFrom-Json" in powershell
    assert "tool install --force" in powershell
    assert "Remote -FromWheel URLs must use HTTPS." in powershell
    assert 'Assert-NativeSuccess "uv tool uninstall" $LASTEXITCODE' in powershell
    assert 'Assert-NativeSuccess "uv tool install" $LASTEXITCODE' in powershell
    assert 'Assert-NativeSuccess "uv tool dir --bin" $LASTEXITCODE' in powershell
    assert 'Assert-NativeSuccess "uv tool update-shell" $LASTEXITCODE' in powershell
    assert 'Assert-NativeSuccess "forge --version" $LASTEXITCODE' in powershell
    assert "*> $DoctorFile" not in powershell
    assert "2> $null" in powershell
    assert powershell.rstrip().endswith("exit 0")
