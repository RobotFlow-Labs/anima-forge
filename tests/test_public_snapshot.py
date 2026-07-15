"""Public-release snapshot privacy contracts."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


def _module():
    spec = importlib.util.spec_from_file_location("forge_public_snapshot", "scripts/check_public_snapshot.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_audit_rejects_nonproduction_tracked_paths_even_when_deleted(tmp_path: Path) -> None:
    module = _module()

    findings = module.audit_paths(
        tmp_path,
        [
            "BUILDING_PLAN.md",
            "CLAUDE.md",
            "FORGE_PIPELINE.md",
            "anima_module.yaml",
            "docs/v3/private.md",
            "setup-claude-skills.sh",
            "src/forge/__init__.py",
        ],
    )

    rejected = {finding.path for finding in findings if finding.rule == "nonproduction-tracked-file"}
    assert rejected == {
        "BUILDING_PLAN.md",
        "CLAUDE.md",
        "FORGE_PIPELINE.md",
        "anima_module.yaml",
        "docs/v3/private.md",
        "setup-claude-skills.sh",
    }


def test_audit_finds_private_content_without_echoing_values(tmp_path: Path) -> None:
    module = _module()
    private_path = "/" + "mnt/forge-data/models"
    private_alias = "datai_" + "srv7_development"
    private_token = "hf" + "_abcdefghijklmnopqrstuvwxyz123456"
    private_ip = "10." + "24.3.8"
    operator_instruction = "Restart " + "Claude Code after syncing"
    source = tmp_path / "notes.md"
    source.write_text(
        f"{private_path}\n{private_alias}\n{private_token}\nhost: {private_ip}\n{operator_instruction}\n",
        encoding="utf-8",
    )

    findings = module.audit_paths(tmp_path, ["notes.md"])

    assert [(finding.rule, finding.line) for finding in findings] == [
        ("private-mount", 1),
        ("internal-ssh-alias", 2),
        ("hugging-face-token", 3),
        ("private-ipv4", 4),
        ("internal-operator-instruction", 5),
    ]
    rendered = repr(findings)
    assert private_path not in rendered
    assert private_alias not in rendered
    assert private_token not in rendered
    assert private_ip not in rendered
    assert operator_instruction not in rendered


def test_audit_ignores_binary_and_public_placeholder_content(tmp_path: Path) -> None:
    module = _module()
    (tmp_path / "binary.bin").write_bytes(b"\0/" + b"mnt/private")
    (tmp_path / "README.md").write_text(
        "Set HF_TOKEN in your shell and keep model weights under ~/.cache/forge.\n",
        encoding="utf-8",
    )

    assert module.audit_paths(tmp_path, ["binary.bin", "README.md"]) == []


def test_audit_finds_windows_paths_and_bare_private_addresses(tmp_path: Path) -> None:
    module = _module()
    source = tmp_path / "windows-notes.txt"
    windows_path = "C:" + "\\" + "Users" + "\\" + "operator" + "\\" + "models" + "\\" + "teacher"
    private_ip = "192" + ".168.50.4"
    unc_path = "\\" * 2 + "internal-host" + "\\" + "weights"
    source.write_text(
        f"model={windows_path}\nserver={private_ip}\nshare={unc_path}\n",
        encoding="utf-8",
    )

    findings = module.audit_paths(tmp_path, [source.name])

    assert [(finding.rule, finding.line) for finding in findings] == [
        ("private-windows-home", 1),
        ("private-ipv4", 2),
        ("private-unc-path", 3),
    ]


def test_audit_finds_common_credentials_and_private_keys_without_echoing_values(tmp_path: Path) -> None:
    module = _module()
    github_token = "ghp_" + "a" * 32
    aws_key = "AKIA" + "B" * 16
    private_key = "-----BEGIN " + "OPENSSH PRIVATE KEY-----"
    source = tmp_path / "credentials.txt"
    source.write_text(f"{github_token}\n{aws_key}\n{private_key}\n", encoding="utf-8")

    findings = module.audit_paths(tmp_path, [source.name])

    assert [(finding.rule, finding.line) for finding in findings] == [
        ("github-token", 1),
        ("aws-access-key", 2),
        ("private-key", 3),
    ]
    rendered = repr(findings)
    assert github_token not in rendered
    assert aws_key not in rendered
    assert private_key not in rendered


def test_intended_audit_excludes_archival_paths_and_includes_untracked_files(tmp_path: Path) -> None:
    module = _module()
    subprocess.run(["git", "init", "--quiet"], cwd=tmp_path, check=True)
    planned = tmp_path / "BUILDING_PLAN.md"
    planned.write_text("private path: /" + "mnt/operator\n", encoding="utf-8")
    public = tmp_path / "README.md"
    public.write_text("public\n", encoding="utf-8")
    subprocess.run(["git", "add", "BUILDING_PLAN.md", "README.md"], cwd=tmp_path, check=True)
    untracked = tmp_path / "new-config.yaml"
    untracked.write_text("private path: /" + "home/operator\n", encoding="utf-8")
    ignored = tmp_path / "models" / "weights.txt"
    ignored.parent.mkdir()
    ignored.write_text("private path: /" + "mnt/ignored\n", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("models/\n", encoding="utf-8")

    paths = module.intended_production_paths(tmp_path)
    findings = module.audit_intended_repository(tmp_path)

    assert "BUILDING_PLAN.md" not in paths
    assert "models/weights.txt" not in paths
    assert "new-config.yaml" in paths
    assert [(finding.path, finding.rule) for finding in findings] == [("new-config.yaml", "private-home")]


def test_worktree_audit_reads_symlink_target_without_following_it(tmp_path: Path) -> None:
    module = _module()
    outside = tmp_path.parent / "outside-private-file"
    outside.write_text("public bytes\n", encoding="utf-8")
    link = tmp_path / "model-link"
    link.symlink_to("/" + "mnt/private/model")

    findings = module.audit_paths(tmp_path, [link.name])

    assert [(finding.rule, finding.line) for finding in findings] == [("private-mount", 1)]


def test_repository_audit_reads_staged_snapshot_not_worktree(tmp_path: Path) -> None:
    module = _module()
    subprocess.run(["git", "init", "--quiet"], cwd=tmp_path, check=True)
    source = tmp_path / "README.md"
    private_path = "/" + "mnt/operator/models"
    source.write_text(f"private path: {private_path}\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=tmp_path, check=True)

    source.write_text("public working-tree content\n", encoding="utf-8")

    assert [(finding.rule, finding.line) for finding in module.audit_repository(tmp_path)] == [("private-mount", 1)]


def test_audit_ignores_package_versions_but_not_real_private_addresses(tmp_path: Path) -> None:
    module = _module()
    source = tmp_path / "versions.txt"
    private_ip = "10" + ".24.3.8"
    source.write_text(
        'version = "10.16.0.72"\n'
        'dependency = "tensorrt==10.16.0.72"\n'
        'url = "nvidia_curand_cu12-10.3.9.90-py3-none.whl"\n'
        f'version = "10.16.0.72" server={private_ip}\n',
        encoding="utf-8",
    )

    findings = module.audit_paths(tmp_path, [source.name])

    assert [(finding.rule, finding.line) for finding in findings] == [("private-ipv4", 4)]


def test_inline_suppression_is_rule_and_path_scoped(tmp_path: Path) -> None:
    module = _module()
    detector = tmp_path / "src" / "forge" / "hub_package.py"
    detector.parent.mkdir(parents=True)
    unc_pattern = "\\" * 2 + "detector-host" + "\\" + "share"
    line = f"pattern={unc_pattern}  # forge-public-audit: allow[private-unc-path]\n"
    detector.write_text(line, encoding="utf-8")
    untrusted = tmp_path / "README.md"
    untrusted.write_text(line, encoding="utf-8")

    assert module.audit_paths(tmp_path, ["src/forge/hub_package.py"]) == []
    findings = module.audit_paths(tmp_path, ["README.md"])
    assert [(finding.rule, finding.line) for finding in findings] == [("private-unc-path", 1)]
