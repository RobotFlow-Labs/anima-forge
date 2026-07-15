"""PRD-33 A2 tests for upstream model asset fetching."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

import forge.cli_commands.fetch as fetch_module
from forge.cli_commands._doctor_core import MIN_MODEL_WEIGHT_BYTES
from forge.cli_v2 import app
from forge.model_assets import ModelAsset

TEST_ASSET = ModelAsset("example/fetch-model", "student:test")


def _model_tree(path: Path, *, valid: bool = True) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}" if valid else "{broken", encoding="utf-8")
    with (path / "model.safetensors").open("wb") as stream:
        stream.truncate(MIN_MODEL_WEIGHT_BYTES + 1)
    return path


def _cache_snapshot(cache: Path, asset: ModelAsset = TEST_ASSET) -> Path:
    return _model_tree(cache / f"models--{asset.local_name}" / "snapshots" / "sha")


def test_named_selector_accepts_known_and_custom_repositories() -> None:
    known = fetch_module.select_fetch_assets(
        name="Qwen/Qwen3-0.6B",
        all_students=False,
        teachers=False,
        all_assets=False,
    )
    custom = fetch_module.select_fetch_assets(
        name="Qwen/Qwen2.5-1.5B",
        all_students=False,
        teachers=False,
        all_assets=False,
    )

    assert known[0].role == "student:nano"
    assert custom == (ModelAsset("Qwen/Qwen2.5-1.5B", "custom", required=False),)


def test_group_selectors_match_v3_fleet() -> None:
    students = fetch_module.select_fetch_assets(
        name=None,
        all_students=True,
        teachers=False,
        all_assets=False,
    )
    teachers = fetch_module.select_fetch_assets(
        name=None,
        all_students=False,
        teachers=True,
        all_assets=False,
    )

    assert len(students) == 8
    assert {asset.repo_id for asset in students} >= {
        "HuggingFaceTB/SmolLM2-135M",
        "Qwen/Qwen3-4B",
        "Qwen/Qwen3.5-4B",
        "google/siglip2-so400m-patch14-384",
    }
    assert {asset.repo_id for asset in teachers} == {
        "openvla/openvla-7b",
        "robotics-diffusion-transformer/RDT2-FM",
        "robotics-diffusion-transformer/RDT2-VQ",
        "lerobot/smolvla_base",
        "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        "allenai/MolmoAct2-LIBERO-LeRobot",
        "allenai/MolmoAct2-LIBERO",
        "allenai/MolmoAct2-FAST-Tokenizer",
        "lerobot/VLA-JEPA-Pretrain",
        "Qwen/Qwen3-VL-2B-Instruct",
        "facebook/vjepa2-vitl-fpc64-256",
    }


def test_fetches_and_reuses_verified_sidecar(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = b"official-normalizer"
    import hashlib

    asset = ModelAsset(
        "example/fetch-model",
        "teacher",
        sidecar_url="https://example.test/normalizer.pt",
        sidecar_filename="normalizer.pt",
        sidecar_sha256=hashlib.sha256(payload).hexdigest(),
    )
    _model_tree(tmp_path / "models" / asset.local_name)
    calls = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, _size):
            return payload if not calls else b""

    monkeypatch.setattr(fetch_module.urllib.request, "urlopen", lambda *_args, **_kwargs: Response())
    calls.clear()
    original_read = Response.read

    def read_once(self, size):
        result = original_read(self, size)
        calls.append(result)
        return result

    monkeypatch.setattr(Response, "read", read_once)
    report = fetch_module.fetch_assets((asset,), model_dir=tmp_path / "models", cache_dir=tmp_path / "hub", token=None)
    assert report["status"] == "ok"
    assert (tmp_path / "models" / "normalizer.pt").read_bytes() == payload

    monkeypatch.setattr(
        fetch_module.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: pytest.fail("verified sidecar should be reused"),
    )
    assert (
        fetch_module.fetch_assets((asset,), model_dir=tmp_path / "models", cache_dir=tmp_path / "hub", token=None)[
            "status"
        ]
        == "ok"
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": None, "all_students": False, "teachers": False, "all_assets": False},
        {"name": "example/model", "all_students": True, "teachers": False, "all_assets": False},
        {"name": "not-a-repo", "all_students": False, "teachers": False, "all_assets": False},
    ],
)
def test_selector_rejects_missing_ambiguous_or_invalid_choice(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        fetch_module.select_fetch_assets(**kwargs)


def test_existing_real_model_is_idempotent_and_skips_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir = tmp_path / "models"
    _model_tree(model_dir / TEST_ASSET.local_name)
    monkeypatch.setattr(
        fetch_module,
        "_snapshot_download",
        lambda *args, **kwargs: pytest.fail("network should not be used"),
    )

    report = fetch_module.fetch_assets(
        (TEST_ASSET,),
        model_dir=model_dir,
        cache_dir=tmp_path / "hub",
        token=None,
    )

    assert report["status"] == "ok"
    assert report["results"][0]["status"] == "already_present"


def test_valid_cache_snapshot_is_linked_without_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = tmp_path / "hub"
    snapshot = _cache_snapshot(cache)
    monkeypatch.setattr(
        fetch_module,
        "_snapshot_download",
        lambda *args, **kwargs: pytest.fail("network should not be used"),
    )

    report = fetch_module.fetch_assets(
        (TEST_ASSET,),
        model_dir=tmp_path / "models",
        cache_dir=cache,
        token=None,
    )

    destination = tmp_path / "models" / TEST_ASSET.local_name
    assert report["results"][0]["status"] == "linked_from_cache"
    assert destination.is_symlink()
    assert destination.resolve() == snapshot.resolve()


def test_download_disables_xet_installs_link_and_restores_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}
    monkeypatch.setenv("HF_HUB_DISABLE_XET", "previous")

    def fake_download(repo_id: str, *, cache_dir: Path, token: str | None) -> Path:
        seen.update(
            repo_id=repo_id,
            cache_dir=cache_dir,
            token=token,
            xet=os.environ["HF_HUB_DISABLE_XET"],
            download_timeout=os.environ["HF_HUB_DOWNLOAD_TIMEOUT"],
            etag_timeout=os.environ["HF_HUB_ETAG_TIMEOUT"],
        )
        return _cache_snapshot(cache_dir)

    monkeypatch.setattr(fetch_module, "_snapshot_download", fake_download)

    report = fetch_module.fetch_assets(
        (TEST_ASSET,),
        model_dir=tmp_path / "models",
        cache_dir=tmp_path / "hub",
        token="hf_test_secret",
    )

    assert report["results"][0]["status"] == "downloaded"
    assert seen == {
        "repo_id": TEST_ASSET.repo_id,
        "cache_dir": tmp_path / "hub",
        "token": "hf_test_secret",
        "xet": "1",
        "download_timeout": "600",
        "etag_timeout": "60",
    }
    assert os.environ["HF_HUB_DISABLE_XET"] == "previous"
    assert (tmp_path / "models" / TEST_ASSET.local_name).is_symlink()


def test_group_fetch_continues_and_redacts_token_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    good = TEST_ASSET
    bad = ModelAsset("example/private-model", "student:test")
    token = "hf_never_print_this"

    def fake_download(repo_id: str, *, cache_dir: Path, token: str | None) -> Path:
        if repo_id == bad.repo_id:
            raise RuntimeError(f"request rejected for {token}")
        return _cache_snapshot(cache_dir, asset=good)

    monkeypatch.setattr(fetch_module, "_snapshot_download", fake_download)

    report = fetch_module.fetch_assets(
        (bad, good),
        model_dir=tmp_path / "models",
        cache_dir=tmp_path / "hub",
        token=token,
    )

    assert report["status"] == "error"
    assert report["exit_code"] == 2
    assert report["summary"] == {"requested": 2, "succeeded": 1, "failed": 1}
    assert token not in json.dumps(report)
    assert [result["status"] for result in report["results"]] == ["error", "downloaded"]


def test_incomplete_real_directory_is_preserved(tmp_path: Path) -> None:
    model_dir = tmp_path / "models"
    incomplete = model_dir / TEST_ASSET.local_name
    incomplete.mkdir(parents=True)
    (incomplete / "keep.txt").write_text("user data", encoding="utf-8")

    report = fetch_module.fetch_assets(
        (TEST_ASSET,),
        model_dir=model_dir,
        cache_dir=tmp_path / "hub",
        token=None,
    )

    assert report["exit_code"] == 2
    assert "incomplete model directory" in report["results"][0]["error"]
    assert (incomplete / "keep.txt").read_text(encoding="utf-8") == "user data"


def test_cli_json_output_is_clean_and_registered(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("forge.cli_v2_root.setup_cli_logging", lambda **kwargs: None)
    monkeypatch.setattr(fetch_module, "_resolve_hf_token", lambda root: None)
    report = {
        "status": "ok",
        "exit_code": 0,
        "model_dir": "models",
        "cache_dir": "hub",
        "authenticated": False,
        "summary": {"requested": 1, "succeeded": 1, "failed": 0},
        "results": [{"repo_id": TEST_ASSET.repo_id, "status": "already_present", "path": "models/x"}],
    }
    monkeypatch.setattr(fetch_module, "fetch_assets", lambda *args, **kwargs: report)

    result = CliRunner().invoke(app, ["models", "fetch", TEST_ASSET.repo_id, "--json"])

    assert result.exit_code == 0
    assert json.loads(result.stdout) == report
    assert result.stderr == ""


def test_cli_selector_error_still_emits_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("forge.cli_v2_root.setup_cli_logging", lambda **kwargs: None)
    monkeypatch.setattr(fetch_module, "_resolve_hf_token", lambda root: None)

    result = CliRunner().invoke(app, ["models", "fetch", "--json"])

    assert result.exit_code == 2
    assert result.stdout == ""
    payload = json.loads(result.stderr)
    assert set(payload) == {"error"}
    assert "choose exactly one" in payload["error"]


def test_dotenv_token_is_read_without_being_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    (tmp_path / ".env").write_text("HF_TOKEN='hf_private_value'\n", encoding="utf-8")

    assert fetch_module._resolve_hf_token(tmp_path) == "hf_private_value"
