"""End-to-end contracts for the PRD-36 real-weights guarantee."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml
from torch.optim import AdamW
from typer.testing import CliRunner

from forge.cli_commands.eval import eval_app
from forge.config import ForgeConfig
from forge.distill import _save_checkpoint
from forge.pipeline import run_pipeline
from forge.provenance import build_provenance


def _provenance(*, mock: bool, model_dir: Path) -> dict[str, str]:
    status = "mock" if mock else "real"
    return build_provenance(
        vision=status,
        language=status,
        labels=status,
        model_dir=model_dir,
        git_sha="abc123",
        forge_version="3.0.0-test",
        torch_version=str(torch.__version__),
    )


def test_pipeline_missing_backbone_exits_two_without_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from forge.cli_v2 import app

    monkeypatch.delenv("FORGE_ALLOW_MOCK", raising=False)
    labels_dir = tmp_path / "data" / "teacher_labels"
    labels_dir.mkdir(parents=True)
    (labels_dir / "metadata.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr("forge.distill.TeacherLabelDataset", lambda *_args, **_kwargs: object())

    config_path = tmp_path / "strict.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "paths": {
                    "model_dir": str(tmp_path / "missing-models"),
                    "data_dir": str(tmp_path / "data"),
                },
                "student": {"allow_mock": False, "autosense": False},
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "outputs"

    result = CliRunner().invoke(
        app,
        [
            "pipeline",
            "--config",
            str(config_path),
            "--stage",
            "distill",
            "--skip-labels",
            "--device",
            "cpu",
            "--max-steps",
            "1",
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 2
    assert "forge models fetch google/siglip2-so400m-patch14-384" in result.output
    assert "forge doctor" in result.output
    assert not output_dir.exists()


def test_distill_checkpoint_contains_complete_provenance(tmp_path: Path) -> None:
    model = torch.nn.Linear(3, 2)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    provenance = _provenance(mock=True, model_dir=tmp_path / "models")
    checkpoint = tmp_path / "checkpoint.pt"

    _save_checkpoint(  # type: ignore[arg-type]
        checkpoint,
        model,
        optimizer,
        scheduler,
        7,
        provenance,
        ForgeConfig.default(),
    )

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert payload["step"] == 7
    assert payload["provenance"] == provenance
    assert payload["student_config"]["language_model"] == "Qwen/Qwen3-0.6B"
    assert set(payload["provenance"]) == {
        "vision",
        "language",
        "labels",
        "model_dir",
        "git_sha",
        "forge_version",
        "torch_version",
    }


def test_pipeline_summary_reuses_checkpoint_provenance(tmp_path: Path, monkeypatch) -> None:
    provenance = _provenance(mock=False, model_dir=tmp_path / "models")

    def fake_train(*_args, **kwargs):
        checkpoint = Path(kwargs["checkpoint_dir"]) / "checkpoints" / "final.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"final")
        return {
            "status": "success",
            "final_loss": 0.25,
            "provenance": provenance,
        }

    monkeypatch.setattr("forge.distill.train_forge", fake_train)
    config = ForgeConfig.default()
    config.paths.output_dir = str(tmp_path / "outputs")

    result = run_pipeline(config, device="cpu", stage="distill", max_distill_steps=1)

    summary = json.loads(Path(result["pipeline_summary_path"]).read_text(encoding="utf-8"))
    assert result["provenance"] == provenance
    assert summary["provenance"] == provenance


def test_eval_serve_refuses_mock_checkpoint_without_opt_in(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("FORGE_ALLOW_MOCK", raising=False)
    checkpoint = tmp_path / "mock.pt"
    torch.save(
        {
            "model_state_dict": {},
            "provenance": _provenance(mock=True, model_dir=tmp_path / "models"),
        },
        checkpoint,
    )

    result = CliRunner().invoke(
        eval_app,
        ["serve", "--checkpoint", str(checkpoint), "--device", "cpu"],
    )

    assert result.exit_code == 2
    assert "Refusing to eval a mock-derived checkpoint" in result.output
    assert "--allow-mock" in result.output


def test_eval_serve_allows_mock_checkpoint_with_explicit_opt_in(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from forge.eval.model_server import ForgeModelServer

    monkeypatch.delenv("FORGE_ALLOW_MOCK", raising=False)
    checkpoint = tmp_path / "mock.pt"
    torch.save(
        {
            "model_state_dict": {},
            "provenance": _provenance(mock=True, model_dir=tmp_path / "models"),
        },
        checkpoint,
    )
    started: list[bool] = []
    monkeypatch.setattr(
        ForgeModelServer,
        "start",
        lambda self, **_kwargs: started.append(self.config.allow_mock),
    )

    result = CliRunner().invoke(
        eval_app,
        [
            "serve",
            "--checkpoint",
            str(checkpoint),
            "--device",
            "cpu",
            "--allow-mock",
        ],
    )

    assert result.exit_code == 0, result.output
    assert started == [True]
