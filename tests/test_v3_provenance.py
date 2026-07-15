"""Focused tests for the v3 real-weights guarantee."""

from __future__ import annotations

from pathlib import Path

import pytest

from forge.config import ForgeConfig, StudentConfig
from forge.errors import ForgeDataNotFoundError, ForgeModelNotFoundError
from forge.student import FORGEStudent, MockLanguageModel, MockVisionEncoder


def _tiny_student_config(*, allow_mock: bool) -> StudentConfig:
    return StudentConfig(
        allow_mock=allow_mock,
        autosense=False,
        bridge_d_vision=16,
        bridge_d_model=16,
        bridge_n_queries=2,
        bridge_n_heads=2,
        bridge_n_layers=1,
        action_head_layers=1,
        action_diffusion_steps=2,
    )


def test_student_config_disallows_mock_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FORGE_ALLOW_MOCK", raising=False)
    assert StudentConfig().allow_mock is False


@pytest.mark.parametrize("value", ["1", "true", "YES", "on"])
def test_student_config_honors_allow_mock_env(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv("FORGE_ALLOW_MOCK", value)
    assert StudentConfig().allow_mock is True


def test_missing_vision_weights_raise_actionable_error(tmp_path: Path) -> None:
    config = _tiny_student_config(allow_mock=False)
    expected = tmp_path / "google--siglip2-so400m-patch14-384"

    with pytest.raises(ForgeModelNotFoundError) as exc_info:
        FORGEStudent(config, model_dir=tmp_path)

    message = str(exc_info.value)
    assert str(expected) in message
    assert "forge models fetch google/siglip2-so400m-patch14-384" in message
    assert "forge doctor" in message


def test_allow_mock_exposes_component_provenance(tmp_path: Path) -> None:
    student = FORGEStudent(_tiny_student_config(allow_mock=True), model_dir=tmp_path)

    assert isinstance(student.vision_encoder, MockVisionEncoder)
    assert isinstance(student.language, MockLanguageModel)
    assert student.vision_provenance == "mock"
    assert student.language_provenance == "mock"
    assert student.component_provenance == {"vision": "mock", "language": "mock"}


def test_missing_language_weights_raise_actionable_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _tiny_student_config(allow_mock=False)

    def fake_vision(self: FORGEStudent) -> MockVisionEncoder:
        self.vision_provenance = "real"
        return MockVisionEncoder(self.config.bridge_d_vision)

    monkeypatch.setattr(FORGEStudent, "_load_vision_encoder", fake_vision)
    expected = tmp_path / "Qwen--Qwen3-0.6B"

    with pytest.raises(ForgeModelNotFoundError) as exc_info:
        FORGEStudent(config, model_dir=tmp_path)

    message = str(exc_info.value)
    assert str(expected) in message
    assert "forge models fetch Qwen/Qwen3-0.6B" in message
    assert "forge doctor" in message


def test_broken_vision_weights_raise_instead_of_falling_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transformers

    config = _tiny_student_config(allow_mock=False)
    expected = tmp_path / "google--siglip2-so400m-patch14-384"
    expected.mkdir()

    def fail_load(*args: object, **kwargs: object) -> object:
        raise OSError("corrupt vision weights")

    monkeypatch.setattr(transformers.SiglipVisionModel, "from_pretrained", fail_load)
    monkeypatch.setattr(transformers.SiglipModel, "from_pretrained", fail_load)

    with pytest.raises(ForgeModelNotFoundError) as exc_info:
        FORGEStudent(config, model_dir=tmp_path)

    assert str(expected) in str(exc_info.value)
    assert "corrupt vision weights" in str(exc_info.value)


def test_broken_language_weights_raise_instead_of_falling_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transformers

    config = _tiny_student_config(allow_mock=False)
    expected = tmp_path / "Qwen--Qwen3-0.6B"
    expected.mkdir()

    def fake_vision(self: FORGEStudent) -> MockVisionEncoder:
        self.vision_provenance = "real"
        return MockVisionEncoder(self.config.bridge_d_vision)

    def fail_load(*args: object, **kwargs: object) -> object:
        raise OSError("corrupt language weights")

    monkeypatch.setattr(FORGEStudent, "_load_vision_encoder", fake_vision)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", fail_load)

    with pytest.raises(ForgeModelNotFoundError) as exc_info:
        FORGEStudent(config, model_dir=tmp_path)

    assert str(expected) in str(exc_info.value)
    assert "corrupt language weights" in str(exc_info.value)


def test_missing_teacher_labels_hard_fail_without_mock(tmp_path: Path) -> None:
    from forge.distill import train_forge

    config = ForgeConfig.default()
    config.student.allow_mock = False
    config.paths.model_dir = str(tmp_path / "models")
    config.paths.data_dir = str(tmp_path / "data")
    config.paths.output_dir = str(tmp_path / "outputs")
    for model_id in (config.student.vision_encoder, config.student.language_model):
        (Path(config.paths.model_dir) / model_id.replace("/", "--")).mkdir(parents=True)

    with pytest.raises(ForgeDataNotFoundError, match="forge pipeline --stage labels"):
        train_forge(config, device="cpu", max_steps=1)

    assert not (tmp_path / "data" / "teacher_labels").exists()
    assert not (tmp_path / "outputs").exists()
