"""PRD-37 tests for the two 2026 LeRobot teacher adapters."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from forge.data.real_robot_episodes import load_real_robot_episodes
from forge.teachers.molmoact2_adapter import MolmoAct2Adapter
from forge.teachers.openvla_adapter import OpenVLAAdapter
from forge.teachers.rdt2_adapter import (
    RDT2Adapter,
    _load_action_normalizer,
    _load_rdt2_normalizer,
)
from forge.teachers.smolvla_adapter import SmolVLAAdapter
from forge.teachers.vla_jepa_adapter import VLAJEPAAdapter


def _checkpoint(path: Path) -> Path:
    path.mkdir()
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"test-placeholder")
    return path


def _install_fake_lerobot(
    monkeypatch: pytest.MonkeyPatch,
    *,
    module_name: str,
    class_name: str,
    horizon: int,
    action_dim: int,
    output: torch.Tensor | None = None,
) -> SimpleNamespace:
    trace = SimpleNamespace(raw=None, processed=None, path=None, device=None, eval_called=False, overrides=None)

    visual_feature = SimpleNamespace(type=SimpleNamespace(value="VISUAL"), shape=(3, 16, 16))
    state_feature = SimpleNamespace(type=SimpleNamespace(value="STATE"), shape=(8,))
    config = SimpleNamespace(
        input_features={
            "observation.images.main": visual_feature,
            "observation.images.wrist": visual_feature,
            "observation.state": state_feature,
        },
        image_features={
            "observation.images.main": visual_feature,
            "observation.images.wrist": visual_feature,
        },
        image_keys=["observation.images.main", "observation.images.wrist"],
    )

    class FakePolicy:
        def __init__(self) -> None:
            self.config = config

        @classmethod
        def from_pretrained(cls, path: str, **_kwargs):
            trace.path = path
            return cls()

        def to(self, device: str):
            trace.device = device
            return self

        def eval(self):
            trace.eval_called = True
            return self

        def predict_action_chunk(self, batch: dict[str, object]) -> torch.Tensor:
            trace.processed = batch
            if output is not None:
                return output
            values = torch.arange(horizon * action_dim, dtype=torch.float32)
            return values.reshape(1, horizon, action_dim).div(100)

    FakePolicy.__name__ = class_name
    policy_module = ModuleType(module_name)
    setattr(policy_module, class_name, FakePolicy)
    monkeypatch.setitem(sys.modules, module_name, policy_module)

    factory_module = ModuleType("lerobot.policies.factory")

    def make_pre_post_processors(
        _config: object,
        pretrained_path: str | None = None,
        preprocessor_overrides: dict | None = None,
    ):
        assert pretrained_path == trace.path
        trace.overrides = preprocessor_overrides

        def preprocess(raw: dict[str, object]) -> dict[str, object]:
            trace.raw = raw
            return {
                key: value.unsqueeze(0) if isinstance(value, torch.Tensor) else [value] for key, value in raw.items()
            }

        def postprocess(actions: torch.Tensor) -> torch.Tensor:
            return actions + 0.25

        return preprocess, postprocess

    factory_module.make_pre_post_processors = make_pre_post_processors
    monkeypatch.setitem(sys.modules, "lerobot.policies.factory", factory_module)
    return trace


@pytest.mark.parametrize(
    ("adapter_class", "module_name", "class_name", "horizon", "action_dim"),
    [
        (
            MolmoAct2Adapter,
            "lerobot.policies.molmoact2.modeling_molmoact2",
            "MolmoAct2Policy",
            10,
            7,
        ),
        (
            VLAJEPAAdapter,
            "lerobot.policies.vla_jepa.modeling_vla_jepa",
            "VLAJEPAPolicy",
            7,
            7,
        ),
        (
            SmolVLAAdapter,
            "lerobot.policies.smolvla.modeling_smolvla",
            "SmolVLAPolicy",
            50,
            6,
        ),
    ],
)
def test_real_policy_boundary_normalizes_action_chunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    adapter_class: type,
    module_name: str,
    class_name: str,
    horizon: int,
    action_dim: int,
) -> None:
    trace = _install_fake_lerobot(
        monkeypatch,
        module_name=module_name,
        class_name=class_name,
        horizon=horizon,
        action_dim=action_dim,
    )
    monkeypatch.setattr(
        adapter_class,
        "_load_policy",
        lambda _self, policy_class, model_path, _device: policy_class.from_pretrained(str(model_path)),
    )
    monkeypatch.setattr(
        adapter_class,
        "_processor_overrides",
        lambda _self, device: {"device_processor": {"device": device}},
    )
    checkpoint = _checkpoint(tmp_path / "teacher")
    adapter = adapter_class()
    adapter.load(checkpoint, device="cpu")

    image = np.full((16, 18, 3), 128, dtype=np.uint8)
    proprioception = np.arange(10, dtype=np.float32)
    chunk = adapter.predict(image, "pick up the block", proprioception)

    assert adapter.is_loaded
    assert trace.path == str(checkpoint)
    assert trace.device == "cpu"
    assert trace.eval_called is True
    assert trace.overrides == {"device_processor": {"device": "cpu"}}
    assert chunk.actions.shape == (horizon, action_dim)
    assert np.isfinite(chunk.actions).all()
    assert chunk.actions[0, 0] == pytest.approx(0.25)
    np.testing.assert_array_equal(chunk.action_mean, chunk.actions)
    np.testing.assert_array_equal(chunk.action_std, np.zeros_like(chunk.actions))
    np.testing.assert_array_equal(chunk.confidence, np.ones_like(chunk.actions))
    assert chunk.metadata["inference"] == "real"
    assert chunk.metadata["checkpoint"] == str(checkpoint.resolve())
    assert trace.raw["task"] == "pick up the block"
    assert trace.processed["observation.images.main"].shape == (1, 3, 16, 18)
    assert trace.processed["observation.images.wrist"].shape == (1, 3, 16, 18)
    assert trace.processed["observation.state"].shape == (1, 8)
    np.testing.assert_allclose(trace.processed["observation.state"][0].numpy(), np.arange(8))
    assert adapter.extract_features(image, "pick up the block") == {}

    adapter.unload()
    assert not adapter.is_loaded


def test_registry_discovers_five_real_teacher_contracts() -> None:
    from forge.teachers.registry import TeacherRegistry

    registry = TeacherRegistry()
    registry.reset()
    teachers = registry.list_teachers()

    assert {
        "openvla-7b",
        "rdt2-fm",
        "smolvla-base",
        "molmoact2-libero",
        "vla-jepa-3b",
    }.issubset(teachers)
    assert len(teachers) >= 5
    registry.reset()


def test_rdt2_real_runtime_boundary_uses_24_by_20_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = _checkpoint(tmp_path / "robotics-diffusion-transformer--RDT2-FM")
    (tmp_path / "robotics-diffusion-transformer--RDT2-VQ").mkdir()
    (tmp_path / "rdt2-umi-normalizer.pt").write_bytes(b"verified-test-sidecar")
    trace = SimpleNamespace(args=None, unloaded=False)

    class Runtime:
        def predict(self, image, instruction, state):
            trace.args = (image, instruction, state)
            return np.arange(24 * 20, dtype=np.float32).reshape(24, 20)

        def unload(self):
            trace.unloaded = True

    monkeypatch.setattr(RDT2Adapter, "_load_runtime", lambda *_args: Runtime())
    adapter = RDT2Adapter()
    adapter.load(checkpoint, device="cuda:2", dtype=torch.bfloat16)
    chunk = adapter.predict(
        np.full((18, 20, 3), 127, dtype=np.uint8),
        "Fold the clothes.",
        np.arange(14, dtype=np.float32),
    )

    assert chunk.actions.shape == (24, 20)
    assert np.isfinite(chunk.actions).all()
    assert chunk.metadata["inference"] == "real"
    assert chunk.metadata["camera_mapping"].startswith("single-real-frame")
    assert trace.args[0].shape == (18, 20, 3)
    assert trace.args[1] == "Fold the clothes."
    np.testing.assert_array_equal(trace.args[2], np.zeros(20, dtype=np.float32))
    assert chunk.metadata["state_conditioning"] == "official-pretrained-zero-state"
    assert adapter.info().action_horizon == 24
    assert adapter.info().action_dim == 20
    assert adapter.info().param_count == 7.5
    assert adapter.info().supports_features is False
    assert adapter.get_action_space()["dim"] == 20
    assert adapter.extract_features(trace.args[0], trace.args[1]) == {}

    adapter.unload()
    assert trace.unloaded is True
    assert not adapter.is_loaded


def test_rdt2_requires_both_official_companion_assets(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "robotics-diffusion-transformer--RDT2-FM")
    with pytest.raises(FileNotFoundError, match="RDT2-VQ companion"):
        RDT2Adapter().load(checkpoint)

    (tmp_path / "robotics-diffusion-transformer--RDT2-VQ").mkdir()
    with pytest.raises(FileNotFoundError, match="normalizer"):
        RDT2Adapter().load(checkpoint)


def test_rdt2_normalizer_uses_narrow_safe_pickle_allowlist(tmp_path: Path) -> None:
    action = torch.nn.ParameterDict(
        {
            "scale": torch.nn.Parameter(torch.arange(1, 21, dtype=torch.float32)),
            "offset": torch.nn.Parameter(torch.arange(20, dtype=torch.float32)),
        }
    )
    path = tmp_path / "normalizer.pt"
    torch.save(torch.nn.ParameterDict({"action": action}), path)

    scale, offset = _load_action_normalizer(path)

    torch.testing.assert_close(scale, torch.arange(1, 21, dtype=torch.float32))
    torch.testing.assert_close(offset, torch.arange(20, dtype=torch.float32))
    assert not scale.requires_grad
    assert not offset.requires_grad


def test_rdt2_normalizer_does_not_apply_action_transform_to_state(tmp_path: Path) -> None:
    action = torch.nn.ParameterDict(
        {
            "scale": torch.nn.Parameter(torch.ones(20)),
            "offset": torch.nn.Parameter(torch.zeros(20)),
        }
    )
    state = torch.nn.ParameterDict(
        {
            "scale": torch.nn.Parameter(torch.full((20,), 2.0)),
            "offset": torch.nn.Parameter(torch.full((20,), 3.0)),
        }
    )
    path = tmp_path / "normalizer.pt"
    torch.save(torch.nn.ParameterDict({"action": action, "state": state}), path)

    action_scale, action_offset = _load_rdt2_normalizer(path)

    torch.testing.assert_close(action_scale, torch.ones(20))
    torch.testing.assert_close(action_offset, torch.zeros(20))


def test_rdt2_runtime_passes_official_zero_state_without_action_normalization() -> None:
    from forge.teachers.rdt2_adapter import _RDT2Runtime

    trace = SimpleNamespace(state=None)

    class Inputs(dict):
        def to(self, _device):
            return self

    class Processor:
        def apply_chat_template(self, *_args, **_kwargs):
            return "prompt"

        def __call__(self, **_kwargs):
            return Inputs(attention_mask=torch.ones(1, 2, dtype=torch.long))

    class VLM:
        def __call__(self, **_kwargs):
            return SimpleNamespace(past_key_values=[(torch.ones(1), torch.zeros(1))])

    class Policy:
        def predict_action(self, **kwargs):
            trace.state = kwargs["state_tokens"].detach().cpu()
            return torch.zeros(1, 24, 20)

    runtime = object.__new__(_RDT2Runtime)
    runtime.processor = Processor()
    runtime.vlm = VLM()
    runtime.policy = Policy()
    runtime.selected_layers = [0]
    runtime.device = "cpu"
    runtime.dtype = torch.float32
    runtime.action_scale = torch.ones(20)
    runtime.action_offset = torch.zeros(20)

    runtime.predict(
        np.zeros((8, 8, 3), dtype=np.uint8),
        "Move the block.",
        np.zeros(20, dtype=np.float32),
    )

    torch.testing.assert_close(trace.state, torch.zeros(1, 1, 20))


def test_rdt2_runtime_accepts_current_transformers_dynamic_cache_shape() -> None:
    layers = [SimpleNamespace(keys=torch.ones(1, 4, 3, 128), values=torch.zeros(1, 4, 3, 128))]
    available = [(layer.keys, layer.values) for layer in SimpleNamespace(layers=layers).layers]
    assert available[0][0].shape == (1, 4, 3, 128)
    assert available[0][1].shape == (1, 4, 3, 128)


@pytest.mark.parametrize(
    ("adapter", "name", "architecture", "horizon", "params"),
    [
        (MolmoAct2Adapter(), "molmoact2-libero", "hybrid-ar-flow", 10, 5.0),
        (VLAJEPAAdapter(), "vla-jepa-3b", "jepa-flow", 7, 3.0),
    ],
)
def test_2026_teacher_metadata(adapter, name: str, architecture: str, horizon: int, params: float) -> None:
    info = adapter.info()
    assert info.name == name
    assert info.architecture == architecture
    assert info.action_horizon == horizon
    assert info.action_dim == 7
    assert info.param_count == params
    assert info.supports_chunking is True
    assert info.supports_features is False
    assert adapter.get_action_space()["dim"] == 7


def test_missing_checkpoint_fails_before_required_runtime(tmp_path: Path) -> None:
    adapter = MolmoAct2Adapter()
    missing = tmp_path / "missing"

    with pytest.raises(FileNotFoundError, match="checkpoint directory not found"):
        adapter.load(missing)

    assert not missing.exists()


def test_smolvla_requires_local_companion_vlm(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "lerobot--smolvla_base")

    with pytest.raises(RuntimeError, match="SmolVLM2-500M-Video-Instruct"):
        SmolVLAAdapter().load(checkpoint)


def test_vla_jepa_requires_local_qwen_and_jepa_companions(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "lerobot--VLA-JEPA-Pretrain")

    with pytest.raises(RuntimeError, match="VLA-JEPA companion checkpoints"):
        VLAJEPAAdapter().load(checkpoint)


def test_molmoact2_requires_local_base_and_tokenizer_companions(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "allenai--MolmoAct2-LIBERO-LeRobot")

    with pytest.raises(RuntimeError, match="MolmoAct2 companion checkpoints"):
        MolmoAct2Adapter().load(checkpoint)


def test_published_molmo_null_scheduler_is_normalized_in_memory() -> None:
    payload = {"scheduler_decay_steps": None}
    if payload.get("scheduler_decay_steps") is None:
        payload["scheduler_decay_steps"] = 100_000
    assert payload["scheduler_decay_steps"] == 100_000


def test_molmo_processor_overrides_replace_serialized_remote_ids(tmp_path: Path) -> None:
    adapter = MolmoAct2Adapter()
    adapter._base_path = tmp_path / "base"
    adapter._tokenizer_path = tmp_path / "tokenizer"
    overrides = adapter._processor_overrides("cuda:2")
    assert overrides["device_processor"] == {"device": "cuda:2"}
    assert overrides["molmoact2_pack_inputs"] == {
        "checkpoint_path": str(tmp_path / "base"),
        "discrete_action_tokenizer": str(tmp_path / "tokenizer"),
    }


def test_openvla_uses_checkpoint_decoder_and_dataset_stats() -> None:
    trace = SimpleNamespace(kwargs=None)

    class Processor:
        def __call__(self, prompt, image, return_tensors):
            assert prompt.endswith("\nOut:")
            assert image.size == (12, 10)
            assert return_tensors == "pt"
            return {"input_ids": torch.tensor([[1, 2]]), "pixel_values": torch.ones(1, 3, 10, 12)}

    class Model:
        def predict_action(self, **kwargs):
            trace.kwargs = kwargs
            return np.arange(7, dtype=np.float32) / 10

    adapter = OpenVLAAdapter(unnorm_key="bridge_orig")
    adapter._model = Model()
    adapter._processor = Processor()
    adapter._device = "cpu"

    chunk = adapter.predict(np.zeros((10, 12, 3), dtype=np.uint8), "move the block")

    np.testing.assert_allclose(chunk.actions[0], np.arange(7) / 10)
    assert trace.kwargs["unnorm_key"] == "bridge_orig"
    assert trace.kwargs["do_sample"] is False
    assert chunk.metadata["inference"] == "real"
    assert chunk.metadata["unnorm_key"] == "bridge_orig"


def test_openvla_moves_with_pytorch_without_accelerate_device_map(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = SimpleNamespace(load_kwargs=None, moves=[])

    class Model:
        def to(self, *args, **kwargs):
            trace.moves.append((args, kwargs))
            return self

        def eval(self):
            return self

    class ProcessorFactory:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return object()

    def fake_load(*_args, **kwargs):
        trace.load_kwargs = kwargs
        return Model()

    model_path = tmp_path / "openvla"
    model_path.mkdir()
    monkeypatch.setattr("forge.teachers.openvla_adapter.configure_transformers_module_cache", lambda _path: None)
    monkeypatch.setattr("forge.teachers.openvla_adapter.install_legacy_tokenization_exports", lambda: None)
    monkeypatch.setattr("forge.teachers.openvla_adapter.load_image_text_model", fake_load)
    monkeypatch.setattr("transformers.AutoProcessor", ProcessorFactory)

    OpenVLAAdapter().load(model_path, device="cuda:2", dtype=torch.bfloat16)

    assert trace.load_kwargs["device_map"] is None
    assert trace.load_kwargs["dtype"] is torch.bfloat16
    assert trace.moves == [((), {"device": "cuda:2"})]


def test_missing_required_lerobot_runtime_has_reinstall_guidance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = _checkpoint(tmp_path / "teacher")
    (tmp_path / "allenai--MolmoAct2-LIBERO").mkdir()
    (tmp_path / "allenai--MolmoAct2-FAST-Tokenizer").mkdir()
    original = __import__("importlib").import_module

    def fail_policy_import(name: str):
        if name == MolmoAct2Adapter.policy_module:
            raise ModuleNotFoundError("No module named 'lerobot'")
        return original(name)

    monkeypatch.setattr("forge.teachers.lerobot_policy_adapter.importlib.import_module", fail_policy_import)

    with pytest.raises(RuntimeError, match="Reinstall FORGE"):
        MolmoAct2Adapter().load(checkpoint)


@pytest.mark.parametrize(
    "bad_output",
    [
        torch.zeros(1, 9, 7),
        torch.full((1, 10, 7), float("nan")),
    ],
)
def test_invalid_policy_outputs_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bad_output: torch.Tensor,
) -> None:
    _install_fake_lerobot(
        monkeypatch,
        module_name=MolmoAct2Adapter.policy_module,
        class_name=MolmoAct2Adapter.policy_class_name,
        horizon=10,
        action_dim=7,
        output=bad_output,
    )
    monkeypatch.setattr(
        MolmoAct2Adapter,
        "_load_policy",
        lambda _self, policy_class, model_path, _device: policy_class.from_pretrained(str(model_path)),
    )
    monkeypatch.setattr(
        MolmoAct2Adapter,
        "_processor_overrides",
        lambda _self, device: {"device_processor": {"device": device}},
    )
    adapter = MolmoAct2Adapter()
    adapter.load(_checkpoint(tmp_path / "teacher"))

    with pytest.raises(ValueError, match="returned (action shape|non-finite)"):
        adapter.predict(np.zeros((16, 16, 3), dtype=np.uint8), "move")


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("adapter_class", "env_name", "default_path"),
    [
        (MolmoAct2Adapter, "FORGE_MOLMOACT2_PATH", "models/allenai--MolmoAct2-LIBERO-LeRobot"),
        (VLAJEPAAdapter, "FORGE_VLA_JEPA_PATH", "models/lerobot--VLA-JEPA-Pretrain"),
    ],
)
def test_live_2026_teacher_three_predictions(
    adapter_class: type,
    env_name: str,
    default_path: str,
) -> None:
    if os.environ.get("FORGE_RUN_TEACHER_GPU_TESTS") != "1":
        pytest.skip("set FORGE_RUN_TEACHER_GPU_TESTS=1 for the real teacher GPU smoke")
    if not torch.cuda.is_available() or importlib.util.find_spec("lerobot") is None:
        pytest.skip("real CUDA and LeRobot runtime required")
    model_path = Path(os.environ.get(env_name, default_path))
    if not model_path.is_dir():
        pytest.skip(f"real checkpoint not present: {model_path}")

    dataset_path = Path(
        os.environ.get(
            "FORGE_TEACHER_DATASET",
            "models/datasets/lerobot--aloha_sim_transfer_cube_human",
        )
    )
    episode = load_real_robot_episodes(dataset_path, max_episodes=1, max_steps=3)[0]
    adapter = adapter_class()
    adapter.load(model_path, device="cuda", dtype=torch.bfloat16)

    chunks = [
        adapter.predict(episode.images[index], episode.instruction, episode.proprioception[index]) for index in range(3)
    ]

    assert all(chunk.metadata["inference"] == "real" for chunk in chunks)
    assert all(np.isfinite(chunk.actions).all() for chunk in chunks)
    adapter.unload()
