"""Tests for PRD-06: Runtime Export (ONNX, MLX, TensorRT)."""

import sys
import tempfile
import types
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    """Simple model for export testing."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 7)

    def forward(self, images, language_ids=None, **kwargs):
        x = self.conv(images)
        x = self.pool(x).flatten(1)
        actions = self.fc(x)
        return {"actions": actions}


class TinyOnnxModel(nn.Module):
    """Small multimodal model used to check dynamic ONNX inference."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 1)
        self.embedding = nn.Embedding(1000, 4)
        self.head = nn.Linear(8, 7)

    def forward(self, images, language_ids=None, **kwargs):
        image_features = self.conv(images).mean(dim=(2, 3))
        language_features = self.embedding(language_ids).mean(dim=1)
        actions = self.head(torch.cat((image_features, language_features), dim=1))
        return {"actions": actions, "vision_features": image_features}


class TinyStochasticOnnxModel(TinyOnnxModel):
    """Small model whose exported action path contains RandomNormalLike."""

    def forward(self, images, language_ids=None, **kwargs):
        output = super().forward(images, language_ids=language_ids, **kwargs)
        return {"actions": output["actions"] + torch.randn_like(output["actions"])}


class TinyNonFiniteOnnxModel(TinyOnnxModel):
    """Small model that deliberately emits NaN actions."""

    def forward(self, images, language_ids=None, **kwargs):
        output = super().forward(images, language_ids=language_ids, **kwargs)
        return {"actions": output["actions"] * torch.tensor(float("nan"))}


def test_mlx_export():
    """Verify MLX weight export and loading."""
    from forge.export.mlx_export import export_mlx, load_mlx_weights

    model = SimpleModel()

    with tempfile.TemporaryDirectory() as tmpdir:
        export_mlx(model, tmpdir, config={"variant": "test"})

        # Check files exist
        from pathlib import Path

        assert (Path(tmpdir) / "weights.npz").exists()
        assert (Path(tmpdir) / "config.json").exists()
        assert (Path(tmpdir) / "metadata.json").exists()

        # Load and verify
        weights = load_mlx_weights(tmpdir)
        assert len(weights) > 0

        # Check shapes match
        for name, param in model.named_parameters():
            assert name in weights, f"Missing weight: {name}"
            assert weights[name].shape == param.shape, f"Shape mismatch: {name}"


def test_mlx_validate():
    """Verify MLX export validation."""
    from forge.export.mlx_export import export_mlx, validate_mlx_export

    model = SimpleModel()

    with tempfile.TemporaryDirectory() as tmpdir:
        export_mlx(model, tmpdir)
        result = validate_mlx_export(model, tmpdir)

        assert result["status"] == "passed"
        assert result["n_pytorch_params"] == result["n_mlx_params"]
        assert len(result["mismatches"]) == 0


def test_mlx_weight_dtype():
    """Verify MLX weights are saved as float16."""
    from forge.export.mlx_export import export_mlx, load_mlx_weights

    model = SimpleModel()

    with tempfile.TemporaryDirectory() as tmpdir:
        export_mlx(model, tmpdir)
        weights = load_mlx_weights(tmpdir)

        for name, w in weights.items():
            assert w.dtype == np.float16, f"{name} has dtype {w.dtype}, expected float16"


def test_mlx_export_supports_bfloat16_backbones():
    """The v3 frozen language backbones use bf16 on CPU during export."""
    from forge.export.mlx_export import export_mlx, load_mlx_weights, validate_mlx_export

    model = SimpleModel().to(dtype=torch.bfloat16)
    with tempfile.TemporaryDirectory() as tmpdir:
        export_mlx(model, tmpdir)
        weights = load_mlx_weights(tmpdir)
        assert weights["conv.weight"].dtype == np.float16
        assert validate_mlx_export(model, tmpdir)["status"] == "passed"


def test_mlx_export_uses_fast_uncompressed_npz():
    from forge.export.mlx_export import export_mlx

    with tempfile.TemporaryDirectory() as tmpdir:
        export_mlx(SimpleModel(), tmpdir)
        with zipfile.ZipFile(f"{tmpdir}/weights.npz") as archive:
            assert all(item.compress_type == zipfile.ZIP_STORED for item in archive.infolist())


def test_tensorrt_availability_check():
    """Verify TensorRT availability check works."""
    from forge.export.tensorrt_export import check_tensorrt_available

    # On Mac, should return False
    result = check_tensorrt_available()
    assert isinstance(result, bool)
    # We're on Mac, so this should be False
    # But don't hard-assert in case someone runs tests on GPU machine


def test_tensorrt_export_fails_without_cuda_runtime():
    """TensorRT export should report a missing runtime or CUDA hardware."""
    from forge.export.tensorrt_export import check_tensorrt_available

    if not check_tensorrt_available():
        from forge.export.tensorrt_export import export_tensorrt

        with pytest.raises((ImportError, RuntimeError), match="TensorRT"):
            export_tensorrt("fake.onnx", "fake.engine")


def test_tensorrt_parser_preserves_external_data_directory(tmp_path):
    """TensorRT must receive the graph path so relative sidecars can load."""
    from forge.export.tensorrt_export import _parse_onnx_file

    class Parser:
        parsed_path = None

        def parse_from_file(self, path):
            self.parsed_path = path
            return True

    model_path = tmp_path / "nested" / "forge.onnx"
    model_path.parent.mkdir()
    parser = Parser()

    assert _parse_onnx_file(parser, model_path)
    assert parser.parsed_path == str(model_path.resolve())


def test_tensorrt_dynamic_profile_bounds_batch_and_sequence():
    from forge.export.tensorrt_export import _add_dynamic_shape_profile

    class Tensor:
        def __init__(self, name, shape):
            self.name = name
            self.shape = shape

    class Network:
        inputs = [Tensor("images", (-1, 3, 384, 384)), Tensor("language_ids", (-1, -1))]
        num_inputs = len(inputs)

        def get_input(self, index):
            return self.inputs[index]

    class Profile:
        shapes = {}

        def set_shape(self, name, minimum, optimum, maximum):
            self.shapes[name] = (minimum, optimum, maximum)

    class Builder:
        profile = Profile()

        def create_optimization_profile(self):
            return self.profile

    class Config:
        attached = None

        def add_optimization_profile(self, profile):
            self.attached = profile
            return 0

    builder = Builder()
    config = Config()
    _add_dynamic_shape_profile(builder, Network(), config)

    assert config.attached is builder.profile
    assert builder.profile.shapes["images"] == (
        (1, 3, 384, 384),
        (1, 3, 384, 384),
        (4, 3, 384, 384),
    )
    assert builder.profile.shapes["language_ids"] == ((1, 1), (1, 128), (4, 256))


def test_tensorrt_benchmark_rejects_invalid_run_counts(tmp_path):
    from forge.export.tensorrt_export import benchmark_tensorrt_runtime

    with pytest.raises(ValueError, match="run counts"):
        benchmark_tensorrt_runtime(tmp_path / "missing.engine", n_runs=0)


def test_tensorrt_calibration_archive_contains_aligned_runtime_inputs(tmp_path):
    import numpy as np

    from forge.export.tensorrt_export import write_tensorrt_calibration_archive

    images = torch.linspace(-1, 1, 2 * 3 * 384 * 384).reshape(2, 3, 384, 384)
    language_ids = torch.arange(32, dtype=torch.int64).reshape(2, 16)
    path = write_tensorrt_calibration_archive(images, language_ids, tmp_path / "calibration.npz")

    with np.load(path, allow_pickle=False) as payload:
        assert payload["images"].shape == (2, 3, 384, 384)
        assert payload["images"].dtype == np.float32
        assert payload["language_ids"].shape == (2, 16)
        assert payload["language_ids"].dtype == np.int64


def test_pipeline_has_no_int8_to_fp16_success_fallback():
    source = Path("src/forge/pipeline.py").read_text(encoding="utf-8")
    assert "fallback_precision" not in source
    assert "Retrying export with FP16" not in source
    assert "write_tensorrt_calibration_archive" in source


def test_onnx_export_simple():
    """Verify ONNX export produces valid file."""
    from forge.export.onnx_export import export_onnx

    model = SimpleModel()

    with tempfile.TemporaryDirectory() as tmpdir:
        from pathlib import Path

        output = Path(tmpdir) / "model.onnx"
        result = export_onnx(model, output, image_size=32, optimize=False)

        assert result.exists()
        assert result.stat().st_size > 0


def test_onnx_export_dynamic_shapes_runtime_parity():
    """Dynamo export supports new batch/sequence sizes with runtime parity."""
    onnxruntime = pytest.importorskip("onnxruntime")

    from forge.export.onnx_export import export_onnx, validate_onnx

    torch.manual_seed(0)
    model = TinyOnnxModel().eval()

    with tempfile.TemporaryDirectory() as tmpdir:
        from pathlib import Path

        output = Path(tmpdir) / "dynamic_model.onnx"
        result = export_onnx(
            model,
            output,
            image_size=8,
            max_seq_len=5,
            optimize=False,
        )

        session = onnxruntime.InferenceSession(
            str(result),
            providers=["CPUExecutionProvider"],
        )
        assert [output.name for output in session.get_outputs()] == ["actions"]
        images = torch.randn(3, 3, 8, 8)
        language_ids = torch.randint(0, 1000, (3, 7))

        with torch.no_grad():
            expected = model(images, language_ids=language_ids)["actions"].numpy()

        actual = session.run(
            ["actions"],
            {
                "images": images.numpy(),
                "language_ids": language_ids.numpy(),
            },
        )[0]

        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)
        validation = validate_onnx(
            model,
            result,
            n_samples=2,
            images=images[:2],
            language_ids=language_ids[:2],
        )
        assert validation["status"] == "passed"
        assert validation["validation_mode"] == "pointwise_parity"
        assert validation["pointwise_comparable"] is True


def test_onnx_validation_uses_runtime_contract_for_stochastic_action_graph(tmp_path):
    pytest.importorskip("onnxruntime")

    from forge.export.onnx_export import export_onnx, validate_onnx

    model = TinyStochasticOnnxModel().eval()
    artifact = export_onnx(
        model,
        tmp_path / "stochastic.onnx",
        image_size=8,
        max_seq_len=5,
        optimize=False,
    )
    result = validate_onnx(
        model,
        artifact,
        n_samples=2,
        images=torch.randn(2, 3, 8, 8),
        language_ids=torch.randint(0, 1000, (2, 5)),
    )

    assert result["status"] == "passed"
    assert result["validation_mode"] == "stochastic_runtime_contract"
    assert result["pointwise_comparable"] is False
    assert result["max_diff"] is None
    assert result["n_samples"] >= 32
    assert result["checks"] == {
        "output_shape_matches": True,
        "pytorch_actions_finite": True,
        "onnx_actions_finite": True,
        "per_dimension_means_match": True,
        "per_dimension_stds_match": True,
    }


def test_onnx_stochastic_validation_rejects_finite_wrong_distribution(monkeypatch, tmp_path):
    """Finite stochastic output with the wrong scale must fail validation."""
    from forge.export.onnx_export import validate_onnx

    class RandomActionModel(nn.Module):
        def forward(self, images, language_ids=None):
            return {"actions": torch.randn(len(images), 7)}

    class BrokenSession:
        def __init__(self, _path):
            self.rng = np.random.default_rng(20260714)

        def run(self, _outputs, inputs):
            batch = len(inputs["images"])
            return [self.rng.normal(0.0, 20.0, size=(batch, 7)).astype(np.float32)]

    monkeypatch.setitem(sys.modules, "onnxruntime", types.SimpleNamespace(InferenceSession=BrokenSession))
    result = validate_onnx(
        RandomActionModel(),
        tmp_path / "broken.onnx",
        n_samples=4,
        images=torch.zeros(4, 3, 8, 8),
        language_ids=torch.zeros(4, 5, dtype=torch.int64),
    )

    assert result["status"] == "failed"
    assert result["validation_mode"] == "stochastic_runtime_contract"
    assert result["checks"]["per_dimension_stds_match"] is False


def test_onnx_export_promotes_bfloat16_for_cpu_runtime():
    """A v3-style bf16 model must produce an artifact executable on CPU."""
    onnxruntime = pytest.importorskip("onnxruntime")

    from forge.export.onnx_export import export_onnx

    model = TinyOnnxModel().to(dtype=torch.bfloat16).eval()
    with tempfile.TemporaryDirectory() as tmpdir:
        from pathlib import Path

        output = export_onnx(
            model,
            Path(tmpdir) / "bf16_model.onnx",
            image_size=8,
            max_seq_len=5,
            optimize=False,
        )
        session = onnxruntime.InferenceSession(
            str(output),
            providers=["CPUExecutionProvider"],
        )
        actions = session.run(
            ["actions"],
            {
                "images": np.random.randn(1, 3, 8, 8).astype(np.float32),
                "language_ids": np.random.randint(0, 1000, size=(1, 5), dtype=np.int64),
            },
        )[0]

        assert actions.shape == (1, 7)
        assert actions.dtype == np.float32


def test_onnx_runtime_benchmark_records_actual_provider_and_artifact():
    pytest.importorskip("onnxruntime")

    from forge.export.onnx_export import benchmark_onnx_runtime, export_onnx

    model = TinyOnnxModel().eval()
    with tempfile.TemporaryDirectory() as tmpdir:
        from pathlib import Path

        output = export_onnx(
            model,
            Path(tmpdir) / "benchmark.onnx",
            image_size=8,
            max_seq_len=5,
            optimize=False,
        )
        result = benchmark_onnx_runtime(
            output,
            device="cpu",
            n_warmup=1,
            n_runs=2,
            image_size=8,
            sequence_length=5,
        )

        assert result["status"] == "success"
        assert result["provider"] == "CPUExecutionProvider"
        assert result["device"] == "cpu"
        assert result["provider_device_id"] is None
        assert result["measured_runs"] == 2
        assert result["mean_ms"] > 0
        assert result["fps"] > 0
        assert result["actions_finite"] is True
        assert result["actions_shape"] == [1, 7]
        assert result["action_samples"] == 2
        expected_bytes = sum(Path(path).stat().st_size for path in result["artifact_files"])
        assert result["onnx_size_mb"] == expected_bytes / 1e6


def test_onnx_runtime_benchmark_binds_indexed_cuda_provider(tmp_path: Path, monkeypatch):
    ort = pytest.importorskip("onnxruntime")
    from forge.export.onnx_export import benchmark_onnx_runtime

    artifact = tmp_path / "indexed-cuda.onnx"
    artifact.write_bytes(b"placeholder")
    captured: dict[str, object] = {}

    class IndexedCudaSession:
        def __init__(self, _path, **kwargs):
            captured.update(kwargs)

        def get_providers(self):
            return ["CUDAExecutionProvider"]

        def get_inputs(self):
            return []

        def get_outputs(self):
            return [type("Output", (), {"name": "actions"})()]

        def run(self, *_args, **_kwargs):
            return [np.ones((1, 7), dtype=np.float32)]

    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CUDAExecutionProvider"])
    monkeypatch.setattr(ort, "InferenceSession", IndexedCudaSession)

    result = benchmark_onnx_runtime(artifact, device="cuda:2", n_warmup=0, n_runs=1)

    assert captured == {
        "providers": ["CUDAExecutionProvider"],
        "provider_options": [{"device_id": 2}],
    }
    assert result["status"] == "success"
    assert result["device"] == "cuda:2"
    assert result["provider_device_id"] == 2


def test_onnx_runtime_benchmark_ignores_unreferenced_file():
    pytest.importorskip("onnxruntime")

    from forge.export.onnx_export import benchmark_onnx_runtime, export_onnx

    with tempfile.TemporaryDirectory() as tmpdir:
        from pathlib import Path

        output = export_onnx(
            TinyOnnxModel().eval(),
            Path(tmpdir) / "external.onnx",
            image_size=8,
            max_seq_len=5,
            optimize=False,
        )
        from forge.export.onnx_export import _onnx_artifact_files

        expected_files = _onnx_artifact_files(output)
        unrelated = output.with_name("unreferenced.data")
        unrelated.write_bytes(b"not-part-of-the-model")

        result = benchmark_onnx_runtime(
            output,
            device="cpu",
            n_warmup=1,
            n_runs=1,
            image_size=8,
            sequence_length=5,
        )

        expected_bytes = sum(path.stat().st_size for path in expected_files)
        assert result["onnx_size_mb"] == expected_bytes / 1e6
        assert result["artifact_files"] == [str(path) for path in expected_files]


def test_onnx_runtime_benchmark_rejects_non_finite_actions():
    pytest.importorskip("onnxruntime")

    from forge.export.onnx_export import benchmark_onnx_runtime, export_onnx

    with tempfile.TemporaryDirectory() as tmpdir:
        output = export_onnx(
            TinyNonFiniteOnnxModel().eval(),
            Path(tmpdir) / "nonfinite.onnx",
            image_size=8,
            max_seq_len=5,
            optimize=False,
        )
        result = benchmark_onnx_runtime(
            output,
            device="cpu",
            n_warmup=0,
            n_runs=1,
            image_size=8,
            sequence_length=5,
        )

    assert result["status"] == "failed"
    assert "non-finite" in result["reason"]


def test_onnx_runtime_benchmark_rejects_missing_actions(tmp_path: Path, monkeypatch):
    ort = pytest.importorskip("onnxruntime")
    from forge.export.onnx_export import benchmark_onnx_runtime

    artifact = tmp_path / "missing-actions.onnx"
    artifact.write_bytes(b"placeholder")

    class MissingActionsSession:
        def __init__(self, *_args, **_kwargs):
            pass

        def get_providers(self):
            return ["CPUExecutionProvider"]

        def get_inputs(self):
            return []

        def get_outputs(self):
            return [type("Output", (), {"name": "features"})()]

    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
    monkeypatch.setattr(ort, "InferenceSession", MissingActionsSession)

    result = benchmark_onnx_runtime(artifact, device="cpu", n_warmup=0, n_runs=1)

    assert result == {"status": "failed", "reason": "ONNX runtime output does not contain actions"}


def test_onnx_runtime_benchmark_rejects_empty_actions(tmp_path: Path, monkeypatch):
    ort = pytest.importorskip("onnxruntime")
    from forge.export.onnx_export import benchmark_onnx_runtime

    artifact = tmp_path / "empty-actions.onnx"
    artifact.write_bytes(b"placeholder")

    class EmptyActionsSession:
        def __init__(self, *_args, **_kwargs):
            pass

        def get_providers(self):
            return ["CPUExecutionProvider"]

        def get_inputs(self):
            return []

        def get_outputs(self):
            return [type("Output", (), {"name": "actions"})()]

        def run(self, *_args, **_kwargs):
            return [np.empty((0, 7), dtype=np.float32)]

    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
    monkeypatch.setattr(ort, "InferenceSession", EmptyActionsSession)

    result = benchmark_onnx_runtime(artifact, device="cpu", n_warmup=0, n_runs=1)

    assert result["status"] == "failed"
    assert "empty or non-finite" in result["reason"]
