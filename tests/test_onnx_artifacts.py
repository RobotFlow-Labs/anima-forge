"""Fail-closed ONNX external-data family contracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import pytest

from forge.export.onnx_artifacts import resolve_onnx_artifact_family


def _external_tensor(name: str, location: str) -> onnx.TensorProto:
    tensor = onnx.numpy_helper.from_array(np.ones((1,), dtype=np.float32), name=name)
    tensor.ClearField("raw_data")
    tensor.data_location = onnx.TensorProto.EXTERNAL
    entry = tensor.external_data.add()
    entry.key = "location"
    entry.value = location
    return tensor


def _write_model(path: Path, tensor: onnx.TensorProto) -> None:
    graph = onnx.helper.make_graph([], "external-family", [], [], [tensor])
    onnx.save_model(onnx.helper.make_model(graph), path)


def test_resolver_returns_arbitrary_referenced_external_filename(tmp_path: Path) -> None:
    graph = tmp_path / "forge.onnx"
    weights = tmp_path / "weights.bin"
    weights.write_bytes(b"weights")
    _write_model(graph, _external_tensor("weight", weights.name))

    assert resolve_onnx_artifact_family(graph) == {
        graph.name: graph.resolve(),
        weights.name: weights.resolve(),
    }


@pytest.mark.parametrize(
    "location",
    ["../weights.bin", "/tmp/weights.bin", "C:/weights.bin", "nested/../weights.bin"],
)
def test_resolver_rejects_unsafe_external_locations(tmp_path: Path, location: str) -> None:
    graph = tmp_path / "forge.onnx"
    _write_model(graph, _external_tensor("weight", location))

    with pytest.raises(ValueError, match="location is unsafe"):
        resolve_onnx_artifact_family(graph)


def test_resolver_rejects_symlink_escape(tmp_path: Path) -> None:
    graph_dir = tmp_path / "graph"
    graph_dir.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"weights")
    (graph_dir / "weights.bin").symlink_to(outside)
    graph = graph_dir / "forge.onnx"
    _write_model(graph, _external_tensor("weight", "weights.bin"))

    with pytest.raises(ValueError, match="escapes the graph directory"):
        resolve_onnx_artifact_family(graph)


def test_resolver_rejects_missing_external_file(tmp_path: Path) -> None:
    graph = tmp_path / "forge.onnx"
    _write_model(graph, _external_tensor("weight", "missing.bin"))

    with pytest.raises(ValueError, match="missing or not a file"):
        resolve_onnx_artifact_family(graph)


def test_resolver_rejects_duplicate_location_entries(tmp_path: Path) -> None:
    graph = tmp_path / "forge.onnx"
    weights = tmp_path / "weights.bin"
    weights.write_bytes(b"weights")
    tensor = _external_tensor("weight", weights.name)
    duplicate = tensor.external_data.add()
    duplicate.key = "location"
    duplicate.value = weights.name
    _write_model(graph, tensor)

    with pytest.raises(ValueError, match="duplicate external-data locations"):
        resolve_onnx_artifact_family(graph)


def test_resolver_rejects_graph_self_reference(tmp_path: Path) -> None:
    graph = tmp_path / "forge.onnx"
    _write_model(graph, _external_tensor("weight", graph.name))

    with pytest.raises(ValueError, match="aliases the graph itself"):
        resolve_onnx_artifact_family(graph)


def test_resolver_finds_nested_graph_and_sparse_external_tensors(tmp_path: Path) -> None:
    nested_path = tmp_path / "nested.bin"
    sparse_values_path = tmp_path / "sparse-values.bin"
    sparse_indices_path = tmp_path / "sparse-indices.bin"
    for path in (nested_path, sparse_values_path, sparse_indices_path):
        path.write_bytes(path.name.encode())

    nested_graph = onnx.helper.make_graph(
        [],
        "nested",
        [],
        [],
        [_external_tensor("nested", nested_path.name)],
    )
    node = onnx.helper.make_node("If", ["condition"], ["output"], then_branch=nested_graph, else_branch=nested_graph)
    sparse = onnx.SparseTensorProto()
    sparse.values.CopyFrom(_external_tensor("values", sparse_values_path.name))
    sparse.indices.CopyFrom(_external_tensor("indices", sparse_indices_path.name))
    sparse.dims.extend([1])
    graph = onnx.helper.make_graph([node], "root", [], [])
    graph.sparse_initializer.append(sparse)
    model_path = tmp_path / "forge.onnx"
    onnx.save_model(onnx.helper.make_model(graph), model_path)

    assert set(resolve_onnx_artifact_family(model_path)) == {
        "forge.onnx",
        "nested.bin",
        "sparse-values.bin",
        "sparse-indices.bin",
    }
