"""Safe resolution of an ONNX graph and its external-data artifacts."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any


def _attribute_tensors(attributes: Iterable[Any]) -> Iterator[Any]:
    """Yield tensors from node attributes, including recursively nested graphs."""
    for attribute in attributes:
        if attribute.HasField("t"):
            yield attribute.t
        yield from attribute.tensors
        if attribute.HasField("sparse_tensor"):
            yield attribute.sparse_tensor.values
            yield attribute.sparse_tensor.indices
        for sparse in attribute.sparse_tensors:
            yield sparse.values
            yield sparse.indices
        if attribute.HasField("g"):
            yield from _graph_tensors(attribute.g)
        for graph in attribute.graphs:
            yield from _graph_tensors(graph)


def _node_tensors(nodes: Iterable[Any]) -> Iterator[Any]:
    for node in nodes:
        yield from _attribute_tensors(node.attribute)


def _graph_tensors(graph: Any) -> Iterator[Any]:
    yield from graph.initializer
    for sparse in graph.sparse_initializer:
        yield sparse.values
        yield sparse.indices
    yield from _node_tensors(graph.node)


def _model_tensors(model: Any) -> Iterator[Any]:
    yield from _graph_tensors(model.graph)
    for function in model.functions:
        yield from _attribute_tensors(function.attribute_proto)
        yield from _node_tensors(function.node)


def _safe_external_location(location: str, *, graph_dir: Path) -> tuple[str, Path]:
    """Resolve one portable relative ONNX location without allowing aliases or escapes."""
    if not location or location != location.strip() or "\\" in location:
        raise ValueError(f"ONNX external-data location is unsafe: {location!r}")
    relative = PurePosixPath(location)
    if (
        relative.is_absolute()
        or PureWindowsPath(location).drive
        or any(part in {"", ".", ".."} for part in location.split("/"))
    ):
        raise ValueError(f"ONNX external-data location is unsafe: {location!r}")

    candidate = graph_dir.joinpath(*relative.parts)
    resolved = candidate.resolve()
    try:
        resolved.relative_to(graph_dir)
    except ValueError as exc:
        raise ValueError(f"ONNX external-data location escapes the graph directory: {location!r}") from exc
    if not resolved.is_file():
        raise ValueError(f"ONNX external-data file is missing or not a file: {location!r}")
    return relative.as_posix(), resolved


def resolve_onnx_artifact_family(onnx_path: str | Path) -> dict[str, Path]:
    """Return the exact graph-plus-external-data family required by an ONNX model.

    Keys are portable paths relative to the graph directory. Repeated references to
    the same external-data location are allowed because a single ONNX sidecar commonly
    stores many tensors. Distinct path spellings that resolve to one file are rejected.
    """
    graph = Path(onnx_path).expanduser().resolve()
    if not graph.is_file():
        raise FileNotFoundError(f"Required ONNX graph is missing: {graph}")
    graph_dir = graph.parent

    try:
        import onnx

        model = onnx.load(str(graph), load_external_data=False)
    except Exception as exc:
        raise ValueError(f"Unable to parse ONNX graph {graph}: {exc}") from exc

    references: dict[str, Path] = {}
    resolved_locations: dict[Path, str] = {graph: graph.name}
    external_value = int(onnx.TensorProto.EXTERNAL)
    for tensor in _model_tensors(model):
        locations = [entry.value for entry in tensor.external_data if entry.key == "location"]
        if len(locations) > 1:
            raise ValueError(f"ONNX tensor {tensor.name!r} declares duplicate external-data locations")
        if int(tensor.data_location) == external_value or tensor.external_data:
            if not locations:
                raise ValueError(f"ONNX tensor {tensor.name!r} has external data without a location")
            key, resolved = _safe_external_location(locations[0], graph_dir=graph_dir)
            if resolved == graph:
                raise ValueError(f"ONNX external-data location {key!r} aliases the graph itself")
            existing_key = resolved_locations.get(resolved)
            if existing_key is not None and existing_key != key:
                raise ValueError(f"ONNX external-data locations {existing_key!r} and {key!r} resolve to the same file")
            resolved_locations[resolved] = key
            references[key] = resolved

    family = {graph.name: graph}
    family.update({key: references[key] for key in sorted(references)})
    return family
