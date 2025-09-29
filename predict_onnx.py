"""Utility helpers for quickly validating ONNX exports.

This script can be used in two modes:

1.  Inspect the tensor shapes recorded in the ONNX graph (after running
    ``onnx.shape_inference``).  This is useful to debug broadcasting issues such
    as the ``Add`` failure that was reported when feeding a tensor of shape
    ``(2, 3, 1024, 1024)``.
2.  Optionally run a quick inference pass with random data to confirm that the
    exported model accepts the supplied shape.

Example usages::

    # Print the model IO description and inspect the tensors that feed Add_3
    python predict_onnx.py --model /path/to/best_model.onnx \
        --inspect-node Add_3

    # Run an inference test with the desired input shape
    python predict_onnx.py --model /path/to/best_model.onnx \
        --input-shape 2,3,1024,1024 --run-inference
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np
import onnx
from onnx import shape_inference
import onnxruntime as ort


@dataclass
class TensorDescription:
    """Light-weight representation of a tensor value info entry."""

    name: str
    dtype: Optional[str]
    shape: Sequence[Optional[int]]

    def format(self) -> str:
        dims = [str(dim) if dim is not None else "?" for dim in self.shape]
        shape_str = f"({', '.join(dims)})" if dims else "()"
        dtype = self.dtype or "unknown"
        return f"{self.name}: {dtype} {shape_str}"


def _parse_shape(shape_str: str) -> Sequence[int]:
    try:
        return tuple(int(dim) for dim in shape_str.split(","))
    except ValueError as exc:  # pragma: no cover - defensive
        raise argparse.ArgumentTypeError(
            f"Unable to parse shape '{shape_str}'. Expect comma separated integers."
        ) from exc


def _collect_value_infos(graph: onnx.GraphProto) -> List[onnx.ValueInfoProto]:
    infos: List[onnx.ValueInfoProto] = []
    infos.extend(graph.input)
    infos.extend(graph.value_info)
    infos.extend(graph.output)
    return infos


def _dtype_of(value_info: onnx.ValueInfoProto) -> Optional[str]:
    tensor_type = value_info.type.tensor_type
    if tensor_type.elem_type == 0:  # pragma: no cover - extremely rare
        return None
    return onnx.mapping.TENSOR_TYPE_TO_NP_TYPE.get(tensor_type.elem_type, None)


def _shape_of(value_info: onnx.ValueInfoProto) -> Sequence[Optional[int]]:
    dims: List[Optional[int]] = []
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(int(dim.dim_value))
        elif dim.HasField("dim_param"):
            dims.append(None)
        else:  # pragma: no cover - defensive
            dims.append(None)
    return dims


def _describe_tensor(graph: onnx.GraphProto, name: str) -> Optional[TensorDescription]:
    for info in _collect_value_infos(graph):
        if info.name == name:
            return TensorDescription(name=name, dtype=_dtype_of(info), shape=_shape_of(info))
    return None


def _print_model_ios(graph: onnx.GraphProto) -> None:
    print("Model inputs:")
    for value in graph.input:
        desc = TensorDescription(name=value.name, dtype=_dtype_of(value), shape=_shape_of(value))
        print(f"  - {desc.format()}")

    print("Model outputs:")
    for value in graph.output:
        desc = TensorDescription(name=value.name, dtype=_dtype_of(value), shape=_shape_of(value))
        print(f"  - {desc.format()}")


def _print_node_shapes(graph: onnx.GraphProto, inspect_names: Iterable[str]) -> None:
    wanted = set(inspect_names)
    if not wanted:
        return

    print("\nRequested node input shapes:")
    for node in graph.node:
        if not any(key in node.name for key in wanted):
            continue

        print(f"- Node: {node.name} ({node.op_type})")
        for idx, input_name in enumerate(node.input):
            desc = _describe_tensor(graph, input_name)
            if desc is None:
                print(f"    input[{idx}] {input_name}: <shape unknown>")
            else:
                print(f"    input[{idx}] {desc.format()}")
        for idx, output_name in enumerate(node.output):
            desc = _describe_tensor(graph, output_name)
            if desc is None:
                print(f"    output[{idx}] {output_name}: <shape unknown>")
            else:
                print(f"    output[{idx}] {desc.format()}")


def _run_inference(model_path: str, input_shape: Sequence[int]) -> None:
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    dummy = np.random.randn(*input_shape).astype(np.float32)
    inputs = {session.get_inputs()[0].name: dummy}
    outputs = session.run(None, inputs)
    print("Inference succeeded. Output tensor shapes:")
    for idx, array in enumerate(outputs):
        print(f"  - output[{idx}]: {array.shape}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect and test ONNX exports.")
    parser.add_argument("--model", required=True, help="Path to the ONNX model file.")
    parser.add_argument(
        "--input-shape",
        type=_parse_shape,
        help="Comma separated shape to test inference with, e.g. 1,3,1024,1024.",
    )
    parser.add_argument(
        "--inspect-node",
        action="append",
        default=[],
        help="Substring of node names to inspect (can be used multiple times).",
    )
    parser.add_argument(
        "--run-inference",
        action="store_true",
        help="Run an inference pass with random data using --input-shape.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = onnx.load(args.model)
    onnx.checker.check_model(model)

    inferred = shape_inference.infer_shapes(model)
    graph = inferred.graph

    _print_model_ios(graph)
    _print_node_shapes(graph, args.inspect_node)

    if args.run_inference:
        if args.input_shape is None:
            raise SystemExit("--run-inference requires --input-shape to be specified.")
        _run_inference(args.model, args.input_shape)


if __name__ == "__main__":
    main()
