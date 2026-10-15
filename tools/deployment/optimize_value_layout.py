"""Move spatial-scale slices before the decoder value transpose in ONNX."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np


def _ancestors(graph, outputs):
    producers = {name: node for node in graph.node for name in node.output}
    pending, needed = list(outputs), set()
    while pending:
        name = pending.pop()
        if name not in needed:
            needed.add(name)
            if name in producers:
                pending.extend(producers[name].input)
    return needed


def _describe(model):
    import onnx

    producers = {name: node for node in model.graph.node for name in node.output}
    users = {}
    for node in model.graph.node:
        for name in node.input:
            users.setdefault(name, []).append(node)
    initializers = {value.name: value for value in model.graph.initializer}

    def constant(name):
        if name in initializers:
            return onnx.numpy_helper.to_array(initializers[name]).tolist()
        node = producers.get(name)
        if node is None or node.op_type != "Constant":
            raise ValueError("Value-layout Slice axes must be constant.")
        attribute = next(attr for attr in node.attribute if attr.name == "value")
        return onnx.numpy_helper.to_array(attribute.t).tolist()

    candidates = []
    for node in model.graph.node:
        if node.op_type != "Transpose":
            continue
        perm = next((list(attr.ints) for attr in node.attribute if attr.name == "perm"), None)
        parent = producers.get(node.input[0])
        following = users.get(node.output[0], [])
        if perm != [0, 2, 3, 1] or parent is None or parent.op_type != "Reshape":
            continue
        if len(following) < 2 or any(
            child.op_type != "Slice" or child.input[0] != node.output[0] for child in following
        ):
            continue
        if node.output[0] in {value.name for value in model.graph.output}:
            continue
        if any(len(child.input) not in (4, 5) for child in following):
            raise ValueError("Expected Slice inputs with explicit axes.")
        if any(constant(child.input[3]) not in ([-1], [3]) for child in following):
            raise ValueError("Value-layout slices must select the spatial-token axis.")
        candidates.append(
            {
                "memory": parent.input[0],
                "reshape": parent.name,
                "reshape_output": parent.output[0],
                "transpose": node.name,
                "slice_names": [child.name for child in following],
                "level_names": [child.output[0] for child in following],
                "shape_controls_preserved": True,
            }
        )
    if len(candidates) != 1:
        raise ValueError(f"Expected one decoder value-layout chain; found {len(candidates)}.")
    return candidates[0]


def rewrite_value_layout(model, pipeline=None):
    """Preserve weights, shape controls, I/O and query selection; return graph/metadata.

    Accept a full fixed-query D-FINE/DEIM graph or a split BUTR decoder graph.
    With a BUTR pipeline, resolve level shapes as in the measured deployment.
    The source model is never modified.
    """
    import onnx
    from onnx.reference import ReferenceEvaluator

    rewritten = copy.deepcopy(model)
    graph = rewritten.graph
    descriptor = _describe(rewritten)
    if pipeline is not None:
        memory = descriptor["memory"]
        inputs = {value.name: value for value in graph.input}
        if memory not in pipeline["frontier"] or memory not in inputs:
            raise ValueError("The value memory must be a split decoder boundary input.")
        if pipeline["budget_tensor"] in _ancestors(graph, descriptor["level_names"]):
            raise ValueError("Value layout must be independent of the query budget.")
        info = inputs[memory].type.tensor_type
        shape = [dim.dim_value for dim in info.shape.dim]
        if len(shape) != 3 or shape[0] != 1 or not all(size > 0 for size in shape):
            raise ValueError("Expected fixed batch-one value memory [1, locations, channels].")
        if info.elem_type != onnx.TensorProto.FLOAT:
            raise ValueError("The measured split BUTR layout expects FP32 boundary memory.")
        micro = onnx.utils.Extractor(rewritten).extract_model([memory], descriptor["level_names"])
        values = ReferenceEvaluator(micro).run(None, {memory: np.zeros(shape, dtype=np.float32)})
        shapes = [list(value.shape) for value in values]
        if not all(len(value) == 4 and value[:3] == shapes[0][:3] for value in shapes):
            raise ValueError("Incompatible per-scale value shapes.")
        if sum(value[-1] for value in shapes) != shape[1]:
            raise ValueError("Value slices must partition all spatial tokens.")
        for name, value_shape in zip(descriptor["level_names"], shapes):
            for value in list(graph.value_info):
                if value.name == name:
                    graph.value_info.remove(value)
            graph.value_info.append(
                onnx.helper.make_tensor_value_info(name, info.elem_type, value_shape)
            )
        descriptor.update(
            {
                "memory_shape": shape,
                "level_shapes": shapes,
                "lengths": [value[-1] for value in shapes],
                "heads": shapes[0][1],
                "channels_per_head": shapes[0][2],
                "query_independent": True,
            }
        )

    axis_name = "/model/decoder/decoder/value_layout/slice_axis_before_transpose"
    if axis_name in {value.name for value in graph.initializer}:
        raise ValueError("The graph already contains a split-first layout rewrite.")
    graph.initializer.append(onnx.numpy_helper.from_array(np.array([1], dtype=np.int64), axis_name))
    nodes = []
    for node in graph.node:
        if node.name == descriptor["transpose"]:
            continue
        if node.name in descriptor["slice_names"]:
            before = node.output[0] + "/before_transpose"
            feed = [descriptor["reshape_output"], node.input[1], node.input[2], axis_name]
            if len(node.input) == 5:
                feed.append(node.input[4])
            nodes.extend(
                [
                    onnx.helper.make_node("Slice", feed, [before], name=node.name + "/slice_first"),
                    onnx.helper.make_node(
                        "Transpose",
                        [before],
                        list(node.output),
                        name=node.name + "/transpose_after_slice",
                        perm=[0, 2, 3, 1],
                    ),
                ]
            )
        else:
            nodes.append(node)
    del graph.node[:]
    graph.node.extend(nodes)
    changed = {descriptor["transpose"], *descriptor["slice_names"]}
    original_nodes = {
        node.name: node.SerializeToString() for node in model.graph.node if node.name not in changed
    }
    remaining_nodes = {
        node.name: node.SerializeToString() for node in graph.node if node.name in original_nodes
    }
    if original_nodes != remaining_nodes:
        raise RuntimeError("Unrelated ONNX operations changed.")
    original_parameters = {
        value.name: value.SerializeToString() for value in model.graph.initializer
    }
    parameters = {value.name: value.SerializeToString() for value in graph.initializer}
    if any(parameters.get(name) != data for name, data in original_parameters.items()):
        raise RuntimeError("Original model parameters changed.")
    for name in ("input", "output"):
        if [v.SerializeToString() for v in getattr(model.graph, name)] != [
            v.SerializeToString() for v in getattr(graph, name)
        ]:
            raise RuntimeError("Model I/O changed.")
    onnx.checker.check_model(rewritten)
    return rewritten, descriptor


def main():
    import onnx

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pipeline", type=Path, help="Optional split BUTR pipeline.json")
    args = parser.parse_args()
    if args.onnx.resolve() == args.output.resolve():
        parser.error("Use a separate output path to preserve the original comparison graph.")
    source = onnx.load(args.onnx)
    pipeline = json.loads(args.pipeline.read_text()) if args.pipeline else None
    rewritten, descriptor = rewrite_value_layout(source, pipeline)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(rewritten, args.output)
    metadata = {
        "layout": "splitfirst",
        "descriptor": descriptor,
        "source_onnx_sha256": hashlib.sha256(args.onnx.read_bytes()).hexdigest(),
        "onnx_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "weights_changed": False,
        "query_selection_changed": False,
    }
    args.output.with_suffix(".onnx.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        f"Split-first value layout: {args.output}; {len(descriptor['level_names'])} feature levels"
    )


if __name__ == "__main__":
    main()
