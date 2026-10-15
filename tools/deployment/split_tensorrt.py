"""Split a dynamic BUTR ONNX graph at its confidence-derived query budget.

TensorRT 10.4 requires TopK K to be a shape input. The encoder computes K;
the decoder receives the same scalar through a host shape input.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def split_model(
    onnx_path, output_dir, query_profile=(300, 600, 1500), *, decoder_layout="splitfirst"
):
    import onnx
    from onnx import numpy_helper
    from onnx.utils import Extractor

    from tools.deployment.export_onnx import check_dynamic_budget

    minimum, optimum, maximum = query_profile
    if decoder_layout not in ("original", "splitfirst"):
        raise ValueError("decoder_layout must be original or splitfirst.")
    if not 1 <= minimum <= optimum <= maximum <= 3840:
        raise ValueError("Require 1 <= min <= opt <= max <= 3840 for the query profile.")
    model = onnx.shape_inference.infer_shapes(onnx.load(str(onnx_path)))
    check_dynamic_budget(model)
    graph = model.graph
    shapes = {value.name: value for value in [*graph.input, *graph.value_info, *graph.output]}
    # Shape of a fixed input/intermediate tensor is constant. Folding only these
    # nodes avoids exporting extra host shape inputs across the engine boundary.
    for node in list(graph.node):
        if node.op_type != "Shape" or node.input[0] not in shapes:
            continue
        tensor_type = shapes[node.input[0]].type.tensor_type
        if not tensor_type.HasField("shape"):
            continue  # Unknown rank is not a scalar shape.
        dimensions = tensor_type.shape.dim
        if not all(dim.HasField("dim_value") for dim in dimensions):
            continue
        attributes = {item.name: onnx.helper.get_attribute_value(item) for item in node.attribute}
        values = [dim.dim_value for dim in dimensions][
            attributes.get("start", 0) : attributes.get("end", len(dimensions))
        ]
        graph.initializer.append(
            numpy_helper.from_array(np.asarray(values, dtype=np.int64), node.output[0])
        )
        graph.node.remove(node)
    producers = {output: index for index, node in enumerate(graph.node) for output in node.output}

    def ancestors(names):
        pending, nodes = list(names), set()
        while pending:
            index = producers.get(pending.pop())
            if index is not None and index not in nodes:
                nodes.add(index)
                pending.extend(graph.node[index].input)
        return nodes

    for topk in graph.node:
        if topk.op_type == "TopK":
            operations = {graph.node[index].op_type for index in ancestors([topk.input[1]])}
            if {"Ceil", "ReduceSum"} <= operations:
                break
    else:
        raise ValueError("Cannot find the budget-controlled proposal TopK.")
    budget_name = topk.input[1]
    encoder_nodes = ancestors(topk.input)
    dependent = {value.name for value in graph.input}
    for node in graph.node:
        if any(name in dependent for name in node.input):
            dependent.update(node.output)
    needed = {
        name
        for index, node in enumerate(graph.node)
        if index not in encoder_nodes
        for name in node.input
    }
    frontier = [
        name
        for index, node in enumerate(graph.node)
        if index in encoder_nodes
        for name in node.output
        if name in needed and name in dependent
    ]
    if budget_name not in frontier:
        raise ValueError("Budget did not reach the decoder boundary.")
    encoder_input_names = [
        value.name
        for value in graph.input
        if value.name in {name for i in encoder_nodes for name in graph.node[i].input}
    ]
    decoder_input_names = frontier + [value.name for value in graph.input if value.name in needed]
    extractor = Extractor(model)
    encoder = extractor.extract_model(encoder_input_names, frontier)
    decoder = extractor.extract_model(decoder_input_names, [value.name for value in graph.output])
    # ONNX shape inference can leave fixed encoder dimensions symbolic. Resolve
    # the boundary shapes once at the fixed input size, without fixing K's value.
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(
        encoder.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    sample = {}
    for value in encoder.graph.input:
        dims = value.type.tensor_type.shape.dim
        if not all(dim.HasField("dim_value") for dim in dims):
            raise ValueError(f"Encoder input {value.name} must have a fixed shape.")
        dtype = onnx.helper.tensor_dtype_to_np_dtype(value.type.tensor_type.elem_type)
        sample[value.name] = np.zeros([dim.dim_value for dim in dims], dtype=dtype)
    boundary_shapes = {
        name: array.shape for name, array in zip(frontier, session.run(frontier, sample))
    }
    # Some fixed feature shapes were unknown to ONNX's static inference. Now
    # fold their Shape consumers too, so only values (not entire feature maps
    # used just for shape lookup) cross the engine boundary.
    for node in list(graph.node):
        if node.op_type == "Shape" and node.input[0] in boundary_shapes:
            dimensions = boundary_shapes[node.input[0]]
            attributes = {
                item.name: onnx.helper.get_attribute_value(item) for item in node.attribute
            }
            values = dimensions[attributes.get("start", 0) : attributes.get("end", len(dimensions))]
            graph.initializer.append(
                numpy_helper.from_array(np.asarray(values, dtype=np.int64), node.output[0])
            )
            graph.node.remove(node)
    producers = {output: index for index, node in enumerate(graph.node) for output in node.output}
    encoder_nodes = ancestors(topk.input)
    needed = {
        name
        for index, node in enumerate(graph.node)
        if index not in encoder_nodes
        for name in node.input
    }
    frontier = [
        name
        for index, node in enumerate(graph.node)
        if index in encoder_nodes
        for name in node.output
        if name in needed and name in dependent
    ]
    decoder_input_names = frontier + [value.name for value in graph.input if value.name in needed]
    extractor = Extractor(model)
    encoder = extractor.extract_model(encoder_input_names, frontier)
    decoder = extractor.extract_model(decoder_input_names, [value.name for value in graph.output])
    boundary_shapes = {name: boundary_shapes[name] for name in frontier}
    for part in (encoder, decoder):
        for value in [*part.graph.input, *part.graph.output, *part.graph.value_info]:
            if value.name in boundary_shapes:
                value.type.tensor_type.shape.ClearField("dim")
                for size in boundary_shapes[value.name]:
                    value.type.tensor_type.shape.dim.add().dim_value = size
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": "butr-tensorrt-split-v1",
        "encoder": "encoder.engine",
        "decoder": "decoder.engine",
        "budget_tensor": budget_name,
        "query_profile": list(query_profile),
        "encoder_inputs": encoder_input_names,
        "decoder_inputs": decoder_input_names,
        "frontier": frontier,
        "postprocessing_in_decoder": True,
        "boundary_shapes": {name: list(shape) for name, shape in boundary_shapes.items()},
    }
    onnx.checker.check_model(decoder)
    onnx.save(decoder, str(output_dir / "decoder-original.onnx"))
    manifest["decoder_layout"] = decoder_layout
    if decoder_layout == "splitfirst":
        from tools.deployment.optimize_value_layout import rewrite_value_layout

        decoder, descriptor = rewrite_value_layout(decoder, manifest)
        manifest["value_layout"] = descriptor
    for name, part in (("encoder", encoder), ("decoder", decoder)):
        onnx.checker.check_model(part)
        onnx.save(part, str(output_dir / f"{name}.onnx"))
    layout_metadata = {
        "layout": decoder_layout,
        "source_onnx_sha256": hashlib.sha256(
            (output_dir / "decoder-original.onnx").read_bytes()
        ).hexdigest(),
        "onnx_sha256": hashlib.sha256((output_dir / "decoder.onnx").read_bytes()).hexdigest(),
        "weights_changed": False,
        "query_selection_changed": False,
    }
    if "value_layout" in manifest:
        layout_metadata["descriptor"] = manifest["value_layout"]
    (output_dir / "decoder.onnx.json").write_text(json.dumps(layout_metadata, indent=2) + "\n")
    (output_dir / "pipeline.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Split into {output_dir}: {len(frontier)} boundary tensors; budget={budget_name}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--decoder-layout",
        choices=("original", "splitfirst"),
        default="splitfirst",
        help="splitfirst: measured value-layout optimization; original: unmodified control",
    )
    parser.add_argument(
        "--query-profile",
        nargs=3,
        type=int,
        default=(300, 600, 1500),
        metavar=("MIN", "OPT", "MAX"),
    )
    args = parser.parse_args()
    split_model(args.onnx, args.output_dir, args.query_profile, decoder_layout=args.decoder_layout)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    main()
