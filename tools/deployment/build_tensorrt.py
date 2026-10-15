"""Build a TensorRT 10.4 FP16 engine from an ONNX graph."""

import argparse
import hashlib
import json
from pathlib import Path


def build_engine(
    onnx_path,
    output,
    *,
    workspace_gib=None,
    shape_profile=None,
    fp32_sensitive=False,
    avg_timing_iterations=8,
):
    if avg_timing_iterations < 1:
        raise ValueError("avg_timing_iterations must be positive.")
    source_hash = hashlib.sha256(Path(onnx_path).read_bytes()).hexdigest()
    layout = None
    layout_path = Path(onnx_path).with_suffix(".onnx.json")
    if layout_path.is_file():
        layout = json.loads(layout_path.read_text())
        if layout["onnx_sha256"] != source_hash:
            raise ValueError("ONNX layout metadata does not match the graph.")
    import tensorrt as trt
    import torch

    if trt.__version__ != "10.4.0":
        raise RuntimeError(
            f"Use TensorRT 10.4.0 for the D-FINE/DEIM protocol; found {trt.__version__}."
        )
    logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, "")
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(Path(onnx_path).resolve())):
        errors = "\n".join(str(parser.get_error(index)) for index in range(parser.num_errors))
        raise RuntimeError(f"TensorRT could not parse the ONNX graph:\n{errors}")
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    config.clear_flag(trt.BuilderFlag.TF32)
    config.avg_timing_iterations = avg_timing_iterations
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    protected_layers = []
    if fp32_sensitive:
        half_compute_types = {trt.LayerType.CONVOLUTION, trt.LayerType.MATRIX_MULTIPLY}
        for index in range(network.num_layers):
            layer = network.get_layer(index)
            if not (
                ("norm" in layer.name.lower() and "batchnormalization" not in layer.name.lower())
                or "/postprocessor/" in layer.name
                or ("/decoder/" in layer.name and layer.type not in half_compute_types)
            ):
                continue
            outputs = [
                i
                for i in range(layer.num_outputs)
                if layer.get_output(i).dtype in (trt.float32, trt.float16)
            ]
            if not outputs:
                continue
            if layer.type != trt.LayerType.PLUGIN_V2:
                layer.precision = trt.float32
            for i in outputs:
                layer.set_output_type(i, trt.float32)
            protected_layers.append(layer.name)
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        print(f"Preserving FP32 for {len(protected_layers)} sensitive layers", flush=True)
    if workspace_gib is not None:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gib * (1 << 30)))
    profile = builder.create_optimization_profile()
    has_profile = False
    for index in range(network.num_inputs):
        tensor = network.get_input(index)
        if tensor.is_shape_tensor:
            if not shape_profile or tensor.name not in shape_profile:
                raise ValueError(f"Shape input {tensor.name} needs a MIN/OPT/MAX value profile.")
            values = shape_profile[tensor.name]
            profile.set_shape_input(tensor.name, *values)
            if [list(value) for value in profile.get_shape_input(tensor.name)] != [
                list(value) for value in values
            ]:
                raise RuntimeError(f"Shape profile readback mismatch: {tensor.name}")
            has_profile = True
        if any(dim < 0 for dim in tensor.shape):
            raise ValueError(
                f"This builder expects fixed input shapes; {tensor.name}: {tensor.shape}"
            )
    if has_profile:
        config.add_optimization_profile(profile)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed; see the preceding TensorRT errors.")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(bytes(serialized))
    metadata = {
        "tensorrt": trt.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "fp16_enabled": True,
        "fp32_sensitive": fp32_sensitive,
        "fp32_layers": protected_layers,
        "precision_policy": "common-sensitive-fp32-v1" if fp32_sensitive else "fp16-default",
        "TF32_enabled": False,
        "avg_timing_iterations": avg_timing_iterations,
        "profiling_verbosity": "DETAILED",
        "workspace_gib": workspace_gib,
        "shape_profile": shape_profile,
        "onnx_sha256": source_hash,
        "engine_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "inputs": {
            network.get_input(i).name: list(network.get_input(i).shape)
            for i in range(network.num_inputs)
        },
    }
    if layout is not None:
        metadata["graph_optimization"] = layout
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Built {output} on {metadata['gpu']} (TensorRT {trt.__version__}, FP16 enabled)")
    return output


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("-o", "--output", required=True, help="engine path")
    parser.add_argument(
        "--workspace-gib", type=float, help="optional workspace limit; default: TensorRT default"
    )
    parser.add_argument(
        "--pipeline", help="split pipeline.json: provides the decoder budget profile"
    )
    parser.add_argument(
        "--avg-timing-iterations",
        type=int,
        default=8,
        help="TensorRT builder timing iterations; measured T4 protocol uses 8",
    )
    parser.add_argument(
        "--fp32-sensitive",
        action="store_true",
        help="preserve normalization, non-matrix decoder operations, and postprocessing in FP32",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    shape_profile = None
    if args.pipeline:
        manifest = json.loads(Path(args.pipeline).read_text())
        shape_profile = {
            manifest["budget_tensor"]: [[value] for value in manifest["query_profile"]]
        }
    build_engine(
        args.onnx,
        args.output,
        workspace_gib=args.workspace_gib,
        shape_profile=shape_profile,
        fp32_sensitive=args.fp32_sensitive,
        avg_timing_iterations=args.avg_timing_iterations,
    )
