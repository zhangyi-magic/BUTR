"""Export BUTR with input-dependent queries (batch size 1, fixed image size).

Adapted from D-FINE (https://github.com/Peterande/D-FINE).
Copyright(c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import argparse
import inspect
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core import YAMLConfig
from src.zoo.models.butr_decoder import BUTRTransformer


class DeploymentModel(nn.Module):
    def __init__(self, model, postprocessor):
        super().__init__()
        self.model = model.deploy()
        self.postprocessor = postprocessor.deploy()

    def forward(self, images, orig_target_sizes):
        return self.postprocessor(self.model(images), orig_target_sizes)


def load_deployment_state(detector, state):
    """Strictly load weights; recreate only missing deterministic deployment caches."""
    state = dict(state)
    initial = detector.state_dict()
    filled = []
    for name in ("decoder.anchors", "decoder.valid_mask", "decoder.decoder.project"):
        if name in initial and name not in state:
            state[name] = initial[name]
            filled.append(name)
    detector.load_state_dict(state, strict=True)
    if filled:
        print("Recreated derived buffers:", ", ".join(filled), flush=True)
    return filled


def check_dynamic_budget(graph_model):
    """Reject graphs whose TopK has lost the confidence-derived budget."""
    producers = {output: node for node in graph_model.graph.node for output in node.output}
    graph_inputs = {item.name for item in graph_model.graph.input}
    for node in graph_model.graph.node:
        if node.op_type != "TopK":
            continue
        pending, visited, operations = [node.input[1]], set(), set()
        while pending:
            name = pending.pop()
            if name in visited:
                continue
            visited.add(name)
            producer = producers.get(name)
            if producer is not None:
                operations.add(producer.op_type)
                pending.extend(producer.input)
        if {"Ceil", "ReduceSum"} <= operations and visited & graph_inputs:
            return
    raise ValueError("ONNX TopK is missing the input-dependent BUTR budget; export rejected.")


def export_model(model, inputs, output, *, require_dynamic_budget=True):
    """Export tensor-valued TopK K and dynamic query axes on detection outputs."""
    import onnx

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fastpath = torch.backends.mha.get_fastpath_enabled()
    # Native fused MHA is not supported by the legacy ONNX exporter.
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        options = {}
        if "dynamo" in inspect.signature(torch.onnx.export).parameters:
            options["dynamo"] = False
        with torch.no_grad():
            torch.onnx.export(
                model.eval(),
                inputs,
                str(output),
                opset_version=16,
                input_names=["images", "orig_target_sizes"],
                output_names=["labels", "boxes", "scores"],
                dynamic_axes={name: {1: "queries"} for name in ("labels", "boxes", "scores")},
                **options,
            )
        graph_model = onnx.load(str(output))
        onnx.checker.check_model(graph_model)
        if require_dynamic_budget:
            check_dynamic_budget(graph_model)
    finally:
        torch.backends.mha.set_fastpath_enabled(fastpath)
    return output


def compare_detections(
    actual, expected, original_sizes, *, box_atol=0.02, score_atol=2e-4, rtol=1e-4
):
    """Compare detection sets, allowing TopK to reorder equal-scoring detections."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    for observed, reference in zip(actual, expected):
        if observed.shape != reference.shape:
            raise AssertionError(f"Output shape mismatch: {observed.shape} != {reference.shape}")
    for batch, size in enumerate(original_sizes):
        labels, boxes, scores = (value[batch] for value in actual)
        ref_labels, ref_boxes, ref_scores = (value[batch] for value in expected)
        np.testing.assert_array_equal(np.sort(labels), np.sort(ref_labels))
        scale = np.tile(size, 2)
        for label in np.unique(ref_labels):
            found, wanted = np.flatnonzero(labels == label), np.flatnonzero(ref_labels == label)
            costs = np.abs((boxes[found, None] - ref_boxes[None, wanted]) / scale).sum(-1)
            costs += np.abs(scores[found, None] - ref_scores[None, wanted])
            rows, columns = linear_sum_assignment(costs)
            np.testing.assert_allclose(
                boxes[found[rows]],
                ref_boxes[wanted[columns]],
                atol=box_atol,
                rtol=rtol,
            )
            np.testing.assert_allclose(
                scores[found[rows]],
                ref_scores[wanted[columns]],
                atol=score_atol,
                rtol=rtol,
            )


def verify_images(model, output, paths, spatial_size):
    """Compare deployed PyTorch and ONNX models on user-supplied images."""
    import onnxruntime as ort
    from PIL import Image
    from torchvision.transforms import Compose, Resize, ToTensor

    transform = Compose([Resize(tuple(spatial_size)), ToTensor()])
    options = ort.SessionOptions()
    options.intra_op_num_threads = torch.get_num_threads()
    session = ort.InferenceSession(str(output), options, providers=["CPUExecutionProvider"])
    counts = []
    for path in paths:
        with Image.open(path) as source:
            image = source.convert("RGB")
        images = transform(image).unsqueeze(0)
        sizes = torch.tensor([image.size], dtype=torch.int64)  # width, height
        with torch.no_grad():
            expected = model(images, sizes)
        actual = session.run(None, {"images": images.numpy(), "orig_target_sizes": sizes.numpy()})
        compare_detections(actual, [value.numpy() for value in expected], sizes.numpy())
        counts.append(actual[0].shape[1])
        print(f"Verified {path}: {counts[-1]} queries", flush=True)
    if len(set(counts)) < 2:
        print(
            "These images produced the same budget. Try more varied images to check changing counts."
        )


def main(args):
    torch.set_num_threads(args.num_threads)
    if args.upstream_root:
        import importlib

        root = Path(args.upstream_root).resolve()
        if not (root / "engine/core").is_dir():
            raise ValueError("Expected a DEIM checkout containing engine/core.")
        sys.path.insert(0, str(root))
        upstream_config = importlib.import_module("engine.core").YAMLConfig
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = root / config_path
        cfg = upstream_config(str(config_path), resume=args.resume)
    else:
        cfg = YAMLConfig(args.config, resume=args.resume)
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
    state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    detector = cfg.model
    load_deployment_state(detector, state)
    require_dynamic = any(isinstance(module, BUTRTransformer) for module in detector.modules())
    model = DeploymentModel(detector, cfg.postprocessor).eval()
    height, width = cfg.yaml_cfg["eval_spatial_size"]
    inputs = (torch.zeros(1, 3, height, width), torch.tensor([[width, height]], dtype=torch.int64))
    output = export_model(
        model,
        inputs,
        args.output or Path(args.resume).with_suffix(".onnx"),
        require_dynamic_budget=require_dynamic,
    )
    print(
        f"Exported {output}: batch=1, resolution={height}x{width}, dynamic queries={require_dynamic}"
    )
    if args.verify_images:
        verify_images(model, output, args.verify_images, (height, width))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-r", "--resume", required=True, help="trained checkpoint")
    parser.add_argument("-o", "--output", help="ONNX path (default: next to the checkpoint)")
    parser.add_argument(
        "--upstream-root", help="DEIM checkout for exporting official fixed-query baselines"
    )
    parser.add_argument(
        "--verify-images", nargs="+", help="compare ONNX Runtime with PyTorch on images"
    )
    parser.add_argument(
        "--num-threads", type=int, default=2, help="CPU threads for export/verification"
    )
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
