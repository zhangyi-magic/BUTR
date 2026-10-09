"""Render detections with a TensorRT engine or dynamic BUTR pipeline.

Modified from Dome-DETR.
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.deployment.tensorrt_runtime import load_runner
from tools.inference.onnx_inf import draw, prepare_inputs


def main(args):
    if not 0 <= args.threshold <= 1:
        raise ValueError("--threshold must lie in [0, 1].")
    runner = load_runner(args.engine, args.device, args.max_queries, runtime_mode=args.runtime)
    shape = runner.input_shape("images")
    if len(shape) != 4 or shape[:2] != (1, 3):
        raise ValueError("Expected a batch-1 engine with RGB input.")

    def predict(image):
        inputs = {
            name: torch.from_numpy(value).to(device=args.device, dtype=runner.input_dtype(name))
            for name, value in prepare_inputs(image, shape[2:]).items()
        }
        runner(inputs)
        outputs = [runner.outputs()[name].cpu().numpy() for name in ("labels", "boxes", "scores")]
        draw(image, outputs, args.threshold)
        return outputs[0].shape[1]

    image_input = Path(args.input).suffix.lower() in {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
        ".tif",
        ".tiff",
    }
    output = Path(
        args.output or ("output/trt_detections.jpg" if image_input else "output/trt_detections.mp4")
    )
    if output.resolve() == Path(args.input).resolve():
        raise ValueError("Output must differ from the input file.")
    output.parent.mkdir(parents=True, exist_ok=True)
    if image_input:
        with Image.open(args.input) as source:
            image = source.convert("RGB")
        count = predict(image)
        image.save(output)
        print(f"Decoded {count} queries")
    else:
        import cv2

        capture = cv2.VideoCapture(args.input)
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {args.input}")
        writer = None
        try:
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
            writer = cv2.VideoWriter(
                str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
            )
            if not writer.isOpened():
                raise ValueError(f"Cannot write video: {output}")
            while True:
                success, frame = capture.read()
                if not success:
                    break
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                predict(image)
                writer.write(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR))
        finally:
            capture.release()
            if writer is not None:
                writer.release()
    print(f"Saved detections to {output}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--engine", "--trt", "-trt", required=True, help=".engine or split pipeline.json"
    )
    parser.add_argument("-i", "--input", required=True, help="image or video")
    parser.add_argument("-o", "--output", help="output image/video path")
    parser.add_argument("-d", "--device", default="cuda:0")
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--max-queries", type=int, default=1500)
    parser.add_argument("--runtime", choices=("sync", "async", "cuda_graph"), default="sync")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
