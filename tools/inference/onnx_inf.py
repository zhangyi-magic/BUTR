"""Render image/video detections using an exported BUTR ONNX model.

Modified from Dome-DETR.
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def prepare_inputs(image, spatial_size):
    """Match validation: RGB, direct resize, [0, 1], original (width, height)."""
    height, width = spatial_size
    resized = image.convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
    images = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1)[None] / np.float32(255)
    return {
        "images": np.ascontiguousarray(images),
        "orig_target_sizes": np.asarray([image.size], dtype=np.int64),
    }


def draw(image, outputs, threshold):
    labels, boxes, scores = (value[0] for value in outputs)
    canvas = ImageDraw.Draw(image)
    for label, box, score in zip(labels, boxes, scores):
        if score >= threshold:
            canvas.rectangle(box.tolist(), outline="red", width=2)
            canvas.text(tuple(box[:2]), f"{label} {score:.2f}", fill="red")


def main(args):
    import onnxruntime as ort

    if not 0 <= args.threshold <= 1:
        raise ValueError("--threshold must lie in [0, 1].")
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.num_threads
    session = ort.InferenceSession(args.onnx, options, providers=["CPUExecutionProvider"])
    shape = next(item.shape for item in session.get_inputs() if item.name == "images")
    if len(shape) != 4 or shape[:2] != [1, 3] or not all(isinstance(dim, int) for dim in shape[2:]):
        raise ValueError("Expected a batch-1 ONNX model with fixed image height/width.")
    spatial_size = shape[2:]

    def predict(image):
        outputs = session.run(["labels", "boxes", "scores"], prepare_inputs(image, spatial_size))
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
    output = Path(args.output or ("onnx_results.jpg" if image_input else "onnx_results.mp4"))
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
    parser.add_argument("--onnx", required=True, help="exported ONNX model")
    parser.add_argument("-i", "--input", required=True, help="image or video")
    parser.add_argument("-o", "--output", help="output image/video path")
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--num-threads", type=int, default=2)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
