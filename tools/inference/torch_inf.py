"""Render image/video detections from a BUTR checkpoint.

Modified from Dome-DETR.
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-r", "--resume", required=True, help="model checkpoint")
    parser.add_argument("-i", "--input", required=True, help="image or video")
    parser.add_argument("-d", "--device", default="cpu")
    parser.add_argument("-o", "--output", help="output image/video path")
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("-a", "--annotation", help="optional COCO annotations, drawn in green")
    return parser


def draw(image, labels, boxes, scores, threshold=0.4):
    """Draw predictions in place; callers control whether and where to save."""
    from PIL import ImageDraw

    canvas = ImageDraw.Draw(image)
    for label, box, score in zip(labels.cpu(), boxes.cpu(), scores.cpu()):
        if score.item() < threshold:
            continue
        canvas.rectangle(box.tolist(), outline="red", width=2)
        canvas.text(tuple(box[:2].tolist()), f"{label.item()} {score.item():.2f}", fill="red")


def draw_annotations(image, image_path, annotation_path):
    """COCO boxes use top-left xywh coordinates."""
    from PIL import ImageDraw

    with open(annotation_path, encoding="utf-8") as stream:
        annotations = json.load(stream)
    image_ids = {
        item["id"]
        for item in annotations["images"]
        if Path(item["file_name"]).name == Path(image_path).name
    }
    canvas = ImageDraw.Draw(image)
    for item in annotations["annotations"]:
        if item["image_id"] in image_ids:
            x, y, width, height = item["bbox"]
            canvas.rectangle((x, y, x + width, y + height), outline="green", width=2)


def main(args):
    import torch
    import torchvision.transforms as transforms
    from PIL import Image

    from src.core import YAMLConfig

    if not 0 <= args.threshold <= 1:
        raise ValueError("--threshold must lie in [0, 1].")
    cfg = YAMLConfig(args.config, resume=args.resume)
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
    state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    model = cfg.model
    model.load_state_dict(state)
    model = model.deploy().to(args.device)
    postprocessor = cfg.postprocessor.deploy().to(args.device)
    size = cfg.yaml_cfg.get("eval_spatial_size", (800, 800))
    transform = transforms.Compose([transforms.Resize(tuple(size)), transforms.ToTensor()])

    @torch.inference_mode()
    def predict(image):
        inputs = transform(image).unsqueeze(0).to(args.device)
        # This repository consistently stores original sizes as (width, height).
        original_size = torch.tensor([image.size], device=args.device)
        labels, boxes, scores = postprocessor(model(inputs), original_size)
        draw(image, labels[0], boxes[0], scores[0], args.threshold)

    image_input = Path(args.input).suffix.lower() in {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
        ".tif",
        ".tiff",
    }
    output = Path(args.output or ("torch_results.jpg" if image_input else "torch_results.mp4"))
    if output.resolve() == Path(args.input).resolve():
        raise ValueError("Output must differ from the input file.")
    output.parent.mkdir(parents=True, exist_ok=True)
    if image_input:
        with Image.open(args.input) as source:
            image = source.convert("RGB")
        predict(image)
        if args.annotation:
            draw_annotations(image, args.input, args.annotation)
        image.save(output)
    else:
        import cv2
        import numpy as np

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


if __name__ == "__main__":
    main(build_parser().parse_args())
