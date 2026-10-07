#!/usr/bin/env python3
"""Convert SGC panicle LabelMe data to COCO after EXIF correction and 1824x1824 resizing."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument("--size", type=int, default=1824)
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def collect_files(directory: Path, suffixes: set[str] | None = None):
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    result = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        if suffixes is not None and path.suffix.lower() not in suffixes:
            continue
        if path.stem in result:
            raise RuntimeError(f"Duplicate stem {path.stem!r} in {directory}")
        result[path.stem] = path
    return result


def shape_to_xyxy(shape):
    points = shape.get("points", [])
    if len(points) < 2:
        return None

    try:
        xs = [float(p[0]) for p in points]
        ys = [float(p[1]) for p in points]
    except (TypeError, ValueError, IndexError):
        return None

    if not all(math.isfinite(v) for v in xs + ys):
        return None

    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def clip_xyxy(box, width, height):
    x1, y1, x2, y2 = box
    x1 = min(max(x1, 0.0), float(width))
    y1 = min(max(y1, 0.0), float(height))
    x2 = min(max(x2, 0.0), float(width))
    y2 = min(max(y2, 0.0), float(height))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def convert_split(
    source_groups,
    split_name,
    output_root,
    size,
    quality,
    next_image_id,
    next_ann_id,
):
    image_out = output_root / split_name / "images"
    ann_out = output_root / split_name / "annotations"
    image_out.mkdir(parents=True, exist_ok=True)
    ann_out.mkdir(parents=True, exist_ok=True)

    coco_images = []
    coco_annotations = []
    invalid_shapes = 0
    clipped_boxes = 0
    label_counts = Counter()
    objects_per_image = []

    for source_name, image_dir, json_dir in source_groups:
        images = collect_files(image_dir, IMAGE_EXTS)
        jsons = collect_files(json_dir, {".json"})

        stems = sorted(set(images) & set(jsons))
        missing_json = sorted(set(images) - set(jsons))
        missing_image = sorted(set(jsons) - set(images))

        if missing_json:
            print(f"[Warning] {source_name}: {len(missing_json)} images without JSON")
        if missing_image:
            print(f"[Warning] {source_name}: {len(missing_image)} JSON files without image")
        if not stems:
            raise RuntimeError(f"No valid pairs found for {source_name}")

        print(f"[{split_name}] {source_name}: {len(stems)} pairs")

        for stem in stems:
            image_path = images[stem]
            json_path = jsons[stem]
            labelme = json.loads(json_path.read_text(encoding="utf-8"))

            with Image.open(image_path) as raw:
                # Essential for this dataset: annotations correspond to the
                # visually upright EXIF-corrected image (3456x5184).
                image = ImageOps.exif_transpose(raw).convert("RGB")
                src_w, src_h = image.size

                json_w = labelme.get("imageWidth")
                json_h = labelme.get("imageHeight")
                if json_w is not None and json_h is not None:
                    if (src_w, src_h) != (int(json_w), int(json_h)):
                        raise RuntimeError(
                            f"Size mismatch after EXIF correction for {image_path.name}: "
                            f"image={(src_w, src_h)}, JSON={(json_w, json_h)}"
                        )

                resized = image.resize((size, size), Image.Resampling.LANCZOS)

            output_name = f"{source_name}__{stem}.jpg"
            resized.save(
                image_out / output_name,
                format="JPEG",
                quality=quality,
                subsampling=0,
                optimize=True,
            )

            image_id = next_image_id
            next_image_id += 1
            coco_images.append(
                {
                    "id": image_id,
                    "file_name": output_name,
                    "width": size,
                    "height": size,
                    "source_split": source_name,
                    "source_file_name": image_path.name,
                    "source_width": src_w,
                    "source_height": src_h,
                }
            )

            sx = size / src_w
            sy = size / src_h
            image_object_count = 0

            for shape in labelme.get("shapes", []):
                label_counts[str(shape.get("label", "")).strip()] += 1
                box = shape_to_xyxy(shape)
                if box is None:
                    invalid_shapes += 1
                    continue

                clipped = clip_xyxy(box, src_w, src_h)
                if clipped is None:
                    invalid_shapes += 1
                    continue
                if clipped != box:
                    clipped_boxes += 1

                x1, y1, x2, y2 = clipped
                resized_box = clip_xyxy(
                    (x1 * sx, y1 * sy, x2 * sx, y2 * sy),
                    size,
                    size,
                )
                if resized_box is None:
                    invalid_shapes += 1
                    continue

                rx1, ry1, rx2, ry2 = resized_box
                bw = rx2 - rx1
                bh = ry2 - ry1

                coco_annotations.append(
                    {
                        "id": next_ann_id,
                        "image_id": image_id,
                        "category_id": 1,
                        "bbox": [round(rx1, 4), round(ry1, 4), round(bw, 4), round(bh, 4)],
                        "area": round(bw * bh, 4),
                        "iscrowd": 0,
                    }
                )
                next_ann_id += 1
                image_object_count += 1

            objects_per_image.append(image_object_count)

    coco = {
        "info": {
            "description": f"SGC panicle images resized to {size}x{size}",
            "version": "1.0",
        },
        "licenses": [],
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": [{"id": 1, "name": "grain", "supercategory": "grain"}],
    }

    ann_path = ann_out / f"instances_{split_name}.json"
    ann_path.write_text(json.dumps(coco, ensure_ascii=False), encoding="utf-8")

    stats = {
        "split": split_name,
        "images": len(coco_images),
        "annotations": len(coco_annotations),
        "objects_per_image_mean": sum(objects_per_image) / len(objects_per_image),
        "objects_per_image_min": min(objects_per_image),
        "objects_per_image_max": max(objects_per_image),
        "invalid_shapes_skipped": invalid_shapes,
        "boxes_clipped_to_source_bounds": clipped_boxes,
        "source_label_counts": dict(label_counts),
        "image_directory": str(image_out),
        "annotation_file": str(ann_path),
    }
    return coco, next_image_id, next_ann_id, stats


def validate_coco(coco, split_name):
    image_map = {int(im["id"]): im for im in coco["images"]}
    if len(image_map) != len(coco["images"]):
        raise RuntimeError(f"{split_name}: duplicate image IDs")

    ann_ids = [int(ann["id"]) for ann in coco["annotations"]]
    if len(ann_ids) != len(set(ann_ids)):
        raise RuntimeError(f"{split_name}: duplicate annotation IDs")

    for ann in coco["annotations"]:
        image_id = int(ann["image_id"])
        if image_id not in image_map:
            raise RuntimeError(f"{split_name}: missing image for annotation {ann['id']}")
        x, y, w, h = map(float, ann["bbox"])
        im = image_map[image_id]
        if (
            w <= 0
            or h <= 0
            or x < -1e-3
            or y < -1e-3
            or x + w > im["width"] + 1e-3
            or y + h > im["height"] + 1e-3
        ):
            raise RuntimeError(f"{split_name}: invalid bbox in annotation {ann['id']}")


def main():
    args = parse_args()
    root = args.root.resolve()
    output = args.output.resolve()

    if not root.is_dir():
        raise FileNotFoundError(f"Input root not found: {root}")
    if args.size <= 0:
        raise ValueError("--size must be positive")
    if not 1 <= args.quality <= 100:
        raise ValueError("--quality must be between 1 and 100")

    if output.exists() and any(output.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {output}\nUse --overwrite to rebuild it."
            )
        print(f"Removing existing output: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    sources = {
        "trainval": [
            (
                "train",
                root / "train_img_json/train_img",
                root / "train_img_json/train_json",
            ),
            (
                "val",
                root / "val_img_json/val_img",
                root / "val_img_json/val_json",
            ),
        ],
        "test": [
            (
                "test",
                root / "test_img_json/test_img",
                root / "test_img_json/test_json",
            )
        ],
    }

    next_image_id = 1
    next_ann_id = 1
    all_stats = []

    for split_name in ("trainval", "test"):
        coco, next_image_id, next_ann_id, stats = convert_split(
            sources[split_name],
            split_name,
            output,
            args.size,
            args.quality,
            next_image_id,
            next_ann_id,
        )
        validate_coco(coco, split_name)
        all_stats.append(stats)

    stats_path = output / "conversion_statistics.json"
    stats_path.write_text(
        json.dumps(all_stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n================ Conversion completed ================")
    for stats in all_stats:
        print(f"\n[{stats['split']}]")
        print(f"Images:                      {stats['images']:,}")
        print(f"Annotations:                 {stats['annotations']:,}")
        print(
            "Objects/image:               "
            f"{stats['objects_per_image_mean']:.2f} "
            f"(min {stats['objects_per_image_min']}, max {stats['objects_per_image_max']})"
        )
        print(f"Invalid shapes skipped:      {stats['invalid_shapes_skipped']:,}")
        print(f"Boxes clipped before resize: {stats['boxes_clipped_to_source_bounds']:,}")
        print(f"Images:                      {stats['image_directory']}")
        print(f"Annotations:                 {stats['annotation_file']}")

    print(f"\nStatistics: {stats_path}")
    print(f"Output root: {output}")


if __name__ == "__main__":
    main()
