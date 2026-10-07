"""Prepare nonempty SGC patches and COCO annotations for the Grain experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.make_sgc_panicles_coco_1824 import (
    IMAGE_EXTS,
    clip_xyxy,
    collect_files,
    shape_to_xyxy,
    validate_coco,
)


def window_starts(length, size, stride, edge_policy="align"):
    """Cover the image, aligning the last window to the edge or padding it."""
    if length <= 0 or size <= 0 or not 0 < stride <= size:
        raise ValueError("Image/crop sizes must be positive and 0 < stride <= size.")
    if edge_policy not in {"align", "pad"}:
        raise ValueError("edge_policy must be 'align' or 'pad'.")
    if length <= size:
        return [0]
    starts = list(range(0, length - size + 1, stride))
    if starts[-1] + size < length:
        starts.append(length - size if edge_policy == "align" else starts[-1] + stride)
    return starts


def crop_box(box, x, y, size, min_visibility=0.0, box_policy="center"):
    """Clip a source box to a patch and return local COCO xywh coordinates."""
    x1, y1, x2, y2 = box
    if box_policy not in {"center", "intersection"}:
        raise ValueError("box_policy must be 'center' or 'intersection'.")
    if box_policy == "center":
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        if not (x <= center_x < x + size and y <= center_y < y + size):
            return None
    left, top = max(x1, x), max(y1, y)
    right, bottom = min(x2, x + size), min(y2, y + size)
    if right <= left or bottom <= top:
        return None
    visibility = (right - left) * (bottom - top) / ((x2 - x1) * (y2 - y1))
    if visibility < min_visibility:
        return None
    bbox = [round(left - x, 3), round(top - y, 3), round(right - left, 3), round(bottom - top, 3)]
    return bbox if bbox[2] > 0 and bbox[3] > 0 else None


def convert_split(
    root,
    output,
    split,
    source_names,
    *,
    size=800,
    stride=640,
    edge_policy="align",
    box_policy="center",
    min_visibility=0.0,
    quality=95,
):
    image_dir = output / split / "images"
    annotation_dir = output / split / "annotations"
    image_dir.mkdir(parents=True, exist_ok=True)
    annotation_dir.mkdir(parents=True, exist_ok=True)
    images, annotations = [], []
    stats = dict(
        split=split,
        source_images=0,
        source_annotations=0,
        invalid_shapes_skipped=0,
        unmatched_images=0,
        unmatched_annotations=0,
        candidate_patches=0,
        empty_patches_skipped=0,
    )
    for source in source_names:
        source_dir = root / f"{source}_img_json"
        image_files = collect_files(source_dir / f"{source}_img", IMAGE_EXTS)
        annotation_files = collect_files(source_dir / f"{source}_json", {".json"})
        stems = sorted(image_files.keys() & annotation_files.keys())
        missing_images = len(annotation_files.keys() - image_files.keys())
        missing_annotations = len(image_files.keys() - annotation_files.keys())
        stats["unmatched_images"] += missing_annotations
        stats["unmatched_annotations"] += missing_images
        if missing_images or missing_annotations:
            print(
                f"[{source}] Unpaired files: {missing_annotations} images, "
                f"{missing_images} annotations; processing matched pairs only."
            )
        if not stems:
            raise ValueError(f"No matched images and LabelMe annotations for {source}.")
        for stem in stems:
            labelme = json.loads(annotation_files[stem].read_text(encoding="utf-8"))
            with Image.open(image_files[stem]) as raw:
                image = ImageOps.exif_transpose(raw).convert("RGB")
            width, height = image.size
            json_width, json_height = labelme.get("imageWidth"), labelme.get("imageHeight")
            if json_width is not None and json_height is not None:
                if (width, height) != (int(json_width), int(json_height)):
                    raise ValueError(
                        f"EXIF-corrected dimensions do not match LabelMe: "
                        f"{image_files[stem]} image={image.size}, "
                        f"annotation={(json_width, json_height)}"
                    )
            boxes = []
            for index, shape in enumerate(labelme.get("shapes", [])):
                box = shape_to_xyxy(shape)
                box = clip_xyxy(box, width, height) if box is not None else None
                if box is None:
                    stats["invalid_shapes_skipped"] += 1
                else:
                    boxes.append((index, box))
            stats["source_images"] += 1
            stats["source_annotations"] += len(boxes)
            for y in window_starts(height, size, stride, edge_policy):
                for x in window_starts(width, size, stride, edge_policy):
                    stats["candidate_patches"] += 1
                    local_boxes = [
                        (index, box, crop_box(box, x, y, size, min_visibility, box_policy))
                        for index, box in boxes
                    ]
                    local_boxes = [item for item in local_boxes if item[2] is not None]
                    if not local_boxes:
                        stats["empty_patches_skipped"] += 1
                        continue
                    name = f"{source}__{stem}__x{x:05d}_y{y:05d}.jpg"
                    image_id = len(images) + 1
                    images.append(
                        dict(
                            id=image_id,
                            file_name=name,
                            width=size,
                            height=size,
                            source_split=source,
                            source_file_name=image_files[stem].name,
                            source_width=width,
                            source_height=height,
                            crop_origin=[x, y],
                        )
                    )
                    # PIL pads regions outside the source with black when needed.
                    image.crop((x, y, x + size, y + size)).save(
                        image_dir / name, quality=quality, subsampling=0
                    )
                    for index, source_box, bbox in local_boxes:
                        # Compute area before coordinate rounding, matching the original data.
                        x1, y1, x2, y2 = source_box
                        area = (min(x2, x + size) - max(x1, x)) * (
                            min(y2, y + size) - max(y1, y)
                        )
                        annotations.append(
                            dict(
                                id=len(annotations) + 1,
                                image_id=image_id,
                                category_id=1,
                                bbox=bbox,
                                area=round(area, 3),
                                iscrowd=0,
                                source_annotation_index=index,
                            )
                        )
            image.close()
    if not images:
        raise ValueError(f"No nonempty patches retained for {split}.")
    coco = dict(
        info=dict(
            description="SGC nonempty overlapping patches",
            crop_size=size,
            stride=stride,
            edge_policy=edge_policy,
            box_policy=box_policy,
            min_visibility=min_visibility,
        ),
        licenses=[],
        images=images,
        annotations=annotations,
        categories=[dict(id=1, name="grain", supercategory="grain")],
    )
    validate_coco(coco, split)
    path = annotation_dir / f"instances_{split}.json"
    path.write_text(json.dumps(coco, ensure_ascii=False), encoding="utf-8")
    stats.update(
        retained_patches=len(images),
        annotations=len(annotations),
        objects_per_patch=len(annotations) / len(images),
    )
    print(f"[{split}] {len(images):,} patches, {len(annotations):,} annotations")
    return stats


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="original SGC LabelMe split root")
    parser.add_argument("--output", type=Path, required=True, help="empty output dataset directory")
    parser.add_argument("--size", type=int, default=800)
    parser.add_argument("--stride", type=int, default=640)
    parser.add_argument(
        "--edge-policy",
        choices=("align", "pad"),
        default="align",
        help="align the final window to the image edge, or pad at the next stride",
    )
    parser.add_argument(
        "--box-policy",
        choices=("center", "intersection"),
        default="center",
        help="retain boxes centered in a patch (paper protocol), or any intersecting box",
    )
    parser.add_argument(
        "--min-visibility",
        type=float,
        default=0.0,
        help="additional minimum retained box area fraction after the box-policy filter",
    )
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality, 1 to 100")
    return parser


def main(args):
    if args.size <= 0 or not 0 < args.stride <= args.size:
        raise ValueError("Require size > 0 and 0 < stride <= size.")
    if not 0 <= args.min_visibility <= 1 or not 1 <= args.quality <= 100:
        raise ValueError("Require visibility in [0, 1] and JPEG quality in [1, 100].")
    root, output = args.root.resolve(), args.output.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Output must be an empty directory: {output}")
    # Validate all input directories before creating output files.
    for source in ("train", "val", "test"):
        for suffix in ("img", "json"):
            path = root / f"{source}_img_json" / f"{source}_{suffix}"
            if not path.is_dir():
                raise FileNotFoundError(path)
    stats = [
        convert_split(
            root,
            output,
            split,
            sources,
            size=args.size,
            stride=args.stride,
            edge_policy=args.edge_policy,
            box_policy=args.box_policy,
            min_visibility=args.min_visibility,
            quality=args.quality,
        )
        for split, sources in (("trainval", ("train", "val")), ("test", ("test",)))
    ]
    report = dict(
        source=str(root),
        crop_size=args.size,
        stride=args.stride,
        edge_policy=args.edge_policy,
        box_policy=args.box_policy,
        min_visibility=args.min_visibility,
        jpeg_quality=args.quality,
        splits=stats,
    )
    (output / "cropping_statistics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved cropped dataset to {output}")


if __name__ == "__main__":
    main(build_parser().parse_args())
