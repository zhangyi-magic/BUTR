#!/usr/bin/env python3
"""
Summarize the corrected SGC 800x800 patch dataset.

Usage:
    python summarize_sgc_patches.py \
        --root data/grain

Outputs:
    <root>/dataset_statistics/summary.json
    <root>/dataset_statistics/split_summary.csv
    <root>/dataset_statistics/object_count_quantiles.csv
    <root>/dataset_statistics/box_size_quantiles.csv
    <root>/dataset_statistics/area_bins.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Iterable, Sequence


def percentile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile; q is in [0, 100]."""
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = (len(sorted_values) - 1) * q / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(sorted_values[lo])
    frac = pos - lo
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


def describe(values: Iterable[float]) -> dict[str, float | int]:
    vals = sorted(float(v) for v in values)
    if not vals:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "p01": float("nan"),
            "p05": float("nan"),
            "p10": float("nan"),
            "p25": float("nan"),
            "median": float("nan"),
            "p75": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "max": float("nan"),
        }
    return {
        "count": len(vals),
        "mean": mean(vals),
        "std": pstdev(vals),
        "min": vals[0],
        "p01": percentile(vals, 1),
        "p05": percentile(vals, 5),
        "p10": percentile(vals, 10),
        "p25": percentile(vals, 25),
        "median": median(vals),
        "p75": percentile(vals, 75),
        "p90": percentile(vals, 90),
        "p95": percentile(vals, 95),
        "p99": percentile(vals, 99),
        "max": vals[-1],
    }


def infer_paths(root: Path, split: str) -> tuple[Path, Path]:
    img_dir = root / split / "images"
    candidates = [
        root / split / "annotations" / f"instances_{split}.json",
        root / split / "annotations.json",
    ]
    ann_path = next((p for p in candidates if p.exists()), None)
    if ann_path is None:
        raise FileNotFoundError(
            f"Cannot find annotation JSON for {split}. Tried: "
            + ", ".join(str(p) for p in candidates)
        )
    if not img_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    return img_dir, ann_path


def summarize_split(root: Path, split: str) -> dict:
    img_dir, ann_path = infer_paths(root, split)
    data = json.loads(ann_path.read_text(encoding="utf-8"))

    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])

    image_by_id = {int(im["id"]): im for im in images}
    anns_by_image: dict[int, list[dict]] = defaultdict(list)
    category_counts: Counter[int] = Counter()

    duplicate_ann_ids = len(annotations) - len({int(a["id"]) for a in annotations})
    missing_image_refs = 0
    invalid_boxes = 0
    out_of_bounds_boxes = 0
    area_field_mismatches = 0

    widths: list[float] = []
    heights: list[float] = []
    areas: list[float] = []
    sqrt_areas: list[float] = []
    aspect_ratios: list[float] = []

    # Explicit pixel-area bins. These names are descriptive, not claims about
    # any evaluator's built-in area definitions.
    area_bins = Counter(
        {
            "area_lt_8^2": 0,
            "area_8^2_to_16^2": 0,
            "area_16^2_to_32^2": 0,
            "area_32^2_to_96^2": 0,
            "area_ge_96^2": 0,
        }
    )

    for ann in annotations:
        image_id = int(ann["image_id"])
        if image_id not in image_by_id:
            missing_image_refs += 1
            continue

        anns_by_image[image_id].append(ann)
        category_counts[int(ann["category_id"])] += 1

        bbox = ann.get("bbox", [])
        if len(bbox) != 4:
            invalid_boxes += 1
            continue

        x, y, w, h = map(float, bbox)
        if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
            invalid_boxes += 1
            continue

        im = image_by_id[image_id]
        image_w = float(im["width"])
        image_h = float(im["height"])
        eps = 1e-4
        if x < -eps or y < -eps or x + w > image_w + eps or y + h > image_h + eps:
            out_of_bounds_boxes += 1

        computed_area = w * h
        ann_area = float(ann.get("area", computed_area))
        if abs(ann_area - computed_area) > max(1e-3, 1e-4 * computed_area):
            area_field_mismatches += 1

        widths.append(w)
        heights.append(h)
        areas.append(computed_area)
        sqrt_areas.append(math.sqrt(computed_area))
        aspect_ratios.append(w / h)

        if computed_area < 8**2:
            area_bins["area_lt_8^2"] += 1
        elif computed_area < 16**2:
            area_bins["area_8^2_to_16^2"] += 1
        elif computed_area < 32**2:
            area_bins["area_16^2_to_32^2"] += 1
        elif computed_area < 96**2:
            area_bins["area_32^2_to_96^2"] += 1
        else:
            area_bins["area_ge_96^2"] += 1

    object_counts = [len(anns_by_image[int(im["id"])]) for im in images]
    zero_gt_images = sum(c == 0 for c in object_counts)

    json_filenames = [str(im["file_name"]) for im in images]
    missing_files = [name for name in json_filenames if not (img_dir / name).is_file()]
    disk_image_count = sum(
        1
        for p in img_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    )

    image_sizes = Counter((int(im["width"]), int(im["height"])) for im in images)

    total_area = sum(areas)
    total_canvas_area = sum(float(im["width"]) * float(im["height"]) for im in images)
    annotation_coverage_ratio = (
        total_area / total_canvas_area if total_canvas_area else float("nan")
    )

    result = {
        "split": split,
        "paths": {
            "image_dir": str(img_dir),
            "annotation_json": str(ann_path),
        },
        "num_images_json": len(images),
        "num_images_on_disk": disk_image_count,
        "num_annotations": len(annotations),
        "num_categories": len(categories),
        "categories": categories,
        "category_annotation_counts": dict(sorted(category_counts.items())),
        "image_size_counts": {f"{w}x{h}": count for (w, h), count in sorted(image_sizes.items())},
        "images_with_zero_annotations": zero_gt_images,
        "images_with_annotations": len(images) - zero_gt_images,
        "object_count_per_image": describe(object_counts),
        "bbox_width_px": describe(widths),
        "bbox_height_px": describe(heights),
        "bbox_area_px2": describe(areas),
        "bbox_sqrt_area_px": describe(sqrt_areas),
        "bbox_aspect_ratio_w_over_h": describe(aspect_ratios),
        "area_bins": dict(area_bins),
        "area_bin_percentages": {
            key: (value / len(areas) * 100.0 if areas else float("nan"))
            for key, value in area_bins.items()
        },
        "sum_bbox_area_over_sum_image_area": annotation_coverage_ratio,
        "integrity_checks": {
            "duplicate_annotation_ids": duplicate_ann_ids,
            "annotations_referencing_missing_image_ids": missing_image_refs,
            "invalid_or_nonpositive_boxes": invalid_boxes,
            "boxes_outside_image_bounds": out_of_bounds_boxes,
            "annotation_area_field_mismatches": area_field_mismatches,
            "json_images_missing_on_disk": len(missing_files),
            "missing_file_examples": missing_files[:10],
        },
    }
    return result


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(x: float | int) -> str:
    if isinstance(x, int):
        return f"{x:,}"
    if math.isnan(float(x)):
        return "nan"
    return f"{float(x):,.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["trainval", "test"],
    )
    args = parser.parse_args()

    root = args.root.resolve()
    out_dir = root / "dataset_statistics"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = [summarize_split(root, split) for split in args.splits]

    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    split_rows = []
    for r in results:
        obj = r["object_count_per_image"]
        split_rows.append(
            {
                "split": r["split"],
                "images": r["num_images_json"],
                "annotations": r["num_annotations"],
                "positive_images": r["images_with_annotations"],
                "zero_gt_images": r["images_with_zero_annotations"],
                "objects_per_image_mean": obj["mean"],
                "objects_per_image_std": obj["std"],
                "objects_per_image_median": obj["median"],
                "objects_per_image_min": obj["min"],
                "objects_per_image_max": obj["max"],
                "bbox_width_mean_px": r["bbox_width_px"]["mean"],
                "bbox_height_mean_px": r["bbox_height_px"]["mean"],
                "bbox_area_mean_px2": r["bbox_area_px2"]["mean"],
                "bbox_sqrt_area_mean_px": r["bbox_sqrt_area_px"]["mean"],
                "annotation_coverage_percent": 100.0 * r["sum_bbox_area_over_sum_image_area"],
            }
        )

    write_csv(
        out_dir / "split_summary.csv",
        list(split_rows[0].keys()),
        split_rows,
    )

    quantile_keys = [
        "min",
        "p01",
        "p05",
        "p10",
        "p25",
        "median",
        "p75",
        "p90",
        "p95",
        "p99",
        "max",
        "mean",
        "std",
    ]

    object_rows = []
    box_rows = []
    area_rows = []

    for r in results:
        for key in quantile_keys:
            object_rows.append(
                {
                    "split": r["split"],
                    "statistic": key,
                    "objects_per_image": r["object_count_per_image"][key],
                }
            )

        for metric in [
            "bbox_width_px",
            "bbox_height_px",
            "bbox_area_px2",
            "bbox_sqrt_area_px",
            "bbox_aspect_ratio_w_over_h",
        ]:
            for key in quantile_keys:
                box_rows.append(
                    {
                        "split": r["split"],
                        "metric": metric,
                        "statistic": key,
                        "value": r[metric][key],
                    }
                )

        for bin_name, count in r["area_bins"].items():
            area_rows.append(
                {
                    "split": r["split"],
                    "area_bin": bin_name,
                    "count": count,
                    "percentage": r["area_bin_percentages"][bin_name],
                }
            )

    write_csv(
        out_dir / "object_count_quantiles.csv",
        ["split", "statistic", "objects_per_image"],
        object_rows,
    )
    write_csv(
        out_dir / "box_size_quantiles.csv",
        ["split", "metric", "statistic", "value"],
        box_rows,
    )
    write_csv(
        out_dir / "area_bins.csv",
        ["split", "area_bin", "count", "percentage"],
        area_rows,
    )

    print("\n================ SGC PATCH DATASET SUMMARY ================")
    for r in results:
        obj = r["object_count_per_image"]
        print(f"\n[{r['split']}]")
        print(f"Images:                 {r['num_images_json']:,}")
        print(f"Annotations:            {r['num_annotations']:,}")
        print(f"Image sizes:            {r['image_size_counts']}")
        print(f"Zero-GT images:         {r['images_with_zero_annotations']:,}")
        print(
            "Objects/image:          "
            f"{obj['mean']:.2f} ± {obj['std']:.2f} "
            f"(median {obj['median']:.1f}, "
            f"min {obj['min']:.0f}, max {obj['max']:.0f})"
        )
        print(
            "Objects/image quantile: "
            f"P25={obj['p25']:.1f}, P75={obj['p75']:.1f}, "
            f"P90={obj['p90']:.1f}, P95={obj['p95']:.1f}, "
            f"P99={obj['p99']:.1f}"
        )
        print(
            "BBox width (px):        "
            f"mean={r['bbox_width_px']['mean']:.2f}, "
            f"median={r['bbox_width_px']['median']:.2f}, "
            f"P5={r['bbox_width_px']['p05']:.2f}, "
            f"P95={r['bbox_width_px']['p95']:.2f}"
        )
        print(
            "BBox height (px):       "
            f"mean={r['bbox_height_px']['mean']:.2f}, "
            f"median={r['bbox_height_px']['median']:.2f}, "
            f"P5={r['bbox_height_px']['p05']:.2f}, "
            f"P95={r['bbox_height_px']['p95']:.2f}"
        )
        print(
            "Sqrt bbox area (px):    "
            f"mean={r['bbox_sqrt_area_px']['mean']:.2f}, "
            f"median={r['bbox_sqrt_area_px']['median']:.2f}, "
            f"P5={r['bbox_sqrt_area_px']['p05']:.2f}, "
            f"P95={r['bbox_sqrt_area_px']['p95']:.2f}"
        )
        print(f"Summed bbox coverage:   {100.0 * r['sum_bbox_area_over_sum_image_area']:.3f}%")
        print("Area bins:")
        for key, value in r["area_bins"].items():
            pct = r["area_bin_percentages"][key]
            print(f"  {key:22s} {value:9,d} ({pct:6.2f}%)")

        integrity = r["integrity_checks"]
        print("Integrity checks:")
        for key, value in integrity.items():
            if key != "missing_file_examples":
                print(f"  {key}: {value}")

    print("\nSaved:")
    for filename in [
        "summary.json",
        "split_summary.csv",
        "object_count_quantiles.csv",
        "box_size_quantiles.csv",
        "area_bins.csv",
    ]:
        print(" ", out_dir / filename)


if __name__ == "__main__":
    main()
