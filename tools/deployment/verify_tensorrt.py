"""Check dynamic budgets and confident detections of a BUTR TensorRT FP16 pipeline."""

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core import YAMLConfig
from tools.deployment.export_onnx import DeploymentModel
from tools.deployment.tensorrt_runtime import load_runner
from tools.inference.onnx_inf import prepare_inputs


@torch.inference_mode()
def evaluate_ap(args, cfg, model, runner):
    """Evaluate both runtimes on identical images with the YAML's evaluator."""
    loader_cfg = cfg.yaml_cfg["val_dataloader"]
    loader_cfg.pop("batch_size", None)
    loader_cfg.update(total_batch_size=1, num_workers=2, shuffle=False, drop_last=False)
    loader_cfg["dataset"].update(img_folder=args.images, ann_file=args.annotations)
    loader = cfg.val_dataloader
    evaluators = {name: copy.deepcopy(cfg.evaluator) for name in ("pytorch_fp32", "tensorrt")}
    counts = {name: [] for name in evaluators}
    image_ids = []
    for index, (images, targets) in enumerate(loader):
        if args.limit and index >= args.limit:
            break
        images = images.to(args.device)
        sizes = torch.stack([target["orig_size"] for target in targets]).to(args.device)
        reference = model(images, sizes)
        runner({"images": images, "orig_target_sizes": sizes})
        torch.cuda.synchronize(args.device)
        actual = [runner.outputs()[name] for name in ("labels", "boxes", "scores")]
        image_id = targets[0]["image_id"].item()
        image_ids.append(image_id)
        for name, prediction in (("pytorch_fp32", reference), ("tensorrt", actual)):
            labels, boxes, scores = prediction
            counts[name].append(labels.shape[1])
            evaluators[name].update(
                {
                    image_id: {
                        "labels": labels[0].cpu(),
                        "boxes": boxes[0].float().cpu(),
                        "scores": scores[0].float().cpu(),
                    }
                }
            )
        if len(image_ids) % 100 == 0:
            print(f"Evaluated {len(image_ids)} images", flush=True)
    if not image_ids:
        raise ValueError("No evaluation images were processed.")
    report = {"images": len(image_ids), "image_ids": image_ids, "batch_size": 1, "results": {}}
    for name, evaluator in evaluators.items():
        print(name, flush=True)
        evaluator.synchronize_between_processes()
        evaluator.accumulate()
        evaluator.summarize()
        stats = evaluator.coco_eval["bbox"].stats.tolist()
        report["results"][name] = {
            "bbox_stats": stats,
            "AP": stats[0] * 100,
            "mean_queries": float(np.mean(counts[name])),
            "queries": counts[name],
        }
    differences = np.abs(np.asarray(counts["pytorch_fp32"]) - np.asarray(counts["tensorrt"]))
    report["query_comparison"] = {
        "identical_images": int((differences == 0).sum()),
        "max_absolute_difference": int(differences.max()),
        "mean_absolute_difference": float(differences.mean()),
    }
    return report


def compare_confident(actual, expected, size, threshold, box_atol, score_atol):
    """Match sets symmetrically, allowing detections near the threshold to cross it."""
    from scipy.optimize import linear_sum_assignment

    worst_box = worst_score = 0.0
    for reference, observed in ((expected, actual), (actual, expected)):
        labels, boxes, scores = [value[0] for value in reference]
        other_labels, other_boxes, other_scores = [value[0] for value in observed]
        stable = np.flatnonzero(scores >= threshold + score_atol)
        for label in np.unique(labels[stable]):
            wanted = stable[labels[stable] == label]
            found = np.flatnonzero(
                (other_labels == label) & (other_scores >= threshold - score_atol)
            )
            if len(found) < len(wanted):
                raise AssertionError(
                    f"Confident class {label}: missing detections ({len(found)} < {len(wanted)})"
                )
            costs = np.abs((boxes[wanted, None] - other_boxes[None, found]) / np.tile(size, 2)).sum(
                -1
            )
            costs += np.abs(scores[wanted, None] - other_scores[None, found])
            rows, cols = linear_sum_assignment(costs)
            box_error = float(np.abs(boxes[wanted[rows]] - other_boxes[found[cols]]).max())
            score_error = float(np.abs(scores[wanted[rows]] - other_scores[found[cols]]).max())
            if box_error > box_atol or score_error > score_atol:
                raise AssertionError(
                    f"FP16 mismatch: box error={box_error:.5f}px, score error={score_error:.5f}"
                )
            worst_box, worst_score = max(worst_box, box_error), max(worst_score, score_error)
    return worst_box, worst_score


def main(args):
    from PIL import Image

    torch.set_num_threads(2)
    cfg = YAMLConfig(args.config, resume=args.resume)
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
    state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    from tools.deployment.export_onnx import load_deployment_state

    load_deployment_state(cfg.model, state)
    model = DeploymentModel(cfg.model, cfg.postprocessor).eval().to(args.device)
    runner = load_runner(args.engine, args.device, runtime_mode=args.runtime)
    spatial_size = runner.input_shape("images")[-2:]
    if tuple(cfg.yaml_cfg["eval_spatial_size"]) != spatial_size:
        raise ValueError("Checkpoint configuration and TensorRT resolution differ.")
    if args.limit < 0 or (not args.annotations and args.limit == 0):
        raise ValueError("Use a positive spot-check limit; --limit 0 requires --annotations.")
    if args.annotations:
        report = evaluate_ap(args, cfg, model, runner)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Saved {output}")
        return
    paths = [
        path
        for path in Path(args.images).iterdir()
        if path.suffix.lower() in (".jpg", ".jpeg", ".png")
    ]
    if not paths:
        raise ValueError("No images found.")
    rows, failures = [], []
    for path in paths[: args.limit]:
        with Image.open(path) as source:
            image = source.convert("RGB")
        feed = {
            name: torch.from_numpy(value).to(args.device)
            for name, value in prepare_inputs(image, spatial_size).items()
        }
        with torch.no_grad():
            expected = [
                value.cpu().numpy() for value in model(feed["images"], feed["orig_target_sizes"])
            ]
        runner(feed)
        torch.cuda.synchronize(args.device)
        actual = [runner.outputs()[name].cpu().numpy() for name in ("labels", "boxes", "scores")]
        count, ref_count = actual[0].shape[1], expected[0].shape[1]
        if abs(count - ref_count) > args.query_atol:
            failures.append(
                dict(image=str(path), error=f"Budget {count} differs from PyTorch {ref_count}.")
            )
            rows.append(dict(image=str(path), pytorch_queries=ref_count, tensorrt_queries=count))
            continue
        try:
            box_error, score_error = compare_confident(
                actual, expected, image.size, args.threshold, args.box_atol, args.score_atol
            )
        except AssertionError as error:
            failures.append(dict(image=str(path), error=str(error)))
            print(f"Numerical mismatch in {path.name}: {error}", flush=True)
            rows.append(dict(image=str(path), pytorch_queries=ref_count, tensorrt_queries=count))
            continue
        rows.append(
            dict(
                image=str(path),
                pytorch_queries=ref_count,
                tensorrt_queries=count,
                max_box_error_px=box_error,
                max_score_error=score_error,
            )
        )
        print(
            f"Verified {path.name}: queries {ref_count}/{count}, "
            f"box error {box_error:.4f}px, score error {score_error:.5f}",
            flush=True,
        )
    report = dict(
        images=rows,
        threshold=args.threshold,
        box_atol=args.box_atol,
        score_atol=args.score_atol,
        query_atol=args.query_atol,
        varying_queries=len({row["tensorrt_queries"] for row in rows}) > 1,
        failures=failures,
        passed=not failures,
        note="Numerical spot check of confident detections; this does not measure dataset AP.",
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {output}")
    if failures:
        raise AssertionError(f"{len(failures)} images exceed numerical tolerances; see {output}.")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-r", "--resume", required=True)
    parser.add_argument("--engine", required=True, help=".engine or split pipeline.json")
    parser.add_argument("--runtime", choices=("sync", "async", "cuda_graph"), default="sync")
    parser.add_argument("--images", required=True, help="directory of test images")
    parser.add_argument(
        "--annotations",
        help="evaluate AP with the YAML evaluator instead of a numerical spot check",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=32,
        help="image limit; 0 evaluates all images with --annotations",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument(
        "--box-atol", type=float, default=1.0, help="maximum coordinate difference in pixels"
    )
    parser.add_argument("--score-atol", type=float, default=0.01)
    parser.add_argument(
        "--query-atol",
        type=int,
        default=1,
        help="Ceil can shift a budget by one after FP16 rounding",
    )
    parser.add_argument("--output", default="output/tensorrt_verification.json")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
