"""Evaluate official COCO D-FINE/DEIM checkpoints with a BUTR inference plug-in."""

import argparse
import importlib
import json
import os
from pathlib import Path
import sys

import torch

from budget_adapter import BudgetAdapter, bridge_convert_pil_image


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--upstream-root",
        type=Path,
        required=True,
        help="DEIM source checkout, used for both checkpoint families",
    )
    parser.add_argument(
        "--family",
        choices=("dfine", "deim"),
        required=True,
        help="checkpoint family; both use DEIM's D-FINE implementation",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        required=True,
        help="official YAML, relative to upstream-root or absolute",
    )
    parser.add_argument("-r", "--resume", type=Path, required=True, help="official checkpoint")
    parser.add_argument("--images", type=Path, required=True, help="COCO val2017 images")
    parser.add_argument("--annotations", type=Path, required=True, help="instances_val2017.json")
    parser.add_argument("--budget-config", type=Path, help="YAML containing budgeter settings")
    parser.add_argument("--safety", type=float, help="override the budget profile safety factor")
    parser.add_argument("--mode", choices=("baseline", "budget", "paired"), default="paired")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-images", type=int, help="limit images for a smoke run")
    parser.add_argument("--output", type=Path, required=True, help="JSON evaluation report")
    return parser


def load_upstream(args):
    root = args.upstream_root.resolve()
    package = "engine"
    if not (root / package / "core").is_dir():
        raise ValueError(
            f"Expected an official DEIM checkout containing engine/core at {root}; "
            "both D-FINE and DEIM checkpoints use this checkout."
        )
    if package in sys.modules:
        raise RuntimeError(f"{package} is already imported; run this as a standalone command.")
    sys.path.insert(0, str(root))
    core = importlib.import_module(f"{package}.core")
    transforms = importlib.import_module(f"{package}.data.transforms._transforms")
    image_conversion_bridge = bridge_convert_pil_image(transforms.ConvertPILImage)
    path = args.config if args.config.is_absolute() else root / args.config
    cfg = core.YAMLConfig(str(path.resolve()))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    if cfg.yaml_cfg.get("num_classes") != 80 or not cfg.yaml_cfg.get("remap_mscoco_category"):
        raise ValueError("Use an official 80-class COCO config with category remapping enabled.")
    loader = cfg.yaml_cfg["val_dataloader"]
    loader.pop("batch_size", None)
    loader.update(total_batch_size=1, num_workers=args.num_workers, shuffle=False, drop_last=False)
    loader["dataset"].update(
        img_folder=str(args.images.resolve()), ann_file=str(args.annotations.resolve())
    )
    return cfg, path.resolve(), image_conversion_bridge


def load_checkpoint(model, path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "ema" in checkpoint:
        state = checkpoint["ema"]["module"]
        source = "ema.module"
    elif "model" in checkpoint:
        state = checkpoint["model"]
        source = "model"
    else:
        state, source = checkpoint, "state_dict"
    if state and all(key.startswith("module.") for key in state):
        state = {key[7:]: value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    return source


@torch.inference_mode()
def evaluate(model, postprocessor, loader, evaluator, device, max_images=None):
    evaluator.cleanup()
    evaluated = 0
    for samples, targets in loader:
        if max_images is not None and evaluated >= max_images:
            break
        samples = samples.to(device)
        outputs = model(samples)
        sizes = torch.stack([target["orig_size"] for target in targets]).to(device)
        predictions = postprocessor(outputs, sizes)
        evaluator.update(
            {
                target["image_id"].item(): prediction
                for target, prediction in zip(targets, predictions)
            }
        )
        evaluated += len(targets)
        if evaluated % 500 == 0:
            print(f"Evaluated {evaluated} images", flush=True)
    if evaluated == 0:
        raise ValueError("No evaluation images were processed.")
    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()
    return dict(images=evaluated, coco_eval_bbox=evaluator.coco_eval["bbox"].stats.tolist())


def main(args):
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise ValueError("Run this paired evaluation in a single process on one device.")
    if args.num_workers < 0 or (args.max_images is not None and args.max_images < 1):
        raise ValueError("Require nonnegative num-workers and positive max-images.")
    cfg, config_path, image_conversion_bridge = load_upstream(args)
    protected_paths = [args.resume.resolve(), args.annotations.resolve(), config_path]
    if args.budget_config is not None:
        protected_paths.append(args.budget_config.resolve())
    if args.output.resolve() in protected_paths:
        raise ValueError(
            "Output must differ from checkpoint, annotations, and configuration files."
        )
    model = cfg.model
    if type(model.decoder).__name__ != "DFINETransformer":
        raise ValueError("This entry point supports the D-FINE decoder used by D-FINE and DEIM.")
    weights_source = load_checkpoint(model, args.resume)
    device = torch.device(args.device)
    model = model.to(device).eval()
    postprocessor = cfg.postprocessor.to(device).eval()
    adapter = None
    if args.mode != "baseline":
        if args.budget_config is None:
            raise ValueError("Specify --budget-config for budget or paired evaluation.")
        import yaml

        settings = yaml.safe_load(args.budget_config.read_text(encoding="utf-8"))["budgeter"]
        if args.safety is not None:
            settings["safety"] = args.safety
        required = minimum_queries_for_postprocessor(postprocessor)
        if settings["min_queries"] < required:
            raise ValueError(f"min_queries must be >= {required} for the official postprocessor.")
        adapter = BudgetAdapter(model.decoder, **settings)
    report = dict(
        family=args.family,
        implementation="DEIM.engine",
        upstream_root=str(args.upstream_root.resolve()),
        config=str(config_path),
        checkpoint=str(args.resume.resolve()),
        weights_source=weights_source,
        device=str(device),
        batch_size=1,
        max_images=args.max_images,
        input_size=cfg.yaml_cfg.get("eval_spatial_size"),
        budgeter=adapter.settings if adapter else None,
        legacy_image_conversion_bridge=image_conversion_bridge,
        results={},
    )
    try:
        modes = ("baseline", "budget") if args.mode == "paired" else (args.mode,)
        for mode in modes:
            if adapter:
                adapter.reset(enabled=mode == "budget")
            print(f"Evaluating {args.family}: {mode}", flush=True)
            metrics = evaluate(
                model, postprocessor, cfg.val_dataloader, cfg.evaluator, device, args.max_images
            )
            if adapter:
                metrics["decoded_queries"] = adapter.summary()
            report["results"][mode] = metrics
    finally:
        if adapter:
            adapter.remove()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved evaluation report to {args.output}")


def minimum_queries_for_postprocessor(postprocessor):
    """Keep the upstream detection Top-K feasible without changing postprocessing."""
    import math

    return math.ceil(postprocessor.num_top_queries / postprocessor.num_classes)


if __name__ == "__main__":
    main(build_parser().parse_args())
