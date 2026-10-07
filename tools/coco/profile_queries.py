"""Average deployed-model FLOPs over query histograms from COCO evaluation."""

import argparse
import hashlib
import json
from pathlib import Path

from calflops import calculate_flops
import torch

from budget_adapter import BudgetAdapter
from evaluate import load_checkpoint, load_upstream


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--evaluation-report", type=Path, nargs="+", required=True)
    parser.add_argument("--checkpoint", type=Path, help="override checkpoint location in reports")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(args):
    reports = [(path, json.loads(path.read_text(encoding="utf-8"))) for path in args.evaluation_report]
    first = reports[0][1]
    for path, report in reports:
        for key in ("family", "checkpoint", "input_size", "weights_source"):
            if report[key] != first[key]:
                raise ValueError(f"Reports must describe the same model: {path}, {key}.")
        if Path(report["config"]).name != Path(first["config"]).name:
            raise ValueError(f"Reports use different model configurations: {path}.")
    checkpoint = args.checkpoint or Path(first["checkpoint"])
    config = Path("configs/deim_dfine") / Path(first["config"]).name
    protected = [checkpoint, args.upstream_root / config, *args.evaluation_report]
    if args.output.resolve() in {path.resolve() for path in protected}:
        raise ValueError("Output must differ from input reports, checkpoint, and configuration.")

    # Model construction needs the upstream configuration; no dataset is opened here.
    cfg, config_path, _ = load_upstream(
        argparse.Namespace(
            upstream_root=args.upstream_root,
            config=config,
            num_workers=0,
            images=Path("."),
            annotations=Path("."),
        )
    )
    model = cfg.model
    weights_source = load_checkpoint(model, checkpoint)
    if weights_source != first["weights_source"]:
        raise ValueError("Profiling checkpoint uses a different weight source from evaluation.")
    baseline_queries = model.decoder.num_queries
    model = model.to(args.device).eval().deploy()
    if list(cfg.yaml_cfg["eval_spatial_size"]) != first["input_size"]:
        raise ValueError("Profiling configuration resolution differs from evaluation.")
    sample = torch.zeros(1, 3, *first["input_size"], device=args.device)

    histograms = {}
    for path, report in reports:
        for mode, metrics in report["results"].items():
            if mode == "baseline":
                histogram = {baseline_queries: metrics["images"]}
            else:
                counts = metrics.get("decoded_queries", {}).get("histogram")
                if counts is None:
                    raise ValueError(f"Missing query histogram in {path}; rerun COCO evaluation.")
                histogram = {int(query): count for query, count in counts.items()}
            if (
                not histogram
                or any(query < 1 or count < 1 for query, count in histogram.items())
                or sum(histogram.values()) != metrics["images"]
            ):
                raise ValueError(f"Invalid query histogram in {path}: {mode}.")
            histograms[(str(path), mode)] = histogram

    def measure():
        # Match the upstream calflops tool's grad-enabled counting path.
        flops, macs, params = calculate_flops(
            model,
            args=[sample],
            output_as_string=False,
            print_results=False,
            print_detailed=False,
        )
        return dict(flops=flops, macs=macs, parameters=params)

    baseline = measure()
    queries = sorted({q for (_, mode), counts in histograms.items() if mode != "baseline" for q in counts})
    costs = {}
    if queries:
        adapter = BudgetAdapter(model.decoder, min_queries=queries[0], max_queries=queries[0], safety=1.0)
        try:
            for query in queries:
                adapter.settings.update(min_queries=query, max_queries=query)
                adapter.reset(enabled=True)
                costs[query] = measure()
                if set(adapter.decoded_counts) != {query}:
                    raise RuntimeError("Measured query count differs from the requested count.")
        finally:
            adapter.remove()

    averages = []
    for (path, mode), counts in histograms.items():
        flops = baseline["flops"] if mode == "baseline" else sum(
            costs[query]["flops"] * count for query, count in counts.items()
        ) / sum(counts.values())
        averages.append(dict(evaluation_report=path, mode=mode, images=sum(counts.values()), GFLOPs=flops / 1e9))
    output = dict(
        profiler="calflops",
        deploy=True,
        batch_size=1,
        input_size=first["input_size"],
        config=str(config_path),
        checkpoint=str(checkpoint.resolve()),
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        weights_source=weights_source,
        baseline=baseline,
        cost_by_queries=costs,
        averages=averages,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    for result in averages:
        print(f"{result['evaluation_report']} {result['mode']}: {result['GFLOPs']:.4f} GFLOPs")


if __name__ == "__main__":
    main(build_parser().parse_args())
