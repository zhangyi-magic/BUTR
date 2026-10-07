"""Benchmark checkpoint inference on a single CUDA device."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-r", "--resume", required=True)
    parser.add_argument("-d", "--device", default="cuda:0")
    parser.add_argument("--warmup-iters", type=int, default=50)
    parser.add_argument("--measure-iters", type=int)
    parser.add_argument("--forward-only", action="store_true")
    parser.add_argument("--output", help="optional JSON report path")
    parser.add_argument("-u", "--update", nargs="+", help="key=value YAML overrides")
    args = parser.parse_args()

    from src.core import YAMLConfig, yaml_utils
    from src.misc.benchmark import benchmark_latency
    import torch

    updates = yaml_utils.parse_cli(args.update)
    cfg = YAMLConfig(args.config, **updates)
    cfg.resume = args.resume
    cfg.device = args.device
    cfg.yaml_cfg["val_dataloader"].pop("total_batch_size", None)
    cfg.yaml_cfg["val_dataloader"]["batch_size"] = 1
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
    state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    model = cfg.model
    model.load_state_dict(state)
    model = model.to(args.device).eval()
    stats = benchmark_latency(
        model,
        cfg.postprocessor.to(args.device).eval(),
        cfg.val_dataloader,
        args.device,
        warmup_iters=args.warmup_iters,
        measure_iters=args.measure_iters,
        measure_postprocess=not args.forward_only,
    )
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(stats, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
