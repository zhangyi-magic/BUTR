"""Training and evaluation entry point for BUTR.

Modified from D-FINE (https://github.com/Peterande/D-FINE).
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import argparse


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-c", "--config", required=True, help="model YAML configuration")
    checkpoints = parser.add_mutually_exclusive_group()
    checkpoints.add_argument("-r", "--resume", help="resume model and training state")
    checkpoints.add_argument("-t", "--tuning", help="initialize model from a checkpoint")
    parser.add_argument("-d", "--device", help="device, e.g. cuda:0 or cpu")
    parser.add_argument("--seed", type=int, help="random seed")
    parser.add_argument(
        "--use-amp", action="store_true", default=None, help="enable automatic mixed precision"
    )
    parser.add_argument("--output-dir", help="checkpoint and log directory")
    parser.add_argument("--summary-dir", help="TensorBoard directory")
    parser.add_argument("--test-only", action="store_true", help="evaluate without training")
    parser.add_argument("-u", "--update", nargs="+", help="YAML overrides: key=value")
    parser.add_argument("--print-method", choices=("builtin", "rich"), default="builtin")
    parser.add_argument("--print-rank", type=int, default=0)
    parser.add_argument("--local-rank", "--local_rank", type=int, help="torchrun compatibility")
    return parser


def main(args):
    from src.core import YAMLConfig, yaml_utils
    from src.misc import dist_utils
    from src.solver import TASKS

    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)
    try:
        updates = yaml_utils.parse_cli(args.update)
        updates.update(
            {
                key: value
                for key, value in vars(args).items()
                if key != "update" and value is not None
            }
        )
        cfg = YAMLConfig(args.config, **updates)
        if args.resume or args.tuning:
            if "HGNetv2" in cfg.yaml_cfg:
                cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
        solver = TASKS[cfg.yaml_cfg["task"]](cfg)
        if args.test_only:
            solver.val()
        else:
            solver.fit()
    finally:
        dist_utils.cleanup()


if __name__ == "__main__":
    main(build_parser().parse_args())
