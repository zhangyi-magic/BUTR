"""TensorRT latency using the D-FINE/DEIM synchronized wall-clock protocol.

Adapted from D-FINE (https://github.com/Peterande/D-FINE).
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import argparse
import glob
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def Dataset(*args, **kwargs):
    from tools.benchmark.dataset import Dataset as dataset_class

    return dataset_class(*args, **kwargs)


def load_runner(*args, **kwargs):
    from tools.deployment.tensorrt_runtime import load_runner as factory

    return factory(*args, **kwargs)


def preread_images(args):
    """Initialize the same input bytes before importing the GPU inference stack."""
    if args.no_cpu_preread:
        return None, {"enabled": False}
    if args.image_list:
        paths = [
            str(Path(args.infer_dir) / name.strip())
            for name in Path(args.image_list).read_text().splitlines()
            if name.strip()
        ]
    else:
        paths = [
            name
            for extension in ("jpg", "jpeg", "png")
            for name in glob.glob(str(Path(args.infer_dir) / ("*." + extension)))
        ]
    paths = paths[: args.iterations]
    if len(paths) != args.iterations or len(set(paths)) != len(paths):
        raise ValueError("Provide enough unique images for the requested iterations.")
    started = time.perf_counter()
    manifest_path = (
        Path(args.image_manifest)
        if args.image_manifest
        else (Path(args.image_list).with_suffix(".json") if args.image_list else None)
    )
    expected = None
    if manifest_path is not None and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("format") != "butr-benchmark-image-list-v1":
            raise ValueError("Unsupported image-list manifest.")
        expected = manifest["files"]
    elif args.image_manifest:
        raise FileNotFoundError(manifest_path)
    records = []
    for path in paths:
        content = Path(path).read_bytes()
        checksum = hashlib.sha256(content).hexdigest()
        if expected is not None:
            relative = Path(path).resolve().relative_to(Path(args.infer_dir).resolve()).as_posix()
            reference = expected.get(relative)
            if reference != {"sha256": checksum, "bytes": len(content)}:
                raise ValueError(f"Image bytes do not match the measured manifest: {relative}")
        records.append({"image": path, "bytes": len(content), "sha256": checksum})
    return paths, {
        "enabled": True,
        "images": len(records),
        "files": records,
        "verified_against_manifest": expected is not None,
        "bytes": sum(row["bytes"] for row in records),
        "seconds": time.perf_counter() - started,
        "included_in_latency": False,
    }


def benchmark(engine_path, args):
    image_paths, initialization = preread_images(args)
    import torch
    import tensorrt as trt

    runner = load_runner(engine_path, args.device, args.max_queries, runtime_mode=args.runtime)
    batch, channels, height, width = runner.input_shape("images")
    if batch != 1 or channels != 3 or height != width:
        raise ValueError("The benchmark expects batch size 1 and a fixed square image size.")
    dataset = Dataset(args.infer_dir, device=args.device, size=height, mode=args.preprocess)
    if image_paths is not None:
        dataset.im_path_list = image_paths
    elif args.image_list:
        paths = Path(args.image_list).read_text().splitlines()
        dataset.im_path_list = [str(Path(args.infer_dir) / path) for path in paths if path.strip()]
    if len(dataset) < args.iterations:
        raise ValueError(
            f"Need {args.iterations} images, found {len(dataset)}; no image repetition is performed."
        )
    warmup = {
        "images": torch.ones(
            (1, 3, height, width), device=args.device, dtype=runner.input_dtype("images")
        ),
        "orig_target_sizes": torch.tensor(
            [[width, height]], device=args.device, dtype=runner.input_dtype("orig_target_sizes")
        ),
    }
    for _ in tqdm(range(args.warmup_iters), desc="Warmup"):
        runner(warmup)
    torch.cuda.synchronize(args.device)
    timings, queries = [], []
    for index in tqdm(range(args.iterations), desc="Running Inference"):
        item = dataset[index]
        inputs = {
            "images": item["images"]
            .unsqueeze(0)
            .to(dtype=runner.input_dtype("images"))
            .contiguous(),
            "orig_target_sizes": item["orig_target_sizes"]
            .reshape(1, 2)
            .to(dtype=runner.input_dtype("orig_target_sizes")),
        }
        # Match upstream TimeProfiler: time.time(), with CUDA synchronization at both ends.
        torch.cuda.synchronize(args.device)
        started = time.time()
        runner(inputs)
        torch.cuda.synchronize(args.device)
        timings.append((time.time() - started) * 1000)
        queries.append(runner.outputs()["labels"].shape[1])
    report = {
        "engine": str(engine_path),
        "tensorrt": trt.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(args.device),
        "batch_size": 1,
        "input_size": [height, width],
        "preprocess": args.preprocess,
        "cpu_input_initialization": initialization,
        "runtime_nvtx_verbosity": "NONE",
        "warmup_iterations": args.warmup_iters,
        "measured_images": args.iterations,
        "mean_ms": float(np.mean(timings)),
        "median_ms": float(np.median(timings)),
        "p95_ms": float(np.percentile(timings, 95)),
        "fps": float(1000 / np.mean(timings)),
        "mean_queries": float(np.mean(queries)),
        "min_queries": min(queries),
        "max_queries": max(queries),
        "latencies_ms": timings,
        "queries": queries,
        "images": dataset.im_path_list[: args.iterations],
        "timing": "CUDA-synchronized wall clock; all engine calls and host budget handling included; preprocessing/upload excluded",
        "runtime_mode": args.runtime,
        "runtime_api": getattr(runner, "runtime_api", "execute_v2"),
        "postprocessing_in_engine": True,
        "fastest_subset_filter": False,
    }
    path = Path(engine_path)
    if path.suffix.lower() == ".json":
        manifest = json.loads(path.read_text())
        report["pipeline"] = manifest
        report["build"] = {}
        for stage in ("encoder", "decoder"):
            metadata_path = path.parent / (manifest[stage] + ".json")
            report["build"][stage] = (
                json.loads(metadata_path.read_text()) if metadata_path.is_file() else None
            )
    else:
        metadata_path = Path(str(engine_path) + ".json")
        report["build"] = json.loads(metadata_path.read_text()) if metadata_path.is_file() else None
    print(
        f"Engine: {engine_path}, Latency: {report['mean_ms']:.2f} ms, "
        f"FPS: {report['fps']:.2f}, mean queries: {report['mean_queries']:.2f}"
    )
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    engines = parser.add_mutually_exclusive_group(required=True)
    engines.add_argument("--engine", help="single .engine file or split pipeline.json")
    engines.add_argument("--engine-dir", "--engine_dir", help="directory of .engine files")
    parser.add_argument("--infer-dir", "--infer_dir", "--COCO_dir", dest="infer_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--runtime",
        choices=("sync", "async", "cuda_graph"),
        default="sync",
        help="sync: upstream execute_v2; async/cuda_graph: optimized execution, same timing boundary",
    )
    parser.add_argument(
        "--preprocess",
        choices=("official", "resize"),
        default="official",
        help="official: upstream aspect resize/pad; resize: BUTR validation preprocessing",
    )
    parser.add_argument(
        "--image-list", help="text file of image paths relative to infer-dir, one per line"
    )
    parser.add_argument(
        "--image-manifest",
        help="expected image hashes; default: image-list .json sidecar if present",
    )
    parser.add_argument("--warmup-iters", type=int, default=1000)
    parser.add_argument(
        "--no-cpu-preread",
        action="store_true",
        help="disable shared CPU input initialization (enabled in reported T4 experiments)",
    )
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument(
        "--max-queries", type=int, default=1500, help="capacity for dynamic output buffers"
    )
    parser.add_argument("--output", default="output/tensorrt_latency.json")
    return parser


def main(args):
    if args.iterations < 1 or args.warmup_iters < 0 or args.max_queries < 1:
        raise ValueError("iterations/max-queries must be positive and warmup-iters nonnegative.")
    engines = [Path(args.engine)] if args.engine else sorted(Path(args.engine_dir).glob("*.engine"))
    if not engines:
        raise ValueError("No TensorRT engines found.")
    reports = [benchmark(path, args) for path in engines]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(reports, indent=2) + "\n")
    print(f"Saved {output}")


if __name__ == "__main__":
    main(build_parser().parse_args())
