"""CPU checks of the TensorRT benchmark protocol, independent of TensorRT installation."""

import contextlib
import io
import hashlib
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

from tools.benchmark.trt_benchmark import benchmark, build_parser, preread_images


class BenchmarkTests(unittest.TestCase):
    def test_input_initialization_reads_pinned_order_and_hashes_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "b.jpg").write_bytes(b"second")
            (root / "a.jpg").write_bytes(b"first")
            names = root / "images.txt"
            names.write_text("b.jpg\na.jpg\n")
            args = build_parser().parse_args(
                [
                    "--engine",
                    "model.engine",
                    "--infer-dir",
                    directory,
                    "--image-list",
                    str(names),
                    "--iterations",
                    "2",
                ]
            )
            paths, metadata = preread_images(args)
            self.assertEqual([Path(path).name for path in paths], ["b.jpg", "a.jpg"])
            self.assertEqual(metadata["files"][0]["sha256"], hashlib.sha256(b"second").hexdigest())
            self.assertFalse(metadata["included_in_latency"])
            names.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "format": "butr-benchmark-image-list-v1",
                        "files": {
                            "b.jpg": {"sha256": hashlib.sha256(b"second").hexdigest(), "bytes": 6},
                            "a.jpg": {"sha256": hashlib.sha256(b"first").hexdigest(), "bytes": 5},
                        },
                    }
                )
            )
            _, metadata = preread_images(args)
            self.assertTrue(metadata["verified_against_manifest"])
            (root / "a.jpg").write_bytes(b"modified")
            with self.assertRaisesRegex(ValueError, "do not match"):
                preread_images(args)
            names.write_text("b.jpg\nb.jpg\n")
            with self.assertRaisesRegex(ValueError, "unique"):
                preread_images(args)

    def test_official_defaults(self):
        args = build_parser().parse_args(["--engine", "model.engine", "--infer-dir", "images"])
        self.assertEqual(
            (args.warmup_iters, args.iterations, args.preprocess), (1000, 1000, "official")
        )

    def test_timer_excludes_loading_and_uses_all_samples(self):
        events = []

        class Runner:
            def __init__(self, *args, **kwargs):
                self.calls = 0

            def input_shape(self, name):
                return (1, 3, 640, 640)

            def input_dtype(self, name):
                return torch.float32 if name == "images" else torch.int64

            def __call__(self, inputs):
                self.calls += 1
                events.append("execute")
                assert inputs["images"].shape == (1, 3, 640, 640)
                assert inputs["orig_target_sizes"].shape == (1, 2)

            def outputs(self):
                return {"labels": torch.empty(1, self.calls)}

        class Images:
            im_path_list = ["a.jpg", "b.jpg", "c.jpg"]

            def __init__(self, *args, **kwargs):
                pass

            def __len__(self):
                return 3

            def __getitem__(self, index):
                events.append("load")
                return {
                    "images": torch.zeros(3, 640, 640),
                    "orig_target_sizes": torch.tensor([900, 600]),
                }

        args = build_parser().parse_args(
            [
                "--engine",
                "model.engine",
                "--infer-dir",
                "images",
                "--device",
                "cpu",
                "--warmup-iters",
                "2",
                "--iterations",
                "3",
                "--no-cpu-preread",
            ]
        )
        times = iter([1.0, 1.003, 2.0, 2.008, 3.0, 3.001])

        def clock():
            events.append("clock")
            return next(times)

        with (
            patch.dict("sys.modules", {"tensorrt": types.SimpleNamespace(__version__="10.4.0")}),
            patch("tools.benchmark.trt_benchmark.load_runner", Runner),
            patch("tools.benchmark.trt_benchmark.Dataset", Images),
            patch("torch.cuda.synchronize", side_effect=lambda *args: events.append("sync")),
            patch("torch.cuda.get_device_name", return_value="test GPU"),
            patch("tools.benchmark.trt_benchmark.time.time", side_effect=clock),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            report = benchmark("model.engine", args)
        np.testing.assert_allclose(report["latencies_ms"], [3, 8, 1])
        self.assertAlmostEqual(report["mean_ms"], 4)
        self.assertEqual(report["queries"], [3, 4, 5])
        self.assertFalse(report["fastest_subset_filter"])
        self.assertEqual(
            events,
            ["execute", "execute", "sync"]
            + ["load", "sync", "clock", "execute", "sync", "clock"] * 3,
        )
