"""CPU check of the actual builder configuration, including None-returning profile API."""

import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from tools.deployment.build_tensorrt import build_engine


class BuilderTests(unittest.TestCase):
    def test_measured_policy_and_shape_profile_are_retained(self):
        class Layer:
            def __init__(self, name, kind):
                self.name, self.type = name, kind
                self.num_outputs = 1
                self.precision = None
                self.output = types.SimpleNamespace(dtype="float32")
                self.assigned = None

            def get_output(self, index):
                return self.output

            def set_output_type(self, index, dtype):
                self.assigned = dtype

        layers = [
            Layer("/backbone/BatchNormalization", "other"),
            Layer("/decoder/fc", "matrix"),
            Layer("/decoder/Slice", "other"),
            Layer("/encoder/norm", "other"),
        ]
        query = types.SimpleNamespace(name="budget", is_shape_tensor=True, shape=(1,))
        network = types.SimpleNamespace(
            num_layers=len(layers),
            num_inputs=1,
            get_layer=lambda i: layers[i],
            get_input=lambda i: query,
        )
        flags = set()
        cleared = set()
        profiles = []
        config = types.SimpleNamespace(
            set_flag=flags.add, clear_flag=cleared.add, add_optimization_profile=profiles.append
        )
        shape_values = {}
        profile = types.SimpleNamespace(
            set_shape_input=lambda name, *values: shape_values.update({name: values}),
            get_shape_input=lambda name: shape_values[name],
        )
        builder = types.SimpleNamespace(
            create_network=lambda flags: network,
            create_builder_config=lambda: config,
            create_optimization_profile=lambda: profile,
            build_serialized_network=lambda network, config: b"test-engine",
        )
        trt = types.SimpleNamespace(
            __version__="10.4.0",
            float32="float32",
            float16="float16",
            Logger=type("Logger", (), {"INFO": 1, "__init__": lambda self, level: None}),
            init_libnvinfer_plugins=lambda *a: None,
            Builder=lambda logger: builder,
            NetworkDefinitionCreationFlag=types.SimpleNamespace(EXPLICIT_BATCH=0),
            OnnxParser=lambda *a: types.SimpleNamespace(parse_from_file=lambda path: True),
            BuilderFlag=types.SimpleNamespace(
                FP16="FP16", TF32="TF32", OBEY_PRECISION_CONSTRAINTS="OBEY"
            ),
            ProfilingVerbosity=types.SimpleNamespace(DETAILED="DETAILED"),
            LayerType=types.SimpleNamespace(
                CONVOLUTION="conv", MATRIX_MULTIPLY="matrix", PLUGIN_V2="plugin"
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "model.onnx"
            source.write_bytes(b"original-graph")
            output = Path(directory) / "model.engine"
            with (
                patch.dict("sys.modules", {"tensorrt": trt}),
                patch("torch.cuda.get_device_name", return_value="Tesla T4"),
                patch("torch.cuda.get_device_capability", return_value=(7, 5)),
            ):
                build_engine(
                    source,
                    output,
                    shape_profile={"budget": [[300], [600], [1500]]},
                    fp32_sensitive=True,
                )
            metadata = json.loads(output.with_suffix(".engine.json").read_text())
        self.assertEqual(config.avg_timing_iterations, 8)
        self.assertEqual(config.profiling_verbosity, "DETAILED")
        self.assertIn("TF32", cleared)
        self.assertIn("OBEY", flags)
        self.assertIsNone(layers[0].precision)
        self.assertIsNone(layers[1].precision)
        self.assertEqual([layer.precision for layer in layers[2:]], ["float32", "float32"])
        self.assertEqual(shape_values["budget"], ([300], [600], [1500]))
        self.assertEqual(len(profiles), 1)
        self.assertFalse(metadata["TF32_enabled"])
        self.assertEqual(metadata["precision_policy"], "common-sensitive-fp32-v1")


if __name__ == "__main__":
    unittest.main()
