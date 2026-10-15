"""Exact spatial-layout equivalence and guards for exported deployment graphs."""

import importlib.util
import unittest

import numpy as np

from tools.deployment.optimize_value_layout import rewrite_value_layout


@unittest.skipUnless(importlib.util.find_spec("onnx"), "Install deployment ONNX dependencies")
class ValueLayoutTests(unittest.TestCase):
    def model(self, axis=3):
        import onnx
        from onnx import TensorProto as T, helper as h, numpy_helper as nh

        constants = [
            nh.from_array(np.array([1, 14, 2, 4], dtype=np.int64), "shape"),
            nh.from_array(np.array([axis], dtype=np.int64), "axis"),
        ]
        nodes = [
            h.make_node("Reshape", ["memory", "shape"], ["reshaped"], name="reshape"),
            h.make_node(
                "Transpose", ["reshaped"], ["transposed"], name="transpose", perm=[0, 2, 3, 1]
            ),
        ]
        outputs = []
        for index, (start, end) in enumerate(((0, 8), (8, 12), (12, 13), (13, 14))):
            for name, value in ((f"start{index}", start), (f"end{index}", end)):
                constants.append(nh.from_array(np.array([value], dtype=np.int64), name))
            nodes.append(
                h.make_node(
                    "Slice",
                    ["transposed", f"start{index}", f"end{index}", "axis"],
                    [f"level{index}"],
                    name=f"slice{index}",
                )
            )
            outputs.append(
                h.make_tensor_value_info(f"level{index}", T.FLOAT, [1, 2, 4, end - start])
            )
        nodes.append(h.make_node("Identity", ["budget"], ["budget_out"], name="keep_budget"))
        outputs.append(h.make_tensor_value_info("budget_out", T.INT64, [1]))
        graph = h.make_graph(
            nodes,
            "layout",
            [
                h.make_tensor_value_info("memory", T.FLOAT, [1, 14, 8]),
                h.make_tensor_value_info("budget", T.INT64, [1]),
            ],
            outputs,
            constants,
        )
        return h.make_model(graph, opset_imports=[h.make_opsetid("", 16)])

    def test_layout_is_bitwise_equal_and_budget_unchanged(self):
        from onnx.reference import ReferenceEvaluator

        original = self.model()
        source = original.SerializeToString()
        pipeline = {"frontier": ["memory", "budget"], "budget_tensor": "budget"}
        for settings in (None, pipeline):
            with self.subTest(split_pipeline=settings is not None):
                changed, descriptor = rewrite_value_layout(original, settings)
                self.assertEqual(original.SerializeToString(), source)
                for budget in (3, 7, 12):
                    feed = {
                        "memory": np.random.default_rng(budget)
                        .normal(size=(1, 14, 8))
                        .astype(np.float32),
                        "budget": np.array([budget], dtype=np.int64),
                    }
                    before = ReferenceEvaluator(original).run(None, feed)
                    after = ReferenceEvaluator(changed).run(None, feed)
                    for left, right in zip(before, after):
                        np.testing.assert_array_equal(left, right)
                self.assertEqual(len(descriptor["level_names"]), 4)

    def test_wrong_slice_axis_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "spatial-token axis"):
            rewrite_value_layout(self.model(axis=2))

    def test_budget_dependent_layout_is_rejected(self):
        model = self.model()
        model.graph.node[0].input[1] = "budget"
        with self.assertRaisesRegex(ValueError, "independent of the query budget"):
            rewrite_value_layout(
                model, {"frontier": ["memory", "budget"], "budget_tensor": "budget"}
            )

    def test_missing_or_repeated_rewrite_is_rejected(self):
        model = self.model()
        changed, _ = rewrite_value_layout(model)
        with self.assertRaisesRegex(ValueError, "found 0"):
            rewrite_value_layout(changed)


if __name__ == "__main__":
    unittest.main()
