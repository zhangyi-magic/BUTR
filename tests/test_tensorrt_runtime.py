"""Regression for shape-input values changing at a stable host address."""

import unittest

import torch

from tools.deployment.tensorrt_runtime import TensorRTRunner


class ShapeInputTests(unittest.TestCase):
    def test_exact_shapes_update_with_reused_host_buffer(self):
        runner = object.__new__(TensorRTRunner)
        budget = torch.empty(1, dtype=torch.int64)
        runner.device = torch.device("cpu")
        runner.input_names = ["budget"]
        runner.output_names = ["labels"]
        runner.names = ["budget", "labels"]
        runner.host_names = {"budget"}
        runner.shape_input_names = ["budget"]
        runner.dtypes = {"budget": torch.int64, "labels": torch.int64}
        runner.input_shapes = {"budget": (1,)}
        runner.buffers = {"labels": torch.empty(1, 15, dtype=torch.int64)}
        runner.addresses = {"labels": runner.buffers["labels"].data_ptr()}
        runner.bound_addresses = {}
        runner.bindings = [0, runner.addresses["labels"]]
        runner.dynamic_outputs = ["labels"]
        runner.shapes = {}
        runner.last_shape_values = None
        runner.output_views = None
        runner.stream = None

        class Context:
            inferred = None
            notified = None
            shape_calls = 0
            notifications = 0

            def set_tensor_address(self, name, address):
                if name == "budget":
                    self.notified = int(budget.item())
                    self.notifications += 1

            def infer_shapes(self):
                self.inferred = self.notified
                self.shape_calls += 1
                return []

            def get_tensor_shape(self, name):
                return (1, self.inferred)

            def execute_v2(self, bindings):
                runner.buffers["labels"][0, : self.inferred] = torch.arange(self.inferred)
                return True

        runner.context = Context()
        sequence = [3, 3, 4, 4, 6, 15, 3, 8, 3]
        for count in sequence:
            budget.fill_(count)
            runner({"budget": budget})
            observed = runner.outputs()["labels"]
            self.assertEqual(observed.shape, (1, count))
            torch.testing.assert_close(observed[0], torch.arange(count))
        self.assertEqual(runner.context.notifications, len(sequence))
        self.assertEqual(runner.context.shape_calls, 7)


if __name__ == "__main__":
    unittest.main()
