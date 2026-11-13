"""Check profiling scope and isolation without running a detector training job."""

import contextlib
import importlib.util
import io
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from src.misc.profiler_utils import stats


class ToyDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(3, 4)
        self.auxiliary_head = nn.Linear(3, 4)

    def deploy(self):
        self.eval()
        del self.auxiliary_head
        return self

    def forward(self, images):
        logits = self.head(images.mean(dim=(2, 3)))
        return {"pred_logits": logits[:, None, :].expand(-1, 7, -1)}


@unittest.skipUnless(importlib.util.find_spec("calflops"), "calflops is not installed")
class ProfilingTests(unittest.TestCase):
    def test_profile_reports_actual_queries_and_preserves_training_model(self):
        for base_size in (4, (4, 6)):
            with self.subTest(base_size=base_size):
                model = ToyDetector().train()
                before = {name: value.clone() for name, value in model.state_dict().items()}
                rng = torch.get_rng_state().clone()
                cfg = SimpleNamespace(
                    model=model,
                    train_dataloader=SimpleNamespace(
                        collate_fn=SimpleNamespace(base_size=base_size)
                    ),
                )
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    params, report = stats(cfg)
                self.assertIn("Calculate Flops Results", stdout.getvalue())
                self.assertIn("Deployment parameters:", stdout.getvalue())
                self.assertNotIn("Total Training Params", stdout.getvalue())
                self.assertEqual(params, 16)
                self.assertIn("Deployment parameters: 16", report)
                self.assertIn("Synthetic-input forward FLOPs", report)
                self.assertIn("decoder queries: 7", report)
                self.assertIn("not dataset-average FLOPs", report)
                self.assertNotIn("fwd+bwd", report)
                self.assertTrue(model.training)
                self.assertTrue(hasattr(model, "auxiliary_head"))
                self.assertEqual(len(model._forward_hooks), 0)
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                for name, value in model.state_dict().items():
                    self.assertTrue(torch.equal(before[name], value))


if __name__ == "__main__":
    unittest.main()
