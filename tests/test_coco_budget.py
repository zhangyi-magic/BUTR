"""Check plug-in weight preservation, selection equivalence, and dynamic budgets."""

import unittest
import importlib.util
import inspect
from pathlib import Path
import tempfile

import torch

from src.zoo.models.dfine_decoder import DFINETransformer
from tools.coco.budget_adapter import BudgetAdapter, bridge_convert_pil_image


def decoder():
    return DFINETransformer(
        num_classes=3,
        hidden_dim=32,
        num_queries=12,
        feat_channels=[32, 32, 32],
        feat_strides=[8, 16, 32],
        num_levels=3,
        nhead=4,
        num_layers=2,
        dim_feedforward=64,
        num_denoising=0,
    ).eval()


class CocoBudgetTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('onnx') and importlib.util.find_spec('onnxruntime'),
                         'ONNX export dependencies are optional')
    def test_export_preserves_input_dependent_query_count(self):
        import numpy as np
        import onnxruntime as ort

        class Selection(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.decoder = decoder()
                self.adapter = BudgetAdapter(self.decoder, min_queries=1, max_queries=20, safety=2)

            def forward(self, logits):
                memory = logits[..., :1].repeat(1, 1, 32)
                anchors = logits.new_zeros((1, 10, 4))
                return self.decoder._select_topk(memory, logits, anchors, 12)[0]

        model = Selection().eval()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'selection.onnx'
            options = {'dynamo': False} if 'dynamo' in inspect.signature(torch.onnx.export).parameters else {}
            torch.onnx.export(model, torch.zeros(1, 10, 3), str(path), opset_version=16,
                              input_names=['logits'], output_names=['memory'],
                              dynamic_axes={'memory': {1: 'queries'}}, **options)
            session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
            observed = []
            for value in [-8., 0., 8.]:
                logits = torch.full((1, 10, 3), value)
                with torch.inference_mode():
                    expected = model(logits).numpy()
                actual = session.run(None, {'logits': logits.numpy()})[0]
                np.testing.assert_array_equal(actual, expected)
                observed.append(actual.shape[1])
            self.assertEqual(observed, [1, 10, 10])

    def test_legacy_deim_image_conversion_preserves_pixels(self):
        from PIL import Image
        from torchvision.transforms import v2
        from torchvision.transforms.v2 import functional

        if not hasattr(v2.Transform, "transform"):
            self.skipTest("The bridge applies to the current torchvision transform API.")

        class LegacyConversion(v2.Transform):
            _transformed_types = (Image.Image,)

            def _transform(self, image, params):
                return functional.pil_to_tensor(image).float() / 255.0

        self.assertTrue(bridge_convert_pil_image(LegacyConversion))
        image = Image.new("RGB", (7, 5), (51, 102, 153))
        converted = LegacyConversion()(image)
        self.assertEqual(tuple(converted.shape), (3, 5, 7))
        torch.testing.assert_close(converted[:, 0, 0], torch.tensor([0.2, 0.4, 0.6]))
        self.assertFalse(bridge_convert_pil_image(LegacyConversion))

    def test_disabled_and_fixed_budget_match_original_selection_and_weights(self):
        module = decoder()
        memory, logits = torch.randn(1, 30, 32), torch.randn(1, 30, 3)
        anchors = torch.randn(1, 30, 4)
        expected = module._select_topk(memory, logits, anchors, 12)
        state = {key: value.clone() for key, value in module.state_dict().items()}
        adapter = BudgetAdapter(module, min_queries=12, max_queries=12, safety=6)
        for enabled in (False, True):
            adapter.reset(enabled=enabled)
            actual = module._select_topk(memory, logits, anchors, 12)
            for before, after in zip(expected, actual):
                if before is not None:
                    torch.testing.assert_close(before, after, rtol=0, atol=0)
            self.assertEqual(set(state), set(module.state_dict()))
            for key, before in state.items():
                torch.testing.assert_close(before, module.state_dict()[key], rtol=0, atol=0)
        adapter.remove()
        self.assertNotIn("_select_topk", module.__dict__)

    def test_budget_uses_valid_anchors_and_batch_maximum(self):
        module = decoder()
        adapter = BudgetAdapter(module, min_queries=1, max_queries=20, safety=2)
        memory = torch.randn(2, 10, 32)
        logits = torch.zeros(2, 10, 3)
        logits[1] = 100
        anchors = torch.zeros(2, 10, 4)
        anchors[:, -2:] = torch.inf
        selected, _, _ = module._select_topk(memory, logits, anchors, 12)
        self.assertEqual(adapter.requested_counts, [8, 16])
        # Clip to the ten available proposals, independent of the original twelve queries.
        self.assertEqual(selected.shape[1], 10)
        self.assertEqual(adapter.decoded_counts, [10, 10])
        module.train()
        with self.assertRaisesRegex(RuntimeError, "inference-only"):
            module._select_topk(memory, logits, anchors, 12)
        adapter.remove()

    def test_dynamic_forward_supports_more_queries_than_official_fixed_count(self):
        module = decoder()
        adapter = BudgetAdapter(module, min_queries=16, max_queries=20, safety=6)
        features = [torch.randn(1, 32, size, size) for size in (8, 4, 2)]
        with torch.inference_mode():
            output = module({"feats": features})
        self.assertEqual(output["pred_logits"].shape[1], adapter.decoded_counts[0])
        self.assertGreaterEqual(output["pred_logits"].shape[1], 16)
        self.assertTrue(torch.isfinite(output["pred_boxes"]).all())
        adapter.remove()


if __name__ == "__main__":
    unittest.main()
