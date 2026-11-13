"""MWAS scatter must support mixed encoder/residual dtypes without losing gradients."""

import unittest
from unittest.mock import patch

import torch
from torch import nn

from src.zoo.models.get_roi_features import WindowProcessor


class WindowPrecisionTests(unittest.TestCase):
    def test_single_window_keeps_input_unchanged_and_backpropagates(self):
        processor = WindowProcessor(embed_dim=8, num_heads=2, dim_feedforward=16)
        features = torch.ones(1, 8, 4, 4, requires_grad=True)
        with patch.object(processor, "_encode_features", side_effect=lambda x, *_: x * 2):
            result, _ = processor(features, torch.ones(1, 1, 4, 4), 1, torch.zeros(8, 4, 4))
        self.assertTrue(torch.equal(features, torch.ones_like(features)))
        self.assertTrue(torch.equal(result, torch.full_like(result, 3)))
        result.sum().backward()
        self.assertTrue(torch.equal(features.grad, torch.full_like(features, 3)))

    def test_rectangular_window_grid_scatter_and_gradients(self):
        processor = WindowProcessor(embed_dim=8, num_heads=2, dim_feedforward=16)
        features = torch.ones(2, 8, 4, 6, requires_grad=True)
        mask = torch.zeros(2, 1, 8, 12)
        mask[0, 0, 0, 0] = 1
        mask[1, 0, -1, -1] = 1
        with patch.object(processor, "_encode_features", side_effect=lambda x, *_: x * 2):
            result, selected = processor(features, mask, (2, 3), torch.zeros(8, 4, 6))
        expected = torch.ones_like(features)
        expected[0, :, :2, :2] = 3
        expected[1, :, -2:, -2:] = 3
        self.assertTrue(torch.equal(result, expected))
        self.assertEqual(tuple(selected.shape), (2, 2, 3))
        result.sum().backward()
        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertTrue(torch.equal(features.grad, expected))

    def test_attention_is_independent_across_images(self):
        torch.manual_seed(7)
        processor = WindowProcessor(embed_dim=8, num_heads=2, dim_feedforward=16).eval()
        features = torch.randn(2, 8, 4, 4, requires_grad=True)
        mask = torch.ones(2, 1, 4, 4)
        positions = torch.randn(8, 4, 4)
        batched, _ = processor(features, mask, 2, positions)
        separate = torch.cat([
            processor(features[i:i + 1], mask[i:i + 1], 2, positions)[0]
            for i in range(2)
        ])
        self.assertTrue(torch.equal(batched, separate))
        gradient_batched, = torch.autograd.grad(batched.square().sum(), features, retain_graph=True)
        gradient_separate, = torch.autograd.grad(separate.square().sum(), features)
        self.assertTrue(torch.equal(gradient_batched, gradient_separate))

    def test_mixed_dtype_scatter_preserves_residual_and_encoder_gradient(self):
        for input_dtype, encoded_dtype in (
            (torch.float32, torch.float16),
            (torch.float16, torch.float32),
            (torch.float32, torch.bfloat16),
            (torch.float32, torch.float32),
        ):
            with self.subTest(input_dtype=input_dtype, encoded_dtype=encoded_dtype):
                processor = WindowProcessor(embed_dim=8, num_heads=2, dim_feedforward=16)
                gain = nn.Parameter(torch.tensor(0.5))
                features = torch.arange(2 * 8 * 4 * 4, dtype=input_dtype).reshape(2, 8, 4, 4)
                features = features.div(32).requires_grad_()
                original = features.detach().clone()
                mask = torch.zeros(2, 1, 4, 4)
                mask[0, 0, 0, 0] = 1
                mask[1, 0, 3, 3] = 1
                positions = torch.zeros(8, 4, 4)

                def encode(selected, relative, absolute):
                    return (selected * gain).to(encoded_dtype)

                with patch.object(processor, "_encode_features", side_effect=encode):
                    result, window_mask = processor(features, mask, 2, positions)

                expected = original.clone()
                expected[0, :, :2, :2] += (original[0, :, :2, :2] * 0.5).to(encoded_dtype).to(input_dtype)
                expected[1, :, 2:, 2:] += (original[1, :, 2:, 2:] * 0.5).to(encoded_dtype).to(input_dtype)
                self.assertEqual(result.dtype, input_dtype)
                self.assertTrue(torch.equal(result.detach(), expected))
                self.assertTrue(torch.equal(features.detach(), original))
                self.assertEqual(window_mask.sum().item(), 2)
                result.float().sum().backward()
                self.assertIsNotNone(gain.grad)
                self.assertTrue(torch.isfinite(gain.grad).all())
                self.assertGreater(gain.grad.abs().item(), 0)
                self.assertTrue(torch.isfinite(features.grad).all())


if __name__ == "__main__":
    unittest.main()
