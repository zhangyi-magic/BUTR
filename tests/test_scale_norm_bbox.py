import unittest

import torch

from src.zoo.models.dome_criterion import DomeCriterion


class ScaleNormBBoxTests(unittest.TestCase):
    def test_optional_area_weighting_and_gradients(self):
        target = torch.tensor([[0.4, 0.4, 0.01, 0.01], [0.6, 0.6, 0.25, 0.25]])
        predicted = (target + 0.001).unsqueeze(0).requires_grad_()
        indices = [(torch.arange(2), torch.arange(2))]
        criterion = DomeCriterion(None, {}, ['boxes'])
        arguments = ({'pred_boxes': predicted}, [{'boxes': target}], indices, 2)
        original = criterion.loss_boxes(*arguments)
        self.assertFalse(criterion.scale_norm_bbox)
        torch.testing.assert_close(original['loss_bbox'], (predicted[0] - target).abs().sum() / 2)
        criterion.scale_norm_bbox = True
        weighted = criterion.loss_boxes(*arguments)
        expected = ((predicted[0] - target).abs().sum(-1) * torch.tensor([10., 2.])).sum() / 2
        torch.testing.assert_close(weighted['loss_bbox'], expected)
        torch.testing.assert_close(weighted['loss_giou'], original['loss_giou'])
        sum(weighted.values()).backward()
        self.assertTrue(torch.isfinite(predicted.grad).all())
        torch.testing.assert_close(criterion._scale_norm_weight(torch.zeros(1, 4)), torch.tensor([30.]))
        criterion.sn_apply_to_giou = True
        self.assertGreater(criterion.loss_boxes(*arguments)['loss_giou'], original['loss_giou'])

    def test_empty_targets(self):
        criterion = DomeCriterion(None, {}, ['boxes'], scale_norm_bbox=True)
        prediction = torch.zeros(1, 2, 4, requires_grad=True)
        empty = torch.empty(0, dtype=torch.long)
        losses = criterion.loss_boxes({'pred_boxes': prediction}, [{'boxes': torch.empty(0, 4)}], [(empty, empty)], 1)
        self.assertEqual(sum(losses.values()).item(), 0.)
        sum(losses.values()).backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
