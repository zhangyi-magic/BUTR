"""Official Dome batching, DN padding, and detection loss semantics."""

import unittest

import torch

from src.zoo.models.dome_decoder import DomeTransformer
from src.zoo.models.dome_criterion import DomeCriterion
from src.zoo.models.matcher import HungarianMatcher


def criterion():
    return DomeCriterion(
        HungarianMatcher(dict(cost_class=2, cost_bbox=5, cost_giou=2), use_focal_loss=True),
        dict(loss_vfl=1, loss_bbox=5, loss_giou=2),
        ["vfl", "boxes"], use_defe_loss=True, num_classes=3, reg_max=8,
    )


class DomeBatchProtocolTests(unittest.TestCase):
    def test_variable_query_batch_with_dn_has_finite_losses_and_gradients(self):
        torch.manual_seed(12)
        decoder = DomeTransformer(
            num_classes=3, hidden_dim=32, feat_channels=[32, 32],
            feat_strides=[4, 8], num_levels=2, num_points=[2, 2],
            nhead=8, num_layers=2, dim_feedforward=64, num_denoising=4,
            reg_max=8, min_num_select=2, max_num_select=8,
        ).train()
        mask = torch.ones(2, 2, 2, dtype=torch.bool)
        mask[0] = False
        count = torch.full((2, 1), 0.5, requires_grad=True)
        density = torch.full((2, 1, 4, 4), 0.5, requires_grad=True)
        encoded = dict(
            feats=[torch.randn(2, 32, size, size, requires_grad=True) for size in (8, 4)],
            defe=dict(defe_window_mask=mask, density_map_pooled=torch.ones(2, 1, 2, 2),
                      density_map=density, defe_feature=density,
                      gt_density_map=torch.zeros_like(density), reg_value=count),
        )
        targets = [
            dict(labels=torch.tensor([1]), boxes=torch.tensor([[0.5, 0.5, 0.1, 0.1]])),
            dict(labels=torch.tensor([0, 2]), boxes=torch.tensor([[0.2, 0.3, 0.05, 0.06], [0.7, 0.6, 0.1, 0.1]])),
        ]
        outputs = decoder(encoded, targets, epoch=0)
        self.assertIsInstance(outputs, dict)
        self.assertEqual(outputs["batch_queries_num"], [2, 8])
        self.assertEqual(tuple(outputs["pred_logits"].shape), (2, 8, 3))
        losses = criterion()(outputs, targets, epoch=0)
        self.assertTrue(all(torch.isfinite(v).all() for v in losses.values()))
        sum(losses.values()).backward()
        self.assertIsNotNone(decoder.enc_score_head.weight.grad)
        self.assertTrue(torch.isfinite(decoder.enc_score_head.weight.grad).all())
        self.assertIsNotNone(count.grad)
        self.assertIsNotNone(density.grad)

    def test_padded_classification_queries_do_not_contribute_to_vfl(self):
        loss = criterion()
        boxes = torch.full((1, 4, 4), 0.3)
        targets = [dict(labels=torch.tensor([1]), boxes=torch.full((1, 4), 0.3))]
        indices = [(torch.tensor([0]), torch.tensor([0]))]
        first = dict(pred_logits=torch.zeros(1, 4, 3), pred_boxes=boxes)
        second = dict(pred_logits=first["pred_logits"].clone(), pred_boxes=boxes)
        second["pred_logits"][:, 1:] = 20
        a = loss.loss_labels_vfl(first, targets, indices, 1, batch_queries_num=[1])
        b = loss.loss_labels_vfl(second, targets, indices, 1, batch_queries_num=[1])
        self.assertTrue(torch.equal(a["loss_vfl"], b["loss_vfl"]))

    def test_bbox_loss_matches_standard_l1(self):
        boxes = torch.tensor([[[0.5, 0.5, 0.01, 0.01]]], requires_grad=True)
        targets = [dict(labels=torch.tensor([1]), boxes=torch.tensor([[0.51, 0.49, 0.02, 0.02]]))]
        result = criterion().loss_boxes(dict(pred_boxes=boxes), targets,
                                        [(torch.tensor([0]), torch.tensor([0]))], 1)
        expected = (boxes[0] - targets[0]["boxes"]).abs().sum()
        self.assertTrue(torch.equal(result["loss_bbox"], expected))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
