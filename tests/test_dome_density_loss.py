"""Dome's discrete density selection must retain direct DeFE supervision."""

import unittest

import torch

from src.zoo.models.defe import LiteDeFE
from src.zoo.models.dome_criterion import DomeCriterion
from src.zoo.models.matcher import HungarianMatcher


def criterion(use_defe_loss=True):
    return DomeCriterion(
        HungarianMatcher(dict(cost_class=2, cost_bbox=5, cost_giou=2), use_focal_loss=True),
        dict(loss_vfl=1, loss_bbox=5, loss_giou=2),
        ["vfl", "boxes"],
        use_defe_loss=use_defe_loss, num_classes=3,
        defe_density_map_weight=2,
        density_recall_penalty=0.3,
    )


def detection_outputs(batch_size=1):
    predictions = dict(
        pred_logits=torch.zeros(batch_size, 4, 3, requires_grad=True),
        pred_boxes=torch.full((batch_size, 4, 4), 0.3, requires_grad=True),
    )
    return dict(
        **predictions,
        aux_outputs=[],
        pre_outputs=dict(predictions),
        enc_aux_outputs=[],
        enc_meta={"class_agnostic": False},
    )


def targets():
    return [dict(labels=torch.tensor([1]), boxes=torch.tensor([[0.5, 0.5, 0.1, 0.1]]))]


class DensityLossTests(unittest.TestCase):
    def test_all_density_branch_parameters_receive_gradients(self):
        model = LiteDeFE().train()
        density, count = model(torch.randn(1, 256, 8, 8))
        output = detection_outputs()
        output["defe"] = dict(defe_feature=density, gt_density_map=torch.zeros_like(density),
                              reg_value=count, min_num_select=3, max_num_select=8)
        losses = criterion()(output, targets(), epoch=0)
        self.assertIn("defe_density_loss", losses)
        self.assertIn("defe_reg_loss", losses)
        sum(losses.values()).backward()
        parameters = list(model.named_parameters())
        self.assertTrue(any(name.startswith("regression_head.") for name, _ in parameters))
        for name, parameter in parameters:
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_weighted_density_and_count_loss_use_whole_batch(self):
        output = detection_outputs(batch_size=2)
        density = torch.tensor([0.25, 0.75], dtype=torch.float16).reshape(2, 1, 1, 1).requires_grad_()
        count = torch.full((2, 1), 0.5, requires_grad=True)
        output["defe"] = dict(
            defe_feature=density, gt_density_map=torch.ones(2, 1, 1, 1),
            reg_value=count, min_num_select=3, max_num_select=8,
        )
        losses = criterion()(output, targets() * 2, epoch=0)
        # Both predictions underestimate the target: penalty=1+0.3, weight=2.
        expected = ((0.75**2 + 0.25**2) / 2) * 1.3 * 2
        self.assertEqual(losses["defe_density_loss"].dtype, torch.float32)
        self.assertAlmostEqual(losses["defe_density_loss"].item(), expected, places=6)
        self.assertAlmostEqual(losses["defe_reg_loss"].item(), 0.25, places=6)
        (losses["defe_density_loss"] + losses["defe_reg_loss"]).backward()
        self.assertIsNotNone(density.grad)
        self.assertIsNotNone(count.grad)

    def test_detection_outputs_without_defe_do_not_add_density_losses(self):
        output = detection_outputs()
        losses = criterion(use_defe_loss=False)(output, targets(), epoch=0)
        self.assertNotIn("defe_density_loss", losses)
        self.assertNotIn("defe_reg_loss", losses)

    def test_disabled_defe_ignores_empty_or_present_branch(self):
        for branch in ({}, {"reg_value": torch.zeros(1, 1)}):
            output = detection_outputs()
            output["defe"] = branch
            losses = criterion(use_defe_loss=False)(output, targets())
            self.assertNotIn("defe_density_loss", losses)
            self.assertNotIn("defe_reg_loss", losses)

    def test_fractional_count_targets_are_paired_per_image(self):
        # Counts 4 and 7 map to .2 and .8; .3 overestimates, .6 underestimates.
        count = torch.tensor([[0.3], [0.6]], requires_grad=True)
        branch = dict(
            reg_value=count, min_num_select=3, max_num_select=8,
            defe_feature=torch.zeros(2, 1, 1, 1), gt_density_map=torch.zeros(2, 1, 1, 1),
        )
        batch_targets = [dict(labels=torch.zeros(n, dtype=torch.long)) for n in (4, 7)]
        loss = criterion().loss_defe(branch, batch_targets)["defe_reg_loss"]
        self.assertAlmostEqual(loss.item(), (0.1**2 + 2 * 0.2**2) / 2, places=6)
        loss.backward()
        torch.testing.assert_close(count.grad, torch.tensor([[0.1], [-0.4]]))

    def test_count_targets_clip_at_both_query_bounds_under_autocast(self):
        count = torch.tensor([[0.25], [0.75]], dtype=torch.bfloat16, requires_grad=True)
        branch = dict(
            reg_value=count, min_num_select=3, max_num_select=8,
            defe_feature=torch.zeros(2, 1, 1, 1), gt_density_map=torch.zeros(2, 1, 1, 1),
        )
        batch_targets = [dict(labels=torch.zeros(n, dtype=torch.long)) for n in (0, 12)]
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            losses = criterion().loss_defe(branch, batch_targets)
        self.assertEqual(losses["defe_reg_loss"].dtype, torch.float32)
        self.assertAlmostEqual(losses["defe_reg_loss"].item(), (0.25**2 + 2 * 0.25**2) / 2)
        sum(losses.values()).backward()
        self.assertTrue(torch.isfinite(count.grad).all())

    def test_invalid_count_range_or_batch_is_rejected(self):
        branch = dict(reg_value=torch.zeros(2, 1), min_num_select=3, max_num_select=3)
        with self.assertRaisesRegex(ValueError, "max_num_select"):
            criterion().loss_defe(branch, targets())
        branch["max_num_select"] = 8
        with self.assertRaisesRegex(ValueError, "one count"):
            criterion().loss_defe(branch, targets())

    def test_legacy_list_outputs_keep_detection_losses(self):
        loss_fn = criterion(use_defe_loss=False)
        output = detection_outputs()
        expected = loss_fn(output, targets())
        actual = loss_fn([output, output], targets() * 2)
        self.assertEqual(expected.keys(), actual.keys())
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key])


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
