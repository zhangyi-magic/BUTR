"""CPU regression tests for budgeting, configuration isolation, and detector outputs."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from src.core import YAMLConfig, yaml_utils
from src.misc.qnum_stats import qnum_report, qnum_reset, qnum_update
from src.nn.backbone.hgnetv2 import load_pretrained_weights
from src.zoo.models.budget import compute_query_budget
from src.zoo.models.butr_decoder import BUTRTransformer
from src.zoo.models.dome_criterion import DomeCriterion
from src.zoo.models.matcher import HungarianMatcher
from src.zoo.models.postprocessor import DomePostProcessor
from train import build_parser


def small_decoder(**overrides):
    settings = dict(
        num_classes=3,
        hidden_dim=32,
        feat_channels=(32, 32, 32, 32),
        feat_strides=(4, 8, 16, 32),
        num_levels=4,
        num_points=[2, 2, 2, 2],
        nhead=8,
        num_layers=2,
        dim_feedforward=64,
        reg_max=8,
        min_num_select=3,
        max_num_select=8,
        num_denoising=4,
    )
    settings.update(overrides)
    return BUTRTransformer(**settings)


class BudgetTests(unittest.TestCase):
    def test_paper_formula_mask_and_no_grad(self):
        probabilities = torch.tensor([[[0.9, 0.1], [0.3, 0.8], [0.99, 0.99]]])
        logits = torch.logit(probabilities).requires_grad_()
        mask = torch.tensor([[[True], [True], [False]]])
        budget = compute_query_budget(logits, mask, 1, 20, safety=2)
        self.assertEqual(budget.tolist(), [4])
        self.assertFalse(budget.requires_grad)

    def test_clipping_and_monotonicity(self):
        logits = torch.tensor([-15.0, 0.0, 15.0]).view(3, 1, 1).expand(3, 8, 2)
        mask = torch.ones(1, 8, 1, dtype=torch.bool)
        result = compute_query_budget(logits, mask, 3, 6)
        self.assertEqual(result.tolist(), [3, 4, 6])

    def test_scoring_ablations_and_bins(self):
        logits = torch.zeros(1, 4, 2)
        mask = torch.ones(1, 4, 1, dtype=torch.bool)
        for method, expected in [
            ("max", 1),
            ("max_sigmoid", 2),
            ("sum_prob", 4),
            ("union_prob", 3),
        ]:
            with self.subTest(method=method):
                result = compute_query_budget(logits, mask, 1, 10, score_method=method)
                self.assertEqual(result.item(), expected)
        self.assertEqual(
            compute_query_budget(logits, mask, 1, 10, budget_map="bins", bins=(1, 3, 7)).item(), 3
        )

    def test_invalid_decoder_settings(self):
        for overrides in [
            dict(min_num_select=9),
            dict(budget_train_safety=0),
            dict(budget_map="bins", budget_bins=(5, 3)),
            dict(learn_query_content=True, num_queries=4),
        ]:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                small_decoder(**overrides)

    def test_training_and_eval_use_dynamic_budgets(self):
        decoder = small_decoder(budget_train_safety=1.0, budget_eval_safety=2.0)
        mask = torch.ones(1, 8, 1, dtype=torch.bool)
        for training in (True, False):
            decoder.train(training)
            safety = decoder.budget_train_safety if training else decoder.budget_eval_safety
            budgets = decoder._compute_budget_expected_count(
                torch.zeros(1, 8, 3), mask, K_base=3, K_max=8, safety=safety
            )
            self.assertEqual(budgets.tolist(), [4 if training else 8])


class ConfigTests(unittest.TestCase):
    def test_independent_loads_do_not_leak_values(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.yml"
            second = Path(directory) / "second.yml"
            first.write_text("first_only: 42\n", encoding="utf-8")
            second.write_text("second_only: 7\n", encoding="utf-8")
            self.assertEqual(yaml_utils.load_config(first), {"first_only": 42})
            self.assertEqual(yaml_utils.load_config(second), {"second_only": 7})

    def test_every_model_configuration_loads_independently(self):
        for path in Path("configs/models").rglob("*.yml"):
            if "include" in path.parts:
                continue
            with self.subTest(config=path.name):
                config = YAMLConfig(str(path))
                decoder = config.yaml_cfg["BUTR"]["decoder"]
                self.assertEqual(decoder == "BUTRTransformer", path.name.startswith("BUTR-"))

    def test_cli_preserves_amp_yaml_default(self):
        args = build_parser().parse_args(["-c", "configs/models/BUTR-L-AITOD.yml"])
        self.assertIsNone(args.use_amp)
        self.assertTrue(YAMLConfig(args.config).yaml_cfg["use_amp"])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            build_parser().parse_args(["-c", args.config, "-r", "a.pth", "-t", "b.pth"])


class DecoderTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.features = [torch.randn(2, 32, size, size) for size in (8, 4, 2, 1)]

    def test_eval_uses_batch_max_and_does_not_collect_global_stats(self):
        decoder = small_decoder(eval_spatial_size=[32, 32]).eval()
        self.assertEqual(decoder.anchors.device.type, "cpu")
        qnum_reset()
        with torch.no_grad():
            output = decoder({"feats": self.features})
        self.assertEqual(output["pred_boxes"].shape[:2], (2, output["batch_queries_num"][0]))
        self.assertEqual(len(set(output["batch_queries_num"])), 1)
        self.assertTrue(torch.isfinite(output["pred_boxes"]).all())
        self.assertEqual(qnum_report()["count"], 0)

    def test_training_loss_and_backward_with_denoising(self):
        decoder = small_decoder().train()
        targets = [
            dict(labels=torch.tensor([1]), boxes=torch.tensor([[0.5, 0.5, 0.1, 0.1]]))
            for _ in range(2)
        ]
        output = decoder({"feats": self.features}, targets)
        matcher = HungarianMatcher(
            dict(cost_class=2, cost_bbox=5, cost_giou=2), use_focal_loss=True
        )
        criterion = DomeCriterion(
            matcher,
            dict(loss_vfl=1, loss_bbox=5, loss_giou=2, loss_fgl=0.15, loss_ddf=1.5),
            ["vfl", "boxes", "local"],
            num_classes=3,
            reg_max=8,
        )
        losses = criterion(output, targets, epoch=0)
        loss = sum(losses.values())
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(decoder.enc_score_head.weight.grad)
        self.assertTrue(torch.isfinite(decoder.enc_score_head.weight.grad).all())
        self.assertIn("dn_outputs", output)

    def test_training_with_empty_targets(self):
        decoder = small_decoder().train()
        targets = [
            dict(labels=torch.empty(0, dtype=torch.long), boxes=torch.empty(0, 4)) for _ in range(2)
        ]
        output = decoder({"feats": self.features}, targets)
        self.assertTrue(torch.isfinite(output["pred_boxes"]).all())
        self.assertEqual(output["pred_logits"].shape[1], output["batch_queries_num"][0])

    def test_tiny_features_and_learned_content(self):
        decoder = small_decoder(num_queries=8, learn_query_content=True).eval()
        features = [torch.randn(1, 32, 1, 1) for _ in range(4)]
        with torch.no_grad():
            output = decoder({"feats": features})
        self.assertLessEqual(output["pred_logits"].shape[1], 4)


class PostprocessorTests(unittest.TestCase):
    def test_non_square_image_scaling_and_no_mutation(self):
        post = DomePostProcessor(num_classes=2)
        output = {
            "pred_logits": torch.tensor([[[8.0, -8.0]]]),
            "pred_boxes": torch.tensor([[[0.5, 0.5, 0.2, 0.4]]]),
        }
        result = post(output, torch.tensor([[200, 100]]))[0]
        torch.testing.assert_close(result["boxes"], torch.tensor([[80.0, 30.0, 120.0, 70.0]]))
        self.assertEqual(post.num_top_queries, 1500)
        labels, boxes, scores = post.deploy()([output], torch.tensor([[200, 100]]))
        torch.testing.assert_close(boxes[0], result["boxes"])

    def test_non_focal_boxes_are_in_image_coordinates(self):
        post = DomePostProcessor(num_classes=2, use_focal_loss=False)
        result = post(
            {
                "pred_logits": torch.tensor([[[8.0, -8.0, -8.0]]]),
                "pred_boxes": torch.tensor([[[0.5, 0.5, 0.2, 0.4]]]),
            },
            torch.tensor([[200, 100]]),
        )[0]
        torch.testing.assert_close(result["boxes"], torch.tensor([[80.0, 30.0, 120.0, 70.0]]))

    def test_query_statistics(self):
        qnum_reset()
        qnum_update([3, 5])
        result = qnum_report()
        self.assertEqual(result["mean"], 4)
        self.assertEqual(result["std"], 1)
        self.assertEqual(result["count"], 2)


class BackboneInitializationTests(unittest.TestCase):
    def test_cached_weights_are_loaded_from_a_file(self):
        with tempfile.TemporaryDirectory() as directory:
            torch.save(
                {"example": torch.tensor([3.0])}, Path(directory) / "PPHGNetV2_B0_stage1.pth"
            )
            state = load_pretrained_weights(directory, "B0", "https://example.invalid/model")
            self.assertEqual(state["example"].item(), 3.0)

    def test_single_process_download_does_not_require_a_process_group(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "torch.hub.load_state_dict_from_url", return_value={"example": 3}
            ) as download:
                state = load_pretrained_weights(directory, "B0", "https://example.invalid/model")
            self.assertEqual(state, {"example": 3})
            download.assert_called_once()


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
