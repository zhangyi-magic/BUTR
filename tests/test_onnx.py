"""Optional ONNX Runtime regression: one exported graph must retain changing budgets."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from src.zoo.models.butr_decoder import BUTRTransformer
from src.zoo.models.postprocessor import DomePostProcessor
from tools.deployment.export_onnx import check_dynamic_budget, compare_detections, export_model
from tools.inference.onnx_inf import prepare_inputs
from tools.deployment.split_tensorrt import split_model

HAS_ONNX = all(importlib.util.find_spec(name) is not None for name in ("onnx", "onnxruntime"))


class DecoderFixture(nn.Module):
    """Real decoder/postprocessor with controlled, uniquely ranked encoder scores."""

    def __init__(self, **settings):
        super().__init__()
        self.decoder = BUTRTransformer(
            num_classes=3,
            hidden_dim=32,
            feat_channels=(32,) * 4,
            feat_strides=(4, 8, 16, 32),
            num_levels=4,
            num_points=[2] * 4,
            nhead=8,
            num_layers=2,
            dim_feedforward=64,
            reg_max=8,
            min_num_select=3,
            max_num_select=8,
            num_denoising=0,
            eval_spatial_size=(32, 32),
            budget_eval_safety=1.0,
            **settings,
        ).eval()
        self.decoder.enc_output = nn.Identity()
        with torch.no_grad():
            self.decoder.enc_score_head.weight.zero_()
            self.decoder.enc_score_head.weight[:, 0] = 1
            self.decoder.enc_score_head.bias.zero_()
        self.decoder.convert_to_deploy()
        self.decoder.decoder.convert_to_deploy()
        self.post = DomePostProcessor(num_classes=3).deploy()

    def forward(self, packed_features, sizes):
        features = list(packed_features.split([64, 16, 4, 1], dim=2))
        features = [value.reshape(1, 32, side, side) for value, side in zip(features, (8, 4, 2, 1))]
        return self.post(self.decoder({"feats": features}), sizes)


@unittest.skipUnless(HAS_ONNX, "Install tools/deployment/requirements.txt for ONNX checks")
class ONNXTests(unittest.TestCase):
    def test_same_graph_changes_query_count_and_matches_pytorch(self):
        import onnxruntime as ort

        old_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            for settings in (
                {},
                {"learn_query_content": True, "num_queries": 8},
                {"budget_map": "bins", "budget_bins": (3, 6, 8)},
            ):
                with self.subTest(settings=settings), tempfile.TemporaryDirectory() as directory:
                    torch.manual_seed(0)
                    model = DecoderFixture(**settings).eval()
                    base = torch.randn(1, 32, 85)
                    sizes = torch.tensor([[200, 100]], dtype=torch.int64)

                    def inputs(level):
                        features = base.clone()
                        features[:, 0] = level + torch.arange(85) / 10000
                        return features, sizes

                    path = export_model(model, inputs(-15), Path(directory) / "model.onnx")
                    options = ort.SessionOptions()
                    options.intra_op_num_threads = 2
                    session = ort.InferenceSession(
                        str(path), options, providers=["CPUExecutionProvider"]
                    )
                    counts = []
                    for level in (-15, -3, -2.5, 0, 15):
                        features, sizes = inputs(level)
                        with torch.no_grad():
                            expected = [value.numpy() for value in model(features, sizes)]
                        actual = session.run(
                            None, {"images": features.numpy(), "orig_target_sizes": sizes.numpy()}
                        )
                        for found, wanted in zip(actual, expected):
                            np.testing.assert_allclose(found, wanted, atol=2e-4, rtol=1e-4)
                        counts.append(actual[0].shape[1])
                    self.assertEqual(
                        counts, [3, 6, 8, 8, 8] if settings.get("budget_map") else [3, 5, 7, 8, 8]
                    )
        finally:
            torch.set_num_threads(old_threads)

    def test_frozen_budget_is_rejected(self):
        import onnx

        graph = onnx.helper.make_graph(
            [
                onnx.helper.make_node(
                    "Constant",
                    [],
                    ["k"],
                    value=onnx.helper.make_tensor("k", onnx.TensorProto.INT64, [1], [3]),
                ),
                onnx.helper.make_node("TopK", ["images", "k"], ["values", "indices"]),
            ],
            "frozen",
            [onnx.helper.make_tensor_value_info("images", onnx.TensorProto.FLOAT, [1, 10])],
            [],
        )
        with self.assertRaisesRegex(ValueError, "input-dependent"):
            check_dynamic_budget(onnx.helper.make_model(graph))

    def test_split_graph_matches_same_dynamic_model(self):
        import onnx
        import onnxruntime as ort

        old_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            torch.manual_seed(0)
            model = DecoderFixture().eval()
            base = torch.randn(1, 32, 85)
            sizes = torch.tensor([[200, 100]], dtype=torch.int64)
            with tempfile.TemporaryDirectory() as directory:
                path = export_model(model, (base, sizes), Path(directory) / "full.onnx")
                infer = onnx.shape_inference.infer_shapes
                cleared = []

                def infer_with_unknown_rank(*args, **kwargs):
                    graph = infer(*args, **kwargs)
                    values = {value.name: value for value in graph.graph.value_info}
                    for node in graph.graph.node:
                        if (
                            node.op_type == "Shape"
                            and "/self_attn/" in node.name
                            and node.input[0] in values
                        ):
                            values[node.input[0]].type.tensor_type.ClearField("shape")
                            cleared.append(node.input[0])
                    return graph

                with patch(
                    "onnx.shape_inference.infer_shapes", side_effect=infer_with_unknown_rank
                ):
                    manifest = split_model(path, directory, (3, 5, 8))
                self.assertTrue(cleared, "Fixture must exercise unknown-rank attention metadata")
                options = ort.SessionOptions()
                options.intra_op_num_threads = 2
                encoder = ort.InferenceSession(str(Path(directory) / "encoder.onnx"), options)
                decoder = ort.InferenceSession(str(Path(directory) / "decoder.onnx"), options)
                counts = []
                for level in (-15, -3, -2.5, 0):
                    features = base.clone()
                    features[:, 0] = level + torch.arange(85) / 10000
                    feed = {"images": features.numpy(), "orig_target_sizes": sizes.numpy()}
                    values = encoder.run(
                        None, {name: feed[name] for name in manifest["encoder_inputs"]}
                    )
                    feed.update(dict(zip(manifest["frontier"], values)))
                    actual = decoder.run(
                        None, {name: feed[name] for name in manifest["decoder_inputs"]}
                    )
                    with torch.no_grad():
                        expected = [value.numpy() for value in model(features, sizes)]
                    for observed, reference in zip(actual, expected):
                        np.testing.assert_allclose(observed, reference, atol=2e-4, rtol=1e-4)
                    counts.append(actual[0].shape[1])
                self.assertEqual(counts, [3, 5, 7, 8])
        finally:
            torch.set_num_threads(old_threads)


class DetectionComparisonTests(unittest.TestCase):
    def test_missing_derived_buffers_can_be_recreated_but_missing_weights_cannot(self):
        from tools.deployment.export_onnx import load_deployment_state

        model = nn.Module()
        model.decoder = nn.Module()
        model.decoder.register_buffer("anchors", torch.zeros(1, 10, 4))
        model.decoder.register_buffer("valid_mask", torch.ones(1, 10, 1, dtype=torch.bool))
        model.decoder.decoder = nn.Module()
        model.decoder.decoder.register_buffer("project", torch.arange(9.0))
        model.head = nn.Linear(4, 3)
        state = model.state_dict()
        trained_weights = {
            key: value.clone() for key, value in state.items() if key.startswith("head.")
        }
        load_deployment_state(model, trained_weights)
        for key, value in trained_weights.items():
            torch.testing.assert_close(model.state_dict()[key], value)
        with self.assertRaisesRegex(RuntimeError, "head.weight"):
            load_deployment_state(model, {"head.bias": trained_weights["head.bias"]})

    def test_preprocessing_matches_validation_and_size_order(self):
        from PIL import Image
        from torchvision.transforms import Compose, Resize, ToTensor

        image = Image.fromarray(np.arange(90, dtype=np.uint8).reshape(5, 6, 3))
        actual = prepare_inputs(image, (8, 10))
        expected = Compose([Resize((8, 10)), ToTensor()])(image).unsqueeze(0).numpy()
        np.testing.assert_array_equal(actual["images"], expected)
        np.testing.assert_array_equal(actual["orig_target_sizes"], [[6, 5]])

    def test_permutation_allowed_but_changed_boxes_rejected(self):
        expected = [
            np.array([[1, 1]]),
            np.array([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]]),
            np.array([[0.5, 0.5]]),
        ]
        actual = [value[:, ::-1].copy() for value in expected]
        compare_detections(actual, expected, np.array([[100, 200]]))
        actual[1][0, 0, 0] += 1
        with self.assertRaises(AssertionError):
            compare_detections(actual, expected, np.array([[100, 200]]))
