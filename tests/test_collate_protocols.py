"""Model-specific augmentation and backbone protocol regressions."""

import contextlib
import io
import random
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from src.core import YAMLConfig, create
from src.core.yaml_utils import load_config
from src.data.dataloader import BatchImageCollateFunction


def items():
    return [
        (torch.full((3, 8, 8), float(i)), {
            "boxes": torch.tensor([[0.5, 0.5, 0.1, 0.1]]),
            "labels": torch.tensor([i]),
            "area": torch.tensor([1.0]),
        })
        for i in range(2)
    ]


class CollateProtocolTests(unittest.TestCase):
    def test_plain_collate_preserves_official_multiscale_random_sequence(self):
        with contextlib.redirect_stdout(io.StringIO()):
            collate = BatchImageCollateFunction(
                base_size=80, base_size_repeat=2, mwas_window_size=2, stop_epoch=100,
                mixup_prob=0.0, mixup_epochs=(0, 100),
            )
        collate.set_epoch(0)
        saved = random.getstate()
        try:
            expected = random.Random(0)
            random.seed(0)
            for _ in range(10):
                shape = expected.choice(collate.scales)
                images, targets = collate(items())
                self.assertEqual(tuple(images.shape[-2:]), shape)
                self.assertEqual([len(t["labels"]) for t in targets], [1, 1])
            self.assertEqual(random.getstate(), expected.getstate())
        finally:
            random.setstate(saved)

    def test_deim_mixup_merges_images_targets_and_respects_epoch_window(self):
        with contextlib.redirect_stdout(io.StringIO()):
            collate = BatchImageCollateFunction(mixup_prob=1.0, mixup_epochs=(2, 5))
        original = items()
        collate.set_epoch(2)
        with patch("src.data.dataloader.random.random", return_value=0.0), patch(
            "src.data.dataloader.random.uniform", return_value=0.5,
        ):
            images, targets = collate(original)
        self.assertTrue(torch.equal(images, torch.full_like(images, 0.5)))
        self.assertEqual(targets[0]["labels"].tolist(), [0, 1])
        self.assertEqual(targets[1]["labels"].tolist(), [1, 0])
        self.assertEqual(targets[0]["mixup"].tolist(), [0.5, 0.5])
        self.assertEqual(len(original[0][1]["labels"]), 1)
        collate.set_epoch(5)
        images, targets = collate(items())
        self.assertEqual(float(images[0].max()), 0.0)
        self.assertNotIn("mixup", targets[0])

    def test_model_configs_build_the_correct_collate_without_dataset_files(self):
        for path in Path("configs/models").rglob("*.yml"):
            if "include" in path.parts:
                continue
            with self.subTest(config=path.name):
                cfg = YAMLConfig(str(path))
                settings = cfg.yaml_cfg["train_dataloader"]["collate_fn"]
                globals_cfg = cfg.global_cfg
                globals_cfg["test_collate"] = settings.copy()
                with contextlib.redirect_stdout(io.StringIO()):
                    collate = create("test_collate", globals_cfg)
                self.assertIs(type(collate), BatchImageCollateFunction)
                if path.name.startswith("DEIM-"):
                    self.assertEqual(collate.mixup_prob, 0.5)
                    self.assertEqual(list(collate.mixup_epochs), cfg.yaml_cfg["train_dataloader"]["dataset"]["transforms"]["policy"]["epoch"][:2])
                else:
                    self.assertEqual(settings["mixup_prob"], 0.0)
                    self.assertEqual(collate.mixup_prob, 0.0)

    def test_outside_mixup_window_does_not_consume_random_numbers(self):
        with contextlib.redirect_stdout(io.StringIO()):
            collate = BatchImageCollateFunction(mixup_prob=0.5, mixup_epochs=(2, 5))
        for epoch in (1, 5):
            with self.subTest(epoch=epoch):
                collate.set_epoch(epoch)
                before = random.getstate()
                with patch("src.data.dataloader.random.random") as draw:
                    images, targets = collate(items())
                draw.assert_not_called()
                self.assertEqual(random.getstate(), before)
                self.assertEqual(float(images[0].max()), 0.0)
                self.assertNotIn("mixup", targets[0])

    def test_butr_visdrone_l_backbone_matches_dome(self):
        butr = load_config("configs/models/BUTR-L-VisDrone.yml", {})
        dome = load_config("configs/models/Dome-L-VisDrone.yml", {})
        for key in ("freeze_at", "freeze_norm", "freeze_stem_only"):
            self.assertEqual(butr["HGNetv2"][key], dome["HGNetv2"][key])
        self.assertEqual(butr["HGNetv2"]["freeze_at"], -1)
        self.assertFalse(butr["HGNetv2"]["freeze_norm"])


if __name__ == "__main__":
    unittest.main()
