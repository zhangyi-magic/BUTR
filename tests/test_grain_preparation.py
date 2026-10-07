"""Regression checks for split isolation, crop coordinates, and EXIF alignment."""

import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from tools.dataset.crop_grain import build_parser, crop_box, main, window_starts


class GrainPreparationTests(unittest.TestCase):
    def test_windows_cover_edges_without_duplicate_starts(self):
        self.assertEqual(window_starts(13, 8, 6), [0, 5])
        self.assertEqual(window_starts(20, 8, 6), [0, 6, 12])
        self.assertEqual(window_starts(5, 8, 6), [0])
        self.assertEqual(window_starts(13, 8, 6, "pad"), [0, 6])
        self.assertEqual(window_starts(20, 8, 6, "pad"), [0, 6, 12])
        with self.assertRaises(ValueError):
            window_starts(20, 8, 9)

    def test_crossing_boxes_and_visibility(self):
        self.assertEqual(crop_box((6, 2, 10, 6), 0, 0, 8, box_policy="intersection"), [6, 2, 2, 4])
        self.assertEqual(crop_box((6, 2, 10, 6), 5, 0, 8), [1, 2, 4, 4])
        self.assertIsNone(crop_box((6, 2, 10, 6), 0, 0, 8, 0.6, box_policy="intersection"))
        self.assertIsNone(crop_box((8, 0, 10, 2), 0, 0, 8))

    def test_center_policy_filters_fragments_and_uses_half_open_windows(self):
        self.assertIsNone(crop_box((6, 2, 10, 6), 0, 0, 8))
        self.assertEqual(crop_box((4, 2, 8, 6), 6, 0, 8), [0, 2, 2, 4])
        # A center inside both axes can retain less than half the total box area.
        self.assertEqual(crop_box((7, 7, 12, 12), 0, 0, 10), [7, 7, 3, 3])
        self.assertIsNone(crop_box((7, 7, 12, 12), 0, 0, 10, min_visibility=0.5))

    def make_source(self, root, *, mismatch=False, orientation=False):
        for split in ("train", "val", "test"):
            directory = root / f"{split}_img_json"
            images, annotations = directory / f"{split}_img", directory / f"{split}_json"
            images.mkdir(parents=True)
            annotations.mkdir()
            image = Image.new("RGB", (13, 8), (50, 100, 150))
            if orientation:
                image = image.transpose(Image.Transpose.ROTATE_90)
                exif = image.getexif()
                exif[274] = 6  # Rotate stored pixels clockwise to the annotated orientation.
                image.save(images / "same.jpg", exif=exif)
            else:
                image.save(images / "same.jpg")
            data = dict(
                imageWidth=12 if mismatch else 13,
                imageHeight=8,
                shapes=[
                    dict(label="grain", points=[[6, 2], [9, 6]]),
                    dict(label="grain", points=[[1, 1], [1, 3]]),
                ],
            )
            (annotations / "same.json").write_text(json.dumps(data), encoding="utf-8")
            # Empty source images must not produce any retained patches.
            Image.new("RGB", (13, 8)).save(images / "empty.jpg")
            (annotations / "empty.json").write_text(
                json.dumps(dict(imageWidth=13, imageHeight=8, shapes=[])), encoding="utf-8"
            )

    def args(self, root, output):
        return build_parser().parse_args(
            ["--root", str(root), "--output", str(output), "--size", "8", "--stride", "6"]
        )

    def test_full_conversion_merges_trainval_and_preserves_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root, output = Path(directory) / "raw", Path(directory) / "prepared"
            self.make_source(root)
            main(self.args(root, output))
            for split, expected_sources, count in (
                ("trainval", {"train", "val"}, 4),
                ("test", {"test"}, 2),
            ):
                coco = json.loads(
                    (output / split / "annotations" / f"instances_{split}.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(len(coco["images"]), count)
                self.assertEqual(len(coco["annotations"]), count)
                self.assertEqual(
                    {item["source_split"] for item in coco["images"]}, expected_sources
                )
                self.assertEqual(len({item["file_name"] for item in coco["images"]}), count)
                self.assertEqual(
                    {tuple(item["bbox"]) for item in coco["annotations"]},
                    {(6, 2, 2, 4), (1, 2, 3, 4)},
                )
                for image in coco["images"]:
                    with Image.open(output / split / "images" / image["file_name"]) as patch:
                        self.assertEqual(patch.size, (8, 8))
            report = json.loads((output / "cropping_statistics.json").read_text())
            self.assertEqual(report["splits"][0]["empty_patches_skipped"], 4)
            self.assertEqual(report["splits"][0]["invalid_shapes_skipped"], 2)
            with self.assertRaises(FileExistsError):
                main(self.args(root, output))

    def test_exif_is_corrected_before_dimension_checks_and_cropping(self):
        with tempfile.TemporaryDirectory() as directory:
            root, output = Path(directory) / "raw", Path(directory) / "prepared"
            self.make_source(root, orientation=True)
            main(self.args(root, output))
            coco = json.loads((output / "test/annotations/instances_test.json").read_text())
            self.assertEqual(len(coco["images"]), 2)
            self.assertEqual(coco["images"][0]["source_width"], 13)
            self.assertEqual(coco["images"][0]["source_height"], 8)

    def test_dimension_mismatch_fails_explicitly(self):
        with tempfile.TemporaryDirectory() as directory:
            root, output = Path(directory) / "raw", Path(directory) / "prepared"
            self.make_source(root, mismatch=True)
            with self.assertRaisesRegex(ValueError, "dimensions do not match"):
                main(self.args(root, output))


if __name__ == "__main__":
    unittest.main()
