# Supplementary Grain experiment

Sorghum-Grain-Count (Grain) is used for the supplementary cross-domain evaluation in Appendix G. The main BUTR benchmarks are AI-TOD-v2 and VisDrone.

## Results

The following detection results are reported in Appendix G, Table 11, using 800 × 800 nonempty crops. All compared models are trained on cropped trainval and evaluated on cropped test under matched settings.

| Model | Queries (mean ± std) | AP | AP50 | AP75 | APt | APs | APm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D-FINE-L | 400 | 57.0 | 92.6 | 64.9 | 10.6 | 50.1 | 63.6 |
| D-FINE-L | 1100 | 57.7 | 93.9 | 65.6 | 13.4 | 50.9 | 64.2 |
| DEIM-L | 400 | 57.2 | 92.9 | 65.1 | 10.9 | 50.3 | 63.8 |
| DEIM-L | 1100 | 58.0 | 94.0 | 66.3 | 14.3 | 51.3 | 64.3 |
| Dome-DETR-L | 1103 ± 195 | 57.6 | 94.0 | 65.4 | 13.1 | 50.8 | 64.1 |
| BUTR-L | 378 ± 119 | 58.0 | 93.8 | 66.1 | 14.5 | 51.5 | 64.1 |

D-FINE and DEIM use fixed budgets of 400 or 1100 queries. Dome-DETR and BUTR use dynamic budgets; the mean and standard deviation are computed across evaluation patches, not repeated training runs. APt/APs/APm denote the paper's tiny/small/medium object metrics.

## Download

The [official Sorghum-Grain-Count repository](https://github.com/swarajksu/sorghum-grain-count) provides the dataset download link. Use the annotated panicle images and their original train, validation, and test splits.

## Cropping protocol

The supplied [crop_grain.py](../tools/dataset/crop_grain.py) reads images and LabelMe JSON files with matching filename stems from this original dataset layout:

```text
SGC/
├── train_img_json/
│   ├── train_img/
│   └── train_json/
├── val_img_json/
│   ├── val_img/
│   └── val_json/
└── test_img_json/
    ├── test_img/
    └── test_json/
```

1. Combine the original train and validation splits into trainval. Keep the original test split separate throughout preprocessing.
2. Crop the original 3456 × 5184 images into overlapping 800 × 800 patches with a stride of 640.
3. Retain boxes whose centers lie inside each patch, then clip them to the patch boundaries. Keep only patches with at least one retained grain, and export COCO annotations with category ID 1 (`grain`).

The script applies EXIF orientation correction before cropping and checks the corrected image dimensions against the annotations. Invalid or out-of-image boxes are discarded or clipped to the source bounds; unmatched image/annotation files are reported and skipped. All valid labeled shapes are treated as the single `grain` category, following the existing conversion utility.

Run from the repository root:

```bash
python tools/dataset/crop_grain.py --root /path/to/SGC --output data/grain \
  --size 800 --stride 640
```

Use an empty output directory. The script writes JPEG patches, COCO annotations, and `cropping_statistics.json` containing preprocessing settings and split counts.

### Boundary handling

The default cropping protocol uses the following boundary rules:

- By default, windows advance by 640 pixels, and the final window is aligned with the image edge when needed. The last gap can therefore be smaller than 640. `--edge-policy pad` instead uses the next regular stride position and pads the area beyond the image with black. Source images smaller than the crop size are also padded.
- A box is retained when its center satisfies `x <= center_x < x + 800` and `y <= center_y < y + 800`. It is then clipped and translated into patch coordinates. A grain can appear in multiple overlapping patches. Coordinates and unrounded clipped-box areas are exported to three decimal places.
- Patches are retained only if at least one box remains after these rules.

The defaults produce the Table 10 patch and annotation counts on the original dataset. `--box-policy intersection` instead retains any positive-area intersection; `--min-visibility` adds an area-fraction threshold after the box policy. These optional settings change the experimental protocol and can change the counts.

The cropped dataset in Table 10 has the following statistics:

| Split | Retained patches | Annotations | Objects per patch (mean ± std) |
| --- | ---: | ---: | ---: |
| Trainval | 1,859 | 178,238 | 95.88 ± 89.32 |
| Test | 433 | 45,341 | 104.71 ± 94.31 |

Annotation counts include instances repeated across overlapping patches. This protocol evaluates detection on nonempty crops; it does not evaluate empty-background images or full-image grain counts after merging overlapping predictions.

## Prepared dataset layout

The paths in [grain_detection.yml](../configs/dataset/grain_detection.yml) expect:

```text
data/grain/
├── trainval/
│   ├── images/
│   └── annotations/instances_trainval.json
└── test/
    ├── images/
    └── annotations/instances_test.json
```

Set the dataset paths in the YAML or use command-line overrides if you store the data elsewhere. The supplied configuration uses one class and `AitodCocoEvaluator`.

Summarize the prepared patches with:

```bash
python tools/summarize_sgc_patches.py --root data/grain
```

This writes dataset statistics under `data/grain/dataset_statistics/`. Compare the patch and annotation counts with Table 10 when checking preprocessing.

The historical [make_sgc_panicles_coco_1824.py](../tools/make_sgc_panicles_coco_1824.py) resizes whole images to 1824 × 1824 by default. Changing its `--size` only changes the whole-image resize. Use `tools/dataset/crop_grain.py` to prepare the overlapping patches for this experiment.

## Training and evaluation

The supplementary model configurations are grouped in [configs/models/grain/](../configs/models/grain/). Use the prepared patches with [BUTR-L-GRAIN.yml](../configs/models/grain/BUTR-L-GRAIN.yml):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \
  --master_port=7789 train.py -c configs/models/grain/BUTR-L-GRAIN.yml --seed 0

CUDA_VISIBLE_DEVICES=0 python train.py -c configs/models/grain/BUTR-L-GRAIN.yml --test-only \
  -r output/butr_l_grain/best_stg2.pth
```

The configuration uses 160 epochs, 800 × 800 inputs, a query range of `[300, 1200]`, and safety factors of 1.0 for training and evaluation. D-FINE-L, DEIM-L, and Dome-DETR-L Grain comparison configurations are also provided in `configs/models/grain/`.
