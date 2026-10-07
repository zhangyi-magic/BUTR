# AI-TOD-v2 and VisDrone data preparation

Train on trainval and evaluate on test using the annotations and category IDs specified by each experiment. Grain is a supplementary experiment with a separate [cropping protocol](grain.md).

## Downloads

| Dataset | Download source | Preparation |
| --- | --- | --- |
| AI-TOD-v2 | [AI-TOD images](https://github.com/jwwangchn/AI-TOD#download) and [AI-TOD-v2 annotations](https://chasel-tsui.github.io/AI-TOD-v2/) | Use the shared AI-TOD images with the revised v2 annotations. |
| VisDrone | [Official VisDrone-DET downloads](https://github.com/VisDrone/VisDrone-Dataset#task-1-object-detection-in-images) | Download train, val, and annotated test-dev; combine train and val for training. |

Prepare COCO-format annotations and edit the paths in [aitod_detection.yml](../configs/dataset/aitod_detection.yml) or [visdrone_detection.yml](../configs/dataset/visdrone_detection.yml). Default paths are relative to the repository root:

```text
data/
├── aitod/
│   ├── images/{trainval,test}/
│   └── annotations/{aitodv2_trainval,aitodv2_test}.json
└── visdrone/
    ├── VisDrone2019-DET-trainval/images/
    ├── VisDrone2019-DET-test/images/
    ├── VisDrone2019-DET_trainval_coco.json
    └── VisDrone2019-DET_test_coco_moderate.json
```

## Annotation protocol

Use the same converted annotations as the experiments. Changing ignored regions, categories, or evaluation settings changes the protocol. The [VisDrone conversion utility](../tools/dataset/visdrone2coco.py) accepts `--root` and `--output-dir` and produces separate split annotations; the trainval merge and the paper's test annotations must be prepared separately.

AI-TOD-v2 uses `AitodCocoEvaluator`; VisDrone uses `CocoEvaluatorSlow`. The supplied heads use `num_classes: 9` and `12`, respectively. Preserve these values when loading existing checkpoints, and configure `num_classes` and `remap_mscoco_category` explicitly for a new dataset.

Paths can also be overridden on the command line:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py -c configs/models/BUTR-L-AITOD.yml --seed 0 -u \
  train_dataloader.dataset.img_folder=/path/to/images/trainval \
  train_dataloader.dataset.ann_file=/path/to/aitodv2_trainval.json \
  val_dataloader.dataset.img_folder=/path/to/images/test \
  val_dataloader.dataset.ann_file=/path/to/aitodv2_test.json
```
