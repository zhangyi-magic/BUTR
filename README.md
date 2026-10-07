# [NeurIPS 2026] Dynamic Query as Budgeting for Tiny Object Detection

Official implementation of **Dynamic Query as Budgeting for Tiny Object Detection**, accepted at **NeurIPS 2026**.

**Yi Zhang, Di Xiong, Ge Gao, Xiangyue Zhang, Yihang Qiu, Yuxuan Zhou, and Shuo Chen**  
REMI Group, PCALab@NJU, Nanjing University

[Repository](https://github.com/zhangyi-magic/BUTR)

![BUTR architecture and query initialization](static/model.png)

*Query initialization in two-stage DETRs. BUTR estimates an input-dependent query budget from encoder confidences and retains standard Top-K(x) initialization.*

## Updates

- **[2026.10]** Repository created; code refactoring underway.
- **[2026.09]** Our paper has been accepted at **NeurIPS 2026**!

## Overview

Dynamic queries primarily allocate decoder capacity according to image density. BUTR makes this budgeting explicit: a lightweight, non-learned budgeter estimates a query count from encoder confidences, followed by standard Top-K query initialization. The budgeter adds no learned parameters, auxiliary supervision, or additional loss.

The budgeter sums each valid proposal's maximum sigmoid class confidence into a density proxy, scales it by a conservative safety factor, and clips it to the feasible query range. The decoder then selects the highest-ranked proposals using this budget. See [the budgeter implementation](src/zoo/models/budget.py) and [method notes](docs/method.md) for details and ablation options.

## Main results

The AI-TOD-v2 and VisDrone results below are from Tables 3 and 4 of the paper, using 800 × 800 inputs.

| Dataset | Model | AP | Parameters | GFLOPs | Config |
| --- | --- | ---: | ---: | ---: | --- |
| AI-TOD-v2 | BUTR-S | 32.6 | 11.6M | ~112.5 | [S](configs/models/BUTR-S-AITOD.yml) |
| AI-TOD-v2 | BUTR-M | 34.6 | 22.3M | ~244.8 | [M](configs/models/BUTR-M-AITOD.yml) |
| AI-TOD-v2 | BUTR-L | 35.7 | 31.9M | ~326.1 | [L](configs/models/BUTR-L-AITOD.yml) |
| VisDrone | BUTR-S | 26.7 | 11.6M | ~114.9 | [S](configs/models/BUTR-S-VisDrone.yml) |
| VisDrone | BUTR-M | 29.4 | 22.3M | ~245.8 | [M](configs/models/BUTR-M-VisDrone.yml) |
| VisDrone | BUTR-L | 29.8 | 34.5M | ~329.5 | [L](configs/models/BUTR-L-VisDrone.yml) |

GFLOPs are approximate (`~`): the query budget is input-dependent, so computational cost varies with the number of queries used.

The supplied main configurations use trainval for training and test for evaluation. AI-TOD-v2 uses 160 epochs and a query range of `[300, 1500]`; VisDrone uses 120 epochs and `[250, 500]`. Preserve the individual configurations' training and evaluation safety factors when reproducing results.

### COCO: inference-time plug-in

On COCO, BUTR is applied to **official D-FINE and DEIM checkpoints without retraining**. Each baseline and its BUTR variant share the same weights, architecture, preprocessing, and post-processing; the budgeter changes the number of proposals selected for decoder initialization. Evaluation uses all **5,000 COCO val2017 images**, **640 × 640 inputs, FP32, and batch size 1 per GPU**:

| Model | AP baseline | AP + BUTR | Parameters | GFLOPs baseline | GFLOPs + BUTR | Mean queries + BUTR | Safety |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D-FINE-S | 48.5 | 48.6 | 10.2M | 25.2 | ~24.6 | 232.47 | 3.0 |
| D-FINE-M | 52.3 | 52.4 | 19.3M | 56.7 | ~56.6 | 288.82 | 4.5 |
| D-FINE-L | 54.0 | 54.1 | 30.8M | 91.0 | ~90.2 | 253.33 | 3.5 |
| D-FINE-X | 55.8 | 55.9 | 61.6M | 202.5 | ~201.7 | 252.90 | 4.5 |
| DEIM-S | 49.0 | 49.0 | 10.2M | 25.2 | ~24.6 | 238.57 | 6.5 |
| DEIM-M | 52.7 | 52.8 | 19.3M | 56.7 | ~55.9 | 236.36 | 8.5 |
| DEIM-L | 54.7 | 54.9 | 30.8M | 91.1 | ~89.8 | 223.56 | 8.0 |
| DEIM-X | 56.5 | 56.6 | 61.7M | 202.7 | ~200.6 | 183.48 | 6.0 |

All baselines use **300 queries**. BUTR uses query bounds `[100, 500]`, `min_prob=1e-6`, and the scale-specific safety factors above. GFLOPs are measured with `calflops 0.3.2` on the deployed model and exclude data loading and post-processing. Baseline GFLOPs use a fixed query count; BUTR GFLOPs are averages weighted by the measured query-count histogram across all evaluation images, marked with `~`. See [COCO evaluation notes](docs/coco.md) for results at higher precision, profiling details, and paired evaluation commands.

## Installation

The training environment uses **Ubuntu 22.04, Python 3.10.14, PyTorch 2.4.0, torchvision 0.19.0, and the CUDA 11.8 PyTorch build**, on eight RTX 4090 GPUs. Use this software profile for reproduction:

```bash
conda create -n BUTR python=3.10.14 --override-channels -c conda-forge -y
conda activate BUTR
conda install "pytorch::pytorch=2.4.0=*cuda11.8*" "pytorch::torchvision=0.19.0=*cu118*" pytorch-cuda=11.8 \
  numpy=2.2.6 pyyaml=6.0.3 --override-channels -c pytorch -c nvidia -c conda-forge -y
python -m pip install -r requirements.txt
python -m pip check
```

`requirements.txt` pins the project's direct training, evaluation, visualization, and profiling dependencies. Install PyTorch with the command above before installing these dependencies. The PyTorch build uses CUDA 11.8; the NVIDIA driver must support that build.

`faster-coco-eval-aitod==1.0.2` contains a compiled extension; building from source requires a C++ compiler. See [environment notes and installation checks](docs/environment.md) for package versions, build prerequisites, and commands to verify your installation.

The main BUTR configurations use the PyTorch deformable-attention implementation and do not require compiling the legacy extension in `src/zoo/models/ops/`. That extension is relevant to the optional deformable encoder.

## Data preparation

The main benchmarks are **AI-TOD-v2** and **VisDrone**. Download the data from the official sources below:

| Dataset | Download source | Preparation |
| --- | --- | --- |
| AI-TOD-v2 | [AI-TOD images](https://github.com/jwwangchn/AI-TOD#download) and [AI-TOD-v2 annotations](https://chasel-tsui.github.io/AI-TOD-v2/) | Use the shared AI-TOD images with the revised v2 annotations; train on trainval and evaluate on test. |
| VisDrone | [Official VisDrone-DET downloads](https://github.com/VisDrone/VisDrone-Dataset#task-1-object-detection-in-images) | Download train, val, and the annotated test-dev split; combine train and val for training. |

Prepare images and COCO-format annotation files, then edit the dataset YAML paths or override them on the command line. The defaults are relative to the repository root:

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

| Dataset | Dataset config | Evaluator |
| --- | --- | --- |
| AI-TOD-v2 | [aitod_detection.yml](configs/dataset/aitod_detection.yml) | `AitodCocoEvaluator` |
| VisDrone | [visdrone_detection.yml](configs/dataset/visdrone_detection.yml) | `CocoEvaluatorSlow`, as supplied |

Use the same converted annotations and category IDs as the experiments; changing ignored regions, categories, or evaluation settings changes the protocol. The included VisDrone conversion utility accepts `--root` and `--output-dir`; it produces separate split annotations, so preparing the trainval merge and the paper's test annotations remains a separate data preparation step.

The supplied AI-TOD-v2 and VisDrone heads have `num_classes: 9` and `12`, respectively. These are the existing experiment settings and should be kept when loading their checkpoints. Configure this value and `remap_mscoco_category` explicitly for a new dataset.

Grain is used only for the [supplementary cross-domain experiment](#supplementary-experiment-grain); its download and cropping protocol are documented separately below.

HGNetV2 ImageNet initialization is loaded from `weight/hgnetv2/PPHGNetV2_<name>_stage1.pth`, or downloaded by the backbone when absent. Evaluation and inference disable backbone initialization when loading a complete detector checkpoint. See [weight/README.md](weight/README.md).

## Training

Run commands from the repository root. The configurations use a total training batch size of 16 and validation batch size of 8; total batch sizes must be divisible by the distributed world size.

```bash
# Single GPU
CUDA_VISIBLE_DEVICES=0 python train.py -c configs/models/BUTR-L-AITOD.yml --seed 0

# Eight GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \
  --master_port=7789 train.py -c configs/models/BUTR-L-AITOD.yml --seed 0

# Four GPUs and a custom output directory
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
  --master_port=7789 train.py -c configs/models/BUTR-L-VisDrone.yml --seed 0 \
  --output-dir output/butr_l_visdrone_seed0
```

Set `CUDA_VISIBLE_DEVICES` to the GPUs you want to use and match `--nproc_per_node` to the number of listed GPUs.

Change configuration values with dotted YAML overrides:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py -c configs/models/BUTR-L-AITOD.yml --seed 0 -u \
  train_dataloader.dataset.img_folder=/path/to/images/trainval \
  train_dataloader.dataset.ann_file=/path/to/aitodv2_trainval.json \
  val_dataloader.dataset.img_folder=/path/to/images/test \
  val_dataloader.dataset.ann_file=/path/to/aitodv2_test.json
```

`--resume` restores training state; `--tuning` initializes from a checkpoint and supports adapting the detection head. They are mutually exclusive. AMP follows the YAML setting unless explicitly enabled with `--use-amp` or overridden with `-u use_amp=False`.

Outputs include `log.txt`, TensorBoard summaries when configured, `last.pth`, periodic checkpoints, and the two-stage best checkpoints `best_stg1.pth` / `best_stg2.pth`. The existing second-stage EMA refresh and checkpoint reload schedule is retained. Add `-u profile_model=True` to enable FLOPs profiling.

## Evaluation

```bash
# Single GPU
CUDA_VISIBLE_DEVICES=0 python train.py -c configs/models/BUTR-L-AITOD.yml --test-only \
  -r output/butr_l_aitodv2/best_stg2.pth

# Eight GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \
  --master_port=7775 train.py -c configs/models/BUTR-L-VisDrone.yml --test-only \
  -r output/butr_l_visdrone/best_stg2.pth

# Query counts without batch inflation: a single process, batch size 1
CUDA_VISIBLE_DEVICES=0 python train.py -c configs/models/BUTR-L-AITOD.yml --test-only \
  -r output/butr_l_aitodv2/best_stg2.pth -u val_dataloader.total_batch_size=1
```

Evaluation prints the dataset metrics and decoded-query mean, variance, and standard deviation. Counts are reduced across all evaluation workers. The tensorized BUTR decoder uses the maximum per-image budget within each batch; batch size 1 reports query usage without batch inflation.

## Tips

- **Dome-DETR reproduction.** We made several optimizations to the Dome-DETR implementation and reduced the base anchor size from `0.05` to `0.0125` for tiny objects. These changes may contribute to our reproduced results exceeding those reported in the original paper; keep the implementation and anchor settings in mind when comparing results.
- **Multi-GPU evaluation.** In our experiments with the [official Dome-DETR code](https://github.com/RicePasteM/Dome-DETR) as of September 2026, evaluation appeared to include only the data assigned to rank 0, rather than the full evaluation set. On smaller datasets such as VisDrone, this can produce overly optimistic validation AP. This repository aggregates evaluation results across all ranks. When reproducing results with the upstream code, verify that every evaluation image is included and cross-check with single-GPU evaluation.

## Image and video inference

```bash
CUDA_VISIBLE_DEVICES=0 python tools/inference/torch_inf.py -c configs/models/BUTR-L-AITOD.yml \
  -r output/butr_l_aitodv2/best_stg2.pth -i /path/to/image.png \
  --device cuda:0 --threshold 0.4 --output output/detections.png
```

Use `--annotation /path/to/annotations.json` to overlay ground-truth boxes in green for images. Annotations are used only for drawing. Input resolution follows `eval_spatial_size` in the configuration.

For video input:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/inference/torch_inf.py -c configs/models/BUTR-L-AITOD.yml \
  -r output/butr_l_aitodv2/best_stg2.pth -i /path/to/video.mp4 \
  --device cuda:0 --threshold 0.4 --output output/detections.mp4
```

Video inference applies the detector independently to each frame and saves an annotated video at the original resolution and frame rate, without audio. It does not perform cross-frame tracking; the output frame rate does not indicate inference speed.

## Latency measurement

Run the benchmark as a single process on one CUDA device:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/benchmark/latency.py -c configs/models/BUTR-L-AITOD.yml \
  -r output/butr_l_aitodv2/best_stg2.pth --device cuda:0 \
  --warmup-iters 50 --output output/latency.json
```

The benchmark forces validation batch size 1, synchronizes CUDA before and after each measured forward pass, and includes post-processing by default. Data loading and image transfer are excluded. `--forward-only` excludes post-processing; `--measure-iters` limits the number of measured batches. Keep hardware, input size, precision, and post-processing settings consistent when comparing models.

Legacy ONNX / TensorRT / OpenVINO tools are retained from the upstream projects. They have not been validated for BUTR's input-dependent query count; standard tracing can freeze the budget through the Python integer conversion. Use the PyTorch path for dynamic-budget experiments.

## COCO plug-in evaluation

The [COCO evaluation entry point](tools/coco/evaluate.py) uses the D-FINE implementation included in the official [DEIM](https://github.com/Intellindust-AI-Lab/DEIM) repository for **both D-FINE and DEIM checkpoints**, and strictly loads their weights. It calls the shared BUTR budgeter while retaining the upstream decoder's standard Top-K selection. Use the matching COCO configuration and checkpoint; the tiny-object configurations in this repository use different feature levels and anchor settings.

**Use the existing BUTR environment from Installation. Only the DEIM source checkout is needed for both model families.** Its declared dependencies are covered by BUTR's `requirements.txt`. The COCO model YAMLs are inside that checkout: [dfine_hgnetv2_l_coco.yml](https://github.com/Intellindust-AI-Lab/DEIM/blob/main/configs/deim_dfine/dfine_hgnetv2_l_coco.yml) for D-FINE-L and [deim_hgnetv2_l_coco.yml](https://github.com/Intellindust-AI-Lab/DEIM/blob/main/configs/deim_dfine/deim_hgnetv2_l_coco.yml) for DEIM-L. Clone DEIM (replace `/path/to/DEIM` with your chosen location):

```bash
git clone https://github.com/Intellindust-AI-Lab/DEIM.git /path/to/DEIM
```

`--upstream-root` always points to the DEIM checkout; `--family` identifies the checkpoint family. `-c` accepts an absolute path to its model YAML, or a path relative to `--upstream-root`. BUTR's local [coco_budget.yml](configs/inference/coco_budget.yml) supplies only the budgeter settings.

Run a paired comparison on one GPU, with batch size 1:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/coco/evaluate.py --family dfine \
  --upstream-root /path/to/DEIM \
  -c /path/to/DEIM/configs/deim_dfine/dfine_hgnetv2_l_coco.yml \
  -r /path/to/dfine_l_coco.pth --images /path/to/COCO/val2017 \
  --annotations /path/to/COCO/annotations/instances_val2017.json \
  --budget-config configs/inference/coco_budget.yml --safety 3.5 \
  --device cuda:0 --output output/coco/dfine_l_paired.json
```

The query range is `[100, 500]`. To reproduce the COCO results above, set `--safety` to **3.0 / 4.5 / 3.5 / 4.5 for D-FINE S/M/L/X**, or **6.5 / 8.5 / 8.0 / 6.0 for DEIM S/M/L/X**, respectively. See [COCO evaluation notes](docs/coco.md) for commands, downloads, checkpoint selection, and average-GFLOPs profiling.

## Supplementary experiment: Grain

**Sorghum-Grain-Count (Grain)** is a supplementary evaluation on dense agricultural imagery (Appendix G). Download the original images and annotations through the [official dataset repository](https://github.com/swarajksu/sorghum-grain-count).

This experiment requires preprocessing before training:

1. Combine the original train and validation splits into trainval; keep the original test split for evaluation.
2. Crop the original 3456 × 5184 images into overlapping **800 × 800** patches with a **stride of 640**.
3. Retain boxes whose centers lie inside each patch, clip them to the patch boundaries, and keep only nonempty patches.

Generate the cropped dataset from the original LabelMe splits with:

```bash
python tools/dataset/crop_grain.py --root /path/to/SGC --output data/grain \
  --size 800 --stride 640
```

The cropped dataset contains 1,859 trainval patches with 178,238 annotations and 433 test patches with 45,341 annotations. Overlapping crops can include the same grain more than once; evaluation is performed on nonempty patches.

The Grain model configurations are grouped under [configs/models/grain/](configs/models/grain/). Use [BUTR-L-GRAIN.yml](configs/models/grain/BUTR-L-GRAIN.yml) after preparing the cropped dataset. See [Grain preparation notes](docs/grain.md) for the input/output layouts, crop-boundary rules, and statistics command.

## Code organization

```text
configs/                   Dataset, runtime, BUTR, and baseline configurations
configs/models/grain/      Supplementary Grain experiment configurations
configs/inference/         COCO plug-in budget settings
src/zoo/models/budget.py    Confidence-based budget estimator and mappings
src/zoo/models/butr.py      Backbone / encoder / decoder wrapper
src/zoo/models/butr_decoder.py
                           Tensorized Top-K dynamic-query decoder
src/solver/                Training, checkpoint handling, and evaluation
src/misc/benchmark.py       Explicit CUDA latency measurement
tools/inference/            Image and video prediction
tools/coco/                 Official COCO checkpoint evaluation with budgeting
tools/diagnostics/          Optional proposal plotting helpers
tests/                     CPU regression tests
```

`Dome*` class names and the `DOME` alias remain for compatibility with baseline configurations and existing checkpoints. Training and evaluation do not import visualization tools or automatically generate diagnostic PDFs. Standalone analysis scripts remain under `tools/` and `reference/`.

Besides BUTR S/M/L, the repository includes D-FINE, DEIM, and Dome-DETR comparison configurations for AI-TOD-v2 and VisDrone. Separate Grain configurations support the supplementary experiment described above. The DEIM comparison configurations use SiLU, Mosaic/MixUp, and DEIM's flat-cosine learning-rate schedule, with Varifocal Loss replacing MAL following the paper's tiny-object comparison protocol. See [baseline training protocols](docs/baselines.md) for dataset-specific schedules and adaptations. COCO uses the upstream configurations through `tools/coco/evaluate.py` and the scale-specific safety factors listed above. Not every diagnostic intervention is packaged as a dedicated reproduction configuration.

## Checks

```bash
python -m unittest discover -s tests -v
python -m pip install ruff
python -m ruff check src tools train.py tests
```

Tests cover the budget formula and ablations, configuration isolation, dynamic decoder inference, denoising losses and backward propagation, original-image box scaling, and query statistics. Full AP and CUDA latency reproduction require datasets, trained checkpoints, and the appropriate evaluation environment.

## Citation

```bibtex
@inproceedings{zhang2026butr,
  title     = {Dynamic Query as Budgeting for Tiny Object Detection},
  author    = {Zhang, Yi and Xiong, Di and Gao, Ge and Zhang, Xiangyue and
               Qiu, Yihang and Zhou, Yuxuan and Chen, Shuo},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## Acknowledgments and license

This implementation builds on **[Dome-DETR](https://github.com/RicePasteM/Dome-DETR)**, **[D-FINE](https://github.com/Peterande/D-FINE)**, and PyTorch / torchvision detection utilities. Original copyright notices are preserved in the relevant source files. See [LICENSE](LICENSE) for the repository's Apache License 2.0.
