<h1 align="center">Dynamic Query as Budgeting for Tiny Object Detection</h1>

<p align="center">Official implementation of <a href="">BUTR</a> (NeurIPS 2026), a remarkably simple approach to robust dynamic query budgeting for real-time tiny (and dense) object detection.</p>

<p align="center"><a href="https://zhangyi-magic.github.io/">Yi Zhang</a><sup>1</sup>, Di Xiong<sup>1</sup>, Ge Gao<sup>1</sup>, Xiangyue Zhang<sup>1</sup>, Yihang Qiu<sup>1</sup>, Yuxuan Zhou<sup>1</sup>, and <a href="https://shuochenya.github.io/">Shuo Chen</a><sup>1,2,*</sup><br>
<sup>1</sup> <em>School of Intelligence Science and Technology, Nanjing University, China.</em><br>
<sup>2</sup> <em>Center for Advanced Intelligence Project, RIKEN, Japan.</em><br>
* 📧 Corresponding author: <a href="mailto:shuo.chen@nju.edu.cn">shuo.chen@nju.edu.cn</a></p>

<p align="center">This repository also supports training and evaluation of <a href="https://github.com/RicePasteM/Dome-DETR">Dome-DETR</a>, <a href="https://github.com/Peterande/D-FINE">D-FINE</a>, and <a href="https://github.com/Intellindust-AI-Lab/DEIM">DEIM</a> baselines for tiny object detection. See the <a href="docs/baselines.md">baseline protocols</a>.<br>
<strong>If you find BUTR useful, please consider giving this repository a star ⭐.</strong></p>

<p align="center"><img src="static/model.png" alt="BUTR architecture and query initialization" width="100%"></p>

<p align="center"><em>BUTR estimates an input-dependent query budget and retains standard Top-K query initialization.</em></p>

## Updates

- [2026.10] Repository created; code refactored; retrained checkpoints coming soon.
- [2026.09] Our paper has been accepted at NeurIPS 2026!

## Overview

BUTR treats dynamic queries as decoder budgeting: a lightweight, non-learned budgeter estimates the number of queries from encoder confidences, followed by standard Top-K initialization. It adds no learned parameters, auxiliary supervision, or additional loss. See the [method notes](docs/method.md) and [budgeter](src/zoo/models/budget.py).

## Main results

AI-TOD-v2 and VisDrone results use 800 × 800 inputs, with training on trainval and evaluation on test.

Deployment latency is measured on an NVIDIA T4 with TensorRT 10.4.0, FP16-enabled engines, and batch size 1. See [deployment benchmarks](docs/deployment-results.md) for the reproduction environment, COCO latency calibration, and comparisons with D-FINE, DEIM, Dome-DETR, and DQ-DETR.

### AI-TOD-v2

| Model | AP | #Params | Latency | GFLOPs | Config |
| --- | ---: | ---: | ---: | ---: | :---: |
| BUTR-S | 32.6 | 11.6M | 13.05&nbsp;ms | 112.5 | [yml](configs/models/BUTR-S-AITOD.yml) |
| BUTR-M | 34.6 | 22.3M | 18.78&nbsp;ms | 244.8 | [yml](configs/models/BUTR-M-AITOD.yml) |
| BUTR-L | 35.7 | 31.9M | 22.34&nbsp;ms | 326.1 | [yml](configs/models/BUTR-L-AITOD.yml) |

### VisDrone

| Model | AP | #Params | Latency | GFLOPs | Config |
| --- | ---: | ---: | ---: | ---: | :---: |
| BUTR-S | 26.7 | 11.6M | 14.10&nbsp;ms | 114.9 | [yml](configs/models/BUTR-S-VisDrone.yml) |
| BUTR-M | 29.4 | 22.3M | 20.54&nbsp;ms | 245.8 | [yml](configs/models/BUTR-M-VisDrone.yml) |
| BUTR-L | 29.8 | 34.5M | 25.27&nbsp;ms | 329.5 | [yml](configs/models/BUTR-L-VisDrone.yml) |

AI-TOD-v2 uses 160 training epochs and queries in `[300, 1500]`; VisDrone uses 120 epochs and `[250, 500]`.

GFLOPs and latency are reported as averages and depend on image-specific query budgets. GFLOPs are computational-cost estimates; latency averages three rounds on a fixed 1,000-image test subset of each dataset using the split-before-transpose deployment layout. AP and GFLOPs report the paper results; deployment accuracy checks are described in the benchmark document.

### COCO inference-time plug-in

BUTR is applied to official D-FINE and DEIM checkpoints without retraining. Each pair shares weights and preprocessing. Evaluation uses all 5,000 COCO val2017 images, 640 × 640 inputs, FP32, and batch size 1 per GPU.

| Model | AP<br>baseline | AP<br>+&nbsp;BUTR | #Params | GFLOPs<br>baseline | GFLOPs<br>+&nbsp;BUTR | Mean queries<br>+&nbsp;BUTR | Safety<br>Factor |
| --- | ---: | ---: | ---: | ---: | ---: |----------------------------:|-----------------:|
| D&#8209;FINE&#8209;S | 48.5 | 48.6 | 10.2M | 25.2 | 24.6 | 232 |              3.0 |
| D&#8209;FINE&#8209;M | 52.3 | 52.4 | 19.3M | 56.7 | 56.6 | 289 |              4.5 |
| D&#8209;FINE&#8209;L | 54.0 | 54.1 | 30.8M | 91.0 | 90.2 | 253 |              3.5 |
| D&#8209;FINE&#8209;X | 55.8 | 55.9 | 61.6M | 202.5 | 201.7 | 253 |              4.5 |
| DEIM&#8209;S | 49.0 | 49.0 | 10.2M | 25.2 | 24.6 | 239 |              6.5 |
| DEIM&#8209;M | 52.7 | 52.8 | 19.3M | 56.7 | 55.9 | 236 |              8.5 |
| DEIM&#8209;L | 54.7 | 54.9 | 30.8M | 91.1 | 89.8 | 227 |              8.0 |
| DEIM&#8209;X | 56.5 | 56.6 | 61.7M | 202.7 | 200.6 | 183 |              6.0 |

Baselines use 300 queries; BUTR uses `[100, 500]` and `min_prob=1e-6`. GFLOPs use `calflops 0.3.2` on the deployed model, excluding data loading and post-processing. Dynamic costs are averaged over the measured query-count histogram. See [COCO evaluation and profiling](docs/coco.md).

## Installation

The reference training environment is Ubuntu 22.04, Python 3.10.14, PyTorch 2.4.0, torchvision 0.19.0, and CUDA 11.8.

Before installing Python dependencies, ensure a C++17 compiler is available (`g++ --version`). On Ubuntu, install it with `sudo apt-get update && sudo apt-get install -y build-essential`. For a Conda compiler without administrator privileges, see [build prerequisites](docs/environment.md#build-prerequisites).

```bash
conda create -n BUTR python=3.10.14 --override-channels \
  -c https://conda.anaconda.org/conda-forge -y &&
conda activate BUTR &&
conda install "pytorch=2.4.0=*cuda11.8*" "torchvision=0.19.0=*cu118*" pytorch-cuda=11.8 \
  numpy=2.2.6 pyyaml=6.0.3 --override-channels \
  -c https://conda.anaconda.org/pytorch \
  -c https://conda.anaconda.org/nvidia \
  -c https://conda.anaconda.org/conda-forge -y
```

After the Conda installation succeeds, install the Python dependencies with `BUTR` active and the compiler available. If using a Conda compiler, install it in `BUTR` at this point using the instructions above.

```bash
python -m pip install -r requirements.txt &&
python -m pip check
```

If `BUTR` already exists, skip `conda create` and activate it before continuing. See [environment notes](docs/environment.md) for dependency details and GPU/import checks.

## Data preparation

| Dataset | Download | Dataset config |
| :---: | :---: | :---: |
| AI-TOD-v2 | [Images](https://github.com/jwwangchn/AI-TOD#download) · [v2 annotations](https://chasel-tsui.github.io/AI-TOD-v2/) | [aitod_detection.yml](configs/dataset/aitod_detection.yml) |
| VisDrone | [Official downloads](https://github.com/VisDrone/VisDrone-Dataset#task-1-object-detection-in-images) | [visdrone_detection.yml](configs/dataset/visdrone_detection.yml) |

Edit the image and annotation paths in the dataset YAMLs. See [data preparation](docs/data.md) for directory layouts and annotation requirements, and [weight initialization](weight/README.md) for HGNetV2 weights. Grain is a [supplementary experiment](#supplementary-experiment-grain).

## Training

Run from the repository root:

```bash
# Single GPU
CUDA_VISIBLE_DEVICES=0 python train.py -c configs/models/BUTR-L-AITOD.yml --seed 0

# Eight GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \
  --master_port=7789 train.py -c configs/models/BUTR-L-AITOD.yml --seed 0
```

Use the corresponding model YAML for VisDrone or another scale. Match `--nproc_per_node` to the number of listed GPUs. Total batch sizes must be divisible by the GPU count. See [training options](docs/usage.md#training-options) and [baseline protocols](docs/baselines.md).

## Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python train.py -c configs/models/BUTR-L-AITOD.yml --test-only \
  -r output/butr_l_aitodv2/best_stg2.pth
```

For per-image query statistics, add `-u val_dataloader.total_batch_size=1`: the decoder uses the maximum budget within each batch. See [distributed evaluation](docs/usage.md#distributed-evaluation-and-query-counts).

## Inference

```bash
CUDA_VISIBLE_DEVICES=0 python tools/inference/torch_inf.py -c configs/models/BUTR-L-AITOD.yml \
  -r output/butr_l_aitodv2/best_stg2.pth -i /path/to/image.png \
  --device cuda:0 --threshold 0.4 --output output/detections.png
```

The same tool accepts video input. See [image/video inference and latency measurement](docs/usage.md) for options.

For dynamic ONNX export, split-first TensorRT FP16 deployment, and latency measurement, see the [deployment guide](docs/tensorrt.md). The same layout rewrite is available for D-FINE/DEIM baseline comparisons.

## COCO plug-in evaluation

Use the existing BUTR environment and clone [DEIM](https://github.com/Intellindust-AI-Lab/DEIM), which contains both checkpoint families’ implementations and COCO model YAMLs:

```bash
git clone https://github.com/Intellindust-AI-Lab/DEIM.git /path/to/DEIM

CUDA_VISIBLE_DEVICES=0 python tools/coco/evaluate.py --family dfine \
  --upstream-root /path/to/DEIM \
  -c configs/deim_dfine/dfine_hgnetv2_l_coco.yml \
  -r /path/to/dfine_l_coco.pth --images /path/to/COCO/val2017 \
  --annotations /path/to/COCO/annotations/instances_val2017.json \
  --budget-config configs/inference/coco_budget.yml --safety 3.5 \
  --device cuda:0 --output output/coco/dfine_l_paired.json
```

Use the matching official checkpoint and upstream YAML, with the scale-specific safety factor in the results table. `-c` is relative to `--upstream-root`; the local tiny-object configurations are different. See [COCO instructions](docs/coco.md) for checkpoint downloads, DEIM commands, and profiling.

## Supplementary experiment: Grain

<details>
<summary>Results, data preparation and training</summary>

[Sorghum-Grain-Count](https://github.com/swarajksu/sorghum-grain-count) is used for supplementary cross-domain evaluation. The paper results below (Appendix G, Table 11) use 800 × 800 nonempty crops, with training on the cropped trainval split and evaluation on the cropped test split.

| Model | AP | Mean queries |
| --- | ---: | ---: |
| D-FINE-L | 57.0 | 400 |
| D-FINE-L | 57.7 | 1100 |
| DEIM-L | 57.2 | 400 |
| DEIM-L | 58.0 | 1100 |
| Dome-DETR-L | 57.6 | 1103 |
| BUTR-L | 58.0 | 378 |

Mean queries are averaged across test patches. D-FINE and DEIM use fixed query counts; Dome-DETR and BUTR use dynamic budgets.

Crop the original images into 800 × 800 patches with stride 640, retain boxes by their centers, and keep nonempty patches:

```bash
python tools/dataset/crop_grain.py --root /path/to/SGC --output data/grain \
  --size 800 --stride 640
```

Use the configurations under [configs/models/grain/](configs/models/grain/). See [Grain preparation and training](docs/grain.md) for the full protocol.

</details>

## Tips

- We optimized the Dome-DETR implementation and reduced the base anchor size from `0.05` to `0.0125` for tiny objects. These changes may contribute to our reproduced results exceeding the original paper’s reports.
- In our experiments with [official Dome-DETR](https://github.com/RicePasteM/Dome-DETR) as of September 2026, multi-GPU evaluation appeared to include only rank 0’s data, which can inflate validation AP on small datasets such as VisDrone. This repository aggregates all ranks; cross-check upstream evaluation with a single GPU.

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

This implementation builds on [Dome-DETR](https://github.com/RicePasteM/Dome-DETR), [D-FINE](https://github.com/Peterande/D-FINE), and [DEIM](https://github.com/Intellindust-AI-Lab/DEIM). Original copyright notices are preserved. See [LICENSE](LICENSE) for the Apache License 2.0.
