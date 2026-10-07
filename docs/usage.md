# Training, inference, and profiling

Run commands from the repository root using the [installation profile](../README.md#installation). The [README](../README.md#training) provides the basic training commands; this page covers additional options.

## Training options

Set `CUDA_VISIBLE_DEVICES` to the GPUs you want to use and match `--nproc_per_node` to their number. Total training and validation batch sizes must be divisible by the distributed world size. The main configurations use total batch sizes of 16 and 8, respectively.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
  --master_port=7789 train.py -c configs/models/BUTR-L-VisDrone.yml --seed 0 \
  --output-dir output/butr_l_visdrone_seed0
```

`--resume` restores training state; `--tuning` initializes from a checkpoint and supports adapting the detection head. They are mutually exclusive. AMP follows the YAML setting unless enabled with `--use-amp` or overridden with `-u use_amp=False`. Configuration values can be changed with dotted YAML overrides; see [data preparation](data.md) for path examples.

Training outputs include `log.txt`, TensorBoard summaries when configured, `last.pth`, periodic checkpoints, and `best_stg1.pth` / `best_stg2.pth`. The second-stage EMA refresh and checkpoint reload schedule is retained. Add `-u profile_model=True` to enable model FLOPs profiling.

HGNetV2 ImageNet initialization is loaded from `weight/hgnetv2/PPHGNetV2_<name>_stage1.pth`, or downloaded when absent. Loading a complete detector checkpoint disables backbone initialization. See [weight/README.md](../weight/README.md).

The DEIM comparison configurations use SiLU, Mosaic/MixUp, and the flat-cosine learning-rate schedule, with VFL replacing MAL for the paper's tiny-object comparison protocol. See [baseline training protocols](baselines.md) for dataset-specific schedules.

## Distributed evaluation and query counts

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \
  --master_port=7775 train.py -c configs/models/BUTR-L-VisDrone.yml --test-only \
  -r output/butr_l_visdrone/best_stg2.pth
```

Evaluation aggregates metrics across workers and prints the decoded-query mean, variance, and standard deviation. The decoder uses the maximum per-image budget within a batch. To measure query usage without batch inflation, run a single process with batch size 1:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py -c configs/models/BUTR-L-AITOD.yml --test-only \
  -r output/butr_l_aitodv2/best_stg2.pth -u val_dataloader.total_batch_size=1
```

## Image and video inference

```bash
CUDA_VISIBLE_DEVICES=0 python tools/inference/torch_inf.py -c configs/models/BUTR-L-AITOD.yml \
  -r output/butr_l_aitodv2/best_stg2.pth -i /path/to/image.png \
  --device cuda:0 --threshold 0.4 --output output/detections.png
```

Use `--annotation /path/to/annotations.json` to overlay ground-truth boxes in green. Annotations are used only for drawing. Input resolution follows `eval_spatial_size` in the configuration.

For video input:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/inference/torch_inf.py -c configs/models/BUTR-L-AITOD.yml \
  -r output/butr_l_aitodv2/best_stg2.pth -i /path/to/video.mp4 \
  --device cuda:0 --threshold 0.4 --output output/detections.mp4
```

Video inference applies the detector independently to each frame and saves an annotated video at the original resolution and frame rate, without audio. It does not perform cross-frame tracking; the output frame rate does not indicate inference speed.

## CUDA latency measurement

Run the benchmark as a single process on one CUDA device:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/benchmark/latency.py -c configs/models/BUTR-L-AITOD.yml \
  -r output/butr_l_aitodv2/best_stg2.pth --device cuda:0 \
  --warmup-iters 50 --output output/latency.json
```

The benchmark forces validation batch size 1 and synchronizes CUDA before and after each measured forward pass. It includes post-processing by default and excludes data loading and image transfer. `--forward-only` excludes post-processing; `--measure-iters` limits the number of measured batches. Keep hardware, input size, precision, and post-processing settings consistent when comparing models.

Legacy ONNX / TensorRT / OpenVINO tools have not been validated for BUTR's input-dependent query count: standard tracing can freeze the budget during Python integer conversion. Use the PyTorch path for dynamic-budget experiments. The main configurations use PyTorch deformable attention; building the legacy extension is needed only for the optional deformable encoder.

## Development checks

```bash
python -m unittest discover -s tests -v
python -m pip install ruff
python -m ruff check src tools train.py tests
```

[pyproject.toml](../pyproject.toml) configures Ruff's Python target, exclusions, and lint rules. Tests cover budgeting, configuration isolation, decoder inference and backward propagation, box scaling, and query statistics.

The `Dome*` names and `DOME` alias remain for baseline and checkpoint compatibility. Optional [diagnostics](../tools/diagnostics/README.md) are separate from the training and evaluation path.
