# COCO inference-time budget plug-in

The COCO experiment in Table 6 keeps official D-FINE/DEIM weights fixed and adds the BUTR budgeter during inference. It requires no retraining or additional learned parameters. The baseline and budgeted model use identical weights and the same evaluation pipeline.

## Official models and data

- [D-FINE model zoo](https://github.com/Peterande/D-FINE#model-zoo): use the COCO S/M/L/X configuration and its matching checkpoint.
- [DEIM model zoo](https://github.com/Intellindust-AI-Lab/DEIM#model-zoo): use the **DEIM-D-FINE** S/M/L/X models, not the RT-DETR variants. Table 6 uses the improved checkpoints (e.g. 49.0 AP for S and 54.7 AP for L).
- [COCO2017 downloads](https://cocodataset.org/#download): prepare `val2017/` images and `annotations/instances_val2017.json`.

Use the existing BUTR environment installed with the root `requirements.txt`; it covers DEIM's declared dependencies. Only the **DEIM source checkout** is needed: it contains the D-FINE implementation and COCO configurations for both checkpoint families.

```bash
git clone https://github.com/Intellindust-AI-Lab/DEIM.git /path/to/DEIM
```

Run the commands below from the BUTR root. **Both sets of COCO model YAMLs are inside the cloned DEIM repository, not BUTR's `configs/` directory.** Use [D-FINE-L's YAML](https://github.com/Intellindust-AI-Lab/DEIM/blob/main/configs/deim_dfine/dfine_hgnetv2_l_coco.yml) or [DEIM-L's YAML](https://github.com/Intellindust-AI-Lab/DEIM/blob/main/configs/deim_dfine/deim_hgnetv2_l_coco.yml). Their relative includes are loaded from DEIM as well.

The examples use absolute YAML paths to make the location explicit. `--upstream-root` always points to DEIM, while `--family dfine` or `--family deim` identifies which weights are being evaluated. A relative `-c` path is resolved relative to `--upstream-root`; checkpoint and dataset paths refer to your local files. BUTR's `configs/inference/coco_budget.yml` contains only budgeter settings. Keep the upstream inference resolution (640 × 640 for these HGNetv2 COCO configurations).

## Budget settings

[coco_budget.yml](../configs/inference/coco_budget.yml) uses query bounds `[100, 500]` and `min_prob=1e-6`. Set `--safety` separately for each model:

| Family | S | M | L | X |
| --- | ---: | ---: | ---: | ---: |
| D-FINE | 3.0 | 4.5 | 3.5 | 4.5 |
| DEIM | 6.5 | 8.5 | 8.0 | 6.0 |

The profile defaults to 6.0. Evaluation uses all 5,000 COCO val2017 images, 640 × 640 inputs, FP32, and batch size 1 per GPU.

The plug-in calls the same [compute_query_budget](../src/zoo/models/budget.py) as BUTR: sum valid proposals' maximum sigmoid class confidence, multiply by safety, take the ceiling, and clamp to the range. The original decoder then selects this many proposals using its unchanged Top-K ranking. Invalid anchors are identified from the upstream decoder's infinite anchor coordinates.

## Paired evaluation

D-FINE-L:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/coco/evaluate.py --family dfine \
  --upstream-root /path/to/DEIM \
  -c /path/to/DEIM/configs/deim_dfine/dfine_hgnetv2_l_coco.yml \
  -r /path/to/dfine_l_coco.pth --images /path/to/COCO/val2017 \
  --annotations /path/to/COCO/annotations/instances_val2017.json \
  --budget-config configs/inference/coco_budget.yml --safety 3.5 \
  --device cuda:0 --output output/coco/dfine_l_paired.json
```

DEIM-L:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/coco/evaluate.py --family deim \
  --upstream-root /path/to/DEIM \
  -c /path/to/DEIM/configs/deim_dfine/deim_hgnetv2_l_coco.yml \
  -r /path/to/deim_l_coco.pth --images /path/to/COCO/val2017 \
  --annotations /path/to/COCO/annotations/instances_val2017.json \
  --budget-config configs/inference/coco_budget.yml --safety 8.0 \
  --device cuda:0 --output output/coco/deim_l_paired.json
```

Change the scale in the official configuration path and use the corresponding checkpoint. The default `--mode paired` evaluates the original fixed-query model and the budgeted variant sequentially using the same loaded model. `--mode baseline` evaluates only the original model; `--mode budget` evaluates only the plug-in variant. `--max-images 10 --num-workers 0` can be used for a smoke check, and should be omitted when reporting full-dataset AP.

The entry point runs in a single process with batch size 1 and does not receive ground-truth targets during model inference. It strictly loads `ema.module` when present, otherwise `model` or a bare state dictionary. It retains the official postprocessor, including class-category remapping and detection Top-K. Model parameters, buffers, anchor scale, and class heads are not replaced. The tiny-object configurations in `configs/models/` are therefore not used for this protocol.

The JSON report records the shared `DEIM.engine` implementation, configuration and checkpoint paths, selected weights, input resolution, budget settings, COCO bbox metrics, and decoded-query statistics, including a query-count histogram. For older DEIM source, the entry point bridges the legacy `ConvertPILImage._transform` to the current torchvision `transform` method; it calls the same pixel conversion without editing the upstream files.

## Average GFLOPs

Use the saved evaluation histogram to profile each observed query count, then average the resulting costs over all evaluated images:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/coco/profile_queries.py --upstream-root /path/to/DEIM \
  --evaluation-report output/coco/dfine_l_paired.json \
  --checkpoint /path/to/dfine_l_coco.pth --device cuda:0 \
  --output output/coco/dfine_l_flops.json
```

This tool follows the upstream `calflops` counting path with the deployed model at the evaluation resolution. It uses fixed query counts for profiling and weights their costs by the measured histogram; it does not estimate cost by substituting the mean query count. GFLOPs are operator-count estimates, and exclude data loading and postprocessing. Older reports without a query histogram need to be regenerated. Multiple reports for the same model can be passed to `--evaluation-report` to share the profiling work.

## Results

The following table gives the README results at higher precision. Baselines use 300 queries; BUTR uses the safety factors in [Budget settings](#budget-settings). Average GFLOPs follow the histogram-weighted profiling procedure above.

| Model | Baseline AP | AP + BUTR | Baseline GFLOPs | Average GFLOPs + BUTR | Mean queries + BUTR |
| --- | ---: | ---: | ---: | ---: | ---: |
| D-FINE-S | 48.5041 | 48.5780 | 25.171 | 24.554 | 232.47 |
| D-FINE-M | 52.2768 | 52.3531 | 56.684 | 56.551 | 288.82 |
| D-FINE-L | 53.9702 | 54.0516 | 91.034 | 90.208 | 253.33 |
| D-FINE-X | 55.7644 | 55.8764 | 202.486 | 201.652 | 252.90 |
| DEIM-S | 48.9611 | 48.9944 | 25.171 | 24.610 | 238.57 |
| DEIM-M | 52.6907 | 52.7558 | 56.684 | 55.921 | 236.36 |
| DEIM-L | 54.7433 | 54.8752 | 91.144 | 89.789 | 223.56 |
| DEIM-X | 56.4709 | 56.5631 | 202.669 | 200.604 | 183.48 |
