# Baseline training protocols

The AI-TOD-v2, VisDrone, and supplementary Grain configurations adapt the detector baselines to the datasets used in the paper. COCO plug-in evaluation instead loads official pretrained models without retraining; see [COCO evaluation](coco.md).

All detector configurations use standard L1 and GIoU box losses.

## Backbone normalization

The AI-TOD-v2, VisDrone, and supplementary Grain configurations use:

| Models | `freeze_at` | `freeze_norm` | Backbone training |
| --- | ---: | --- | --- |
| D-FINE / BUTR S and M | -1 | False | Ordinary BN; no frozen backbone stages |
| D-FINE L; BUTR L on Grain | 0 | True | FrozenBatchNorm; frozen stem (`freeze_stem_only: True`) |
| BUTR L on AI-TOD-v2 / VisDrone | -1 | False | Ordinary BN; no frozen backbone stages |
| DEIM S, M, and L | -1 | False | Ordinary BN; no frozen backbone stages |
| Dome-DETR S, M, and L | -1 | False | Ordinary BN; no frozen backbone stages |

D-FINE, DEIM, BUTR S/M, and BUTR-L on Grain match the official [D-FINE-S](https://github.com/Peterande/D-FINE/blob/master/configs/dfine/dfine_hgnetv2_s_coco.yml), [D-FINE-M](https://github.com/Peterande/D-FINE/blob/master/configs/dfine/dfine_hgnetv2_m_coco.yml), [D-FINE-L](https://github.com/Peterande/D-FINE/blob/master/configs/dfine/dfine_hgnetv2_l_coco.yml), and [DEIM](https://github.com/Intellindust-AI-Lab/DEIM/blob/main/configs/base/deim.yml) backbone settings. The dataset-specific adaptations described below remain in place; this does not make the entire recipe identical to upstream COCO training. Dome-DETR retains ordinary BN at every scale, matching its own official [AI-TOD-v2 L configuration](https://github.com/RicePasteM/Dome-DETR/blob/master/configs/dome/Dome-L-AITOD.yml) and [VisDrone L configuration](https://github.com/RicePasteM/Dome-DETR/blob/master/configs/dome/Dome-L-VisDrone.yml), which explicitly differ from D-FINE-L.

BUTR-L on AI-TOD-v2 uses ordinary BN, no frozen stem, `use_amp: False`, and MultiStepLR milestones `[80, 120]` with gamma `0.8`. Augmentation and multi-scale training stop at epoch 100. BUTR-L on VisDrone also uses ordinary BN with no frozen stem, matching Dome-DETR. Historical checkpoints retain their original training provenance; the current configurations are not an exact archived version of those runs.

Earlier D-FINE-L and BUTR-L configurations also used `freeze_at: -1` and `freeze_norm: False`. Existing checkpoints and benchmark measurements retain their original training and evaluation provenance. When evaluating an ordinary-BN checkpoint with a configuration that now defaults to FrozenBN, use `-u HGNetv2.freeze_at=-1 HGNetv2.freeze_norm=False`. Start a separate training run when changing the freezing recipe rather than resuming an old optimizer state, because the trainable parameter groups change.

Changing the normalization type only for inference does not reproduce training with frozen normalization. Checkpoint affine weights and running statistics must be retained when comparing inference variants. Ordinary BN in evaluation mode and FrozenBN use the same stored-statistics formula, but their numerical operation order and runtime kernels differ; check AP as well as latency.

## Dome-DETR official configurations

The six `Dome-{S,M,L}-{AITOD,VisDrone}.yml` configurations follow the published
[Dome-DETR configuration snapshot](https://github.com/RicePasteM/Dome-DETR/tree/2dde3bc1946a3e9fad9abd0612b59fc39bd6b861/configs/dome).
They retain the shared tiny-object settings: total training batch size 16,
validation batch size 8, and anchor size `0.0125`. `use_amp: False` matches the upstream launch command without
`--use-amp`; pass that flag explicitly to enable AMP.

The configurations use ordinary backbone BN and
MWAS window size 10 measured in feature-map cells. MWAS encodes each image's
windows separately and retains gradients through the selected features.
The decoder uses upstream PAQI selection, batch padding, and DN masks.
All local detector families use `DomeCriterion`. Dome-DETR enables
`use_defe_loss: True` to compute DeFE density and count-regression losses over
the whole batch; BUTR, D-FINE, and DEIM keep it disabled. Count targets are
floating-point values normalized to the configured query range and matched
per image. Padded queries are masked out of classification losses.
The official per-scale epoch, learning-rate, and augmentation schedules are
retained, including the S-scale VisDrone configuration's 160-epoch schedule.
The supplementary Grain recipe remains a dataset adaptation.

The implementation follows upstream commit
`2dde3bc1946a3e9fad9abd0612b59fc39bd6b861`. It retains the project's AMP dtype
compatibility and complete distributed evaluation, omits upstream diagnostic
image copies/visualization, and uses device-aware autocast for CPU compatibility.
The count-regression loss corrects the upstream integer truncation and batch
broadcasting of normalized targets. This changes that supervision relative to
the upstream implementation. The shared startup profiler also preserves RNG
state rather than consuming training RNG.

Existing paper checkpoints and deployment measurements used earlier training settings.
These configuration changes do not
retroactively change those results. Use the corresponding original settings when
evaluating those checkpoints; train a new run for the official recipe.

## DEIM adaptations

The local DEIM configurations use the D-FINE decoder with **SiLU** for both `activation` and `mlp_act`, **Mosaic and MixUp** for dense one-to-one matching, unfrozen backbone normalization, and DEIM's iteration-based **flat-cosine learning-rate schedule**. The [official DEIM configuration](https://github.com/Intellindust-AI-Lab/DEIM/blob/main/configs/base/deim.yml) defines these components together with Matchability-Aware Loss (MAL).

For the tiny-object comparisons in this paper, **Varifocal Loss (VFL) replaces MAL** because MAL was unstable in this setting. The shared `DomeCriterion` selects `['vfl', 'boxes', 'local']`; its registered name is retained for checkpoint and configuration compatibility. The configurations also use four feature levels, a base anchor size of `0.0125`, dataset-specific query counts, and adapted optimization settings. These are the paper's comparison configurations.

## Learning-rate and augmentation schedules

Every local DEIM configuration sets `lrsheduler: flatcosine` and `lr_gamma: 0.5`. The training solver selects [FlatCosineLRScheduler](../src/optim/flat_cosine.py), and the training loop updates it after each iteration. This takes precedence over the epoch-based scheduler and separate linear warmup inherited from the shared optimizer YAML.

The schedule has four phases: quadratic warmup for `warmup_iter` iterations, a constant learning rate until `flat_epoch`, cosine decay, and a final `no_aug_epoch` phase at half the initial learning rate. Epoch boundaries are converted to iterations using the training loader's length on each worker.

| Dataset | Models | Total epochs | Warmup iterations | Flat epoch | Final no-augmentation epochs | Mosaic/MixUp epochs | Augmentation stop epoch |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| AI-TOD-v2 | S / M / L | 160 | 5,000 | 93 | 10 | `[13, 93)` | 150 |
| VisDrone | S / M / L | 120 | 4,000 | 70 | 10 | `[10, 70)` | 110 |
| Grain (supplementary) | L | 58 | 2,000 | 29 | 8 | `[4, 29)` | 50 |

Mosaic and MixUp each use probability `0.5` within the listed epoch interval. All models use `BatchImageCollateFunction`; DEIM enables MixUp with `mixup_prob: 0.5`, while BUTR, D-FINE, and Dome-DETR set it to `0.0`. MixUp consumes no random numbers when its probability is zero or the epoch is outside its active interval. The augmentation stop epoch equals `total_epochs - no_aug_epoch`; it also disables multi-scale resizing. The transform policy contains three boundaries: the start of dense augmentation, its end at `flat_epoch`, and the final augmentation stop epoch. Other scheduled photometric/crop augmentations remain active between the dense phase and the final stop epoch.

These values are explicit experiment settings. Changing `epoches` alone does not rescale them. When defining a new schedule, update `warmup_iter`, `flat_epoch`, `no_aug_epoch`, `train_dataloader.dataset.transforms.policy.epoch`, `train_dataloader.collate_fn.mixup_epochs`, and `train_dataloader.collate_fn.stop_epoch` together. `warmup_iter` is measured in iterations, so its duration in epochs also depends on dataset size and batch size.

The formula in the official configuration's comment is an example rather than an automatic scheduling rule: the [official DEIM-L configuration](https://github.com/Intellindust-AI-Lab/DEIM/blob/main/configs/deim_dfine/deim_hgnetv2_l_coco.yml) uses 58 total epochs, `flat_epoch: 29`, and eight final no-augmentation epochs. Use the actual configuration values when reproducing an experiment.

## Configurations

- AI-TOD-v2: [DEIM-S](../configs/models/DEIM-S-AITOD.yml), [DEIM-M](../configs/models/DEIM-M-AITOD.yml), [DEIM-L](../configs/models/DEIM-L-AITOD.yml).
- VisDrone: [DEIM-S](../configs/models/DEIM-S-VisDrone.yml), [DEIM-M](../configs/models/DEIM-M-VisDrone.yml), [DEIM-L](../configs/models/DEIM-L-VisDrone.yml).
- Supplementary Grain: [DEIM-L](../configs/models/grain/DEIM-L-GRAIN.yml).

Use the [training and evaluation commands](../README.md#training) with the corresponding YAML. The official COCO checkpoints retain their upstream training recipe; the inference-time budget plug-in does not change their weights or loss function.
