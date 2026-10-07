# Baseline training protocols

The AI-TOD-v2, VisDrone, and supplementary Grain configurations adapt the detector baselines to the datasets used in the paper. COCO plug-in evaluation instead loads official pretrained models without retraining; see [COCO evaluation](coco.md).

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

Mosaic and MixUp each use probability `0.5` within the listed epoch interval. The augmentation stop epoch equals `total_epochs - no_aug_epoch`; it also disables multi-scale resizing. The transform policy contains three boundaries: the start of dense augmentation, its end at `flat_epoch`, and the final augmentation stop epoch. Other scheduled photometric/crop augmentations remain active between the dense phase and the final stop epoch.

These values are explicit experiment settings. Changing `epoches` alone does not rescale them. When defining a new schedule, update `warmup_iter`, `flat_epoch`, `no_aug_epoch`, `train_dataloader.dataset.transforms.policy.epoch`, `train_dataloader.collate_fn.mixup_epochs`, and `train_dataloader.collate_fn.stop_epoch` together. `warmup_iter` is measured in iterations, so its duration in epochs also depends on dataset size and batch size.

The formula in the official configuration's comment is an example rather than an automatic scheduling rule: the [official DEIM-L configuration](https://github.com/Intellindust-AI-Lab/DEIM/blob/main/configs/deim_dfine/deim_hgnetv2_l_coco.yml) uses 58 total epochs, `flat_epoch: 29`, and eight final no-augmentation epochs. Use the actual configuration values when reproducing an experiment.

## Configurations

- AI-TOD-v2: [DEIM-S](../configs/models/DEIM-S-AITOD.yml), [DEIM-M](../configs/models/DEIM-M-AITOD.yml), [DEIM-L](../configs/models/DEIM-L-AITOD.yml).
- VisDrone: [DEIM-S](../configs/models/DEIM-S-VisDrone.yml), [DEIM-M](../configs/models/DEIM-M-VisDrone.yml), [DEIM-L](../configs/models/DEIM-L-VisDrone.yml).
- Supplementary Grain: [DEIM-L](../configs/models/grain/DEIM-L-GRAIN.yml).

Use the [training and evaluation commands](../README.md#training) with the corresponding YAML. The official COCO checkpoints retain their upstream training recipe; the inference-time budget plug-in does not change their weights or loss function.
