# Pretrained weights

HGNetV2 ImageNet initialization is expected under `weight/hgnetv2/`:

```text
PPHGNetV2_B0_stage1.pth   # BUTR-S
PPHGNetV2_B2_stage1.pth   # BUTR-M
PPHGNetV2_B4_stage1.pth   # BUTR-L
```

The backbone can download these files when initialization is enabled. A full detector checkpoint supplied through `--resume` or `--tuning` disables this initialization.

This checkout contains no trained BUTR detector weights or public download manifest. Training writes detector checkpoints to the configuration's `output_dir`; use those paths for evaluation and inference.
