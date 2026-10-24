# Evaluation checkpoints

All published checkpoints are hosted in [GitHub Releases](https://github.com/zhangyi-magic/BUTR/releases/tag/checkpoints). Download the desired file into this directory; checkpoint binaries are not tracked in the source tree.

| Dataset | Model | AP | Download |
| :---: | :---: | :---: | :---: |
| AI-TOD-v2 | BUTR-S | 32.7 | [pth](https://github.com/zhangyi-magic/BUTR/releases/download/checkpoints/butr_s_aitodv2.pth) |
| AI-TOD-v2 | BUTR-L | 35.7 | [pth](https://github.com/zhangyi-magic/BUTR/releases/download/checkpoints/butr_l_aitodv2.pth) |

`butr_s_aitodv2.pth` contains the EMA weights from the AI-TOD-v2 BUTR-S `best_stg2.pth` checkpoint. The original training log reports 32.6535 AP (32.7 rounded to one decimal).

The file uses the official D-FINE/DEIM conversion format: `{"model": checkpoint["ema"]["module"]}`. Model tensors are unchanged; optimizer and scheduler states are omitted. Use it for evaluation, inference, or weight initialization. Resuming the original training requires the full training checkpoint.

`butr_l_aitodv2.pth` contains the EMA weights from the original AI-TOD-v2 BUTR-L run (epoch 117). The [training log](../docs/logs/butr_l_aitodv2.log) records 35.6061 AP. Evaluating the same weights with `max` scoring and safety 2.0 gives **35.6906 AP** (35.7 rounded), with **309.07** mean queries on the full 14,018-image test set. It uses the same model-only EMA format as the S checkpoint.

Evaluate the L checkpoint with its corresponding backbone and budget settings:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \
  train.py -c configs/models/BUTR-L-AITOD.yml \
  -r checkpoints/butr_l_aitodv2.pth --test-only \
  -u HGNetv2.freeze_at=-1 HGNetv2.freeze_norm=False HGNetv2.use_lab=False \
     BUTRTransformer.budget_score_method=max BUTRTransformer.budget_eval_safety=2.0
```

The command uses the dataset paths in the configuration; override them for your local installation.
