# Evaluation checkpoints

`butr_s_aitodv2.pth` contains the EMA weights from the AI-TOD-v2 BUTR-S `best_stg2.pth` checkpoint. The original training log reports 32.6535 AP (32.7 rounded to one decimal).

The file uses the official D-FINE/DEIM conversion format: `{"model": checkpoint["ema"]["module"]}`. Model tensors are unchanged; optimizer and scheduler states are omitted. Use it for evaluation, inference, or weight initialization. Resuming the original training requires the full training checkpoint.
