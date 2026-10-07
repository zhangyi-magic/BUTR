"""Training and evaluation for BUTR and detector baselines.

Modified from Dome-DETR and D-FINE.
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import math
import sys
from typing import Iterable

import torch
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import SummaryWriter

from ..misc import MetricLogger, SmoothedValue, dist_utils
from ..misc.qnum_stats import qnum_reset, qnum_update, qnum_report
from ..optim import ModelEMA, Warmup


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
    **kwargs,
):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)

    print_freq = kwargs.get("print_freq", 100)
    writer: SummaryWriter = kwargs.get("writer", None)

    ema: ModelEMA = kwargs.get("ema", None)
    scaler: GradScaler = kwargs.get("scaler", None)
    lr_warmup_scheduler: Warmup = kwargs.get("lr_warmup_scheduler", None)
    self_lr_scheduler = kwargs.get("self_lr_scheduler", False)
    lr_scheduler = kwargs.get("lr_scheduler", None)

    for i, (samples, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        samples = samples.to(device, non_blocking=True)
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

        global_step = epoch * len(data_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step, epoch_step=len(data_loader))

        if scaler is not None:
            with torch.autocast(device_type=torch.device(device).type, cache_enabled=True):
                outputs = model(samples, targets, epoch)

            with torch.autocast(device_type=torch.device(device).type, enabled=False):
                loss_dict = criterion(outputs, targets, **metas)

            loss = sum(loss_dict.values())
            scaler.scale(loss).backward()

            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        else:
            outputs = model(samples, targets, epoch)
            loss_dict = criterion(outputs, targets, **metas)

            loss: torch.Tensor = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()

            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()

        # ema
        if ema is not None:
            ema.update(model)

        if self_lr_scheduler:
            optimizer = lr_scheduler.step(global_step, optimizer)
        elif lr_warmup_scheduler is not None:
            lr_warmup_scheduler.step()

        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if writer and dist_utils.is_main_process() and global_step % 10 == 0:
            writer.add_scalar("Loss/total", loss_value.item(), global_step)
            for j, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f"Lr/pg_{j}", pg["lr"], global_step)
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(f"Loss/{k}", v.item(), global_step)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessor, data_loader, coco_evaluator, device):
    """Evaluate detections and report the number of queries actually decoded."""
    model.eval()
    criterion.eval()
    coco_evaluator.cleanup()
    qnum_reset()
    metric_logger = MetricLogger(delimiter="  ")

    for samples, targets in metric_logger.log_every(data_loader, 100, "Test:"):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in target.items()} for target in targets]
        outputs = model(samples, targets=targets, epoch=0)
        sizes = torch.stack([target["orig_size"] for target in targets])
        results = postprocessor(outputs, sizes)
        if isinstance(outputs, dict):
            counts = outputs.get("batch_queries_num")
            if counts is None:
                counts = [outputs["pred_logits"].shape[1]] * len(targets)
        else:
            counts = [output["pred_logits"].shape[1] for output in outputs]
        qnum_update(counts)
        coco_evaluator.update(
            {target["image_id"].item(): result for target, result in zip(targets, results)}
        )

    qnum_report(prefix="[Val][QUERY NUM]", device=device)
    metric_logger.synchronize_between_processes()
    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    # All ranks must make the same checkpoint/EMA decisions after evaluation.
    stats = {}
    if dist_utils.is_main_process():
        stats = {
            f"coco_eval_{kind}": evaluator.stats.tolist()
            for kind, evaluator in coco_evaluator.coco_eval.items()
        }
    if dist_utils.is_dist_available_and_initialized():
        payload = [stats]
        torch.distributed.broadcast_object_list(payload, src=0)
        stats = payload[0]
    return stats, coco_evaluator
