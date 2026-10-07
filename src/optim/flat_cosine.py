"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""

import math
from functools import partial

__all__ = [
    "flat_cosine_schedule",
    "FlatCosineLRScheduler",
]


def flat_cosine_schedule(
        total_iter,
        warmup_iter,
        flat_iter,
        no_aug_iter,
        current_iter,
        init_lr,
        min_lr,
):
    if warmup_iter > 0 and current_iter <= warmup_iter:
        return init_lr * (current_iter / float(warmup_iter)) ** 2
    elif warmup_iter < current_iter <= flat_iter:
        return init_lr
    elif current_iter >= total_iter - no_aug_iter:
        return min_lr
    else:
        cosine_decay = 0.5 * (
            1
            + math.cos(
                math.pi
                * (current_iter - flat_iter)
                / (total_iter - flat_iter - no_aug_iter)
            )
        )
        return min_lr + (init_lr - min_lr) * cosine_decay


class FlatCosineLRScheduler:
    """
    Iteration-level warm-up, flat, cosine decay, and no-augmentation LR schedule
    used by DEIM.
    """

    def __init__(
            self,
            optimizer,
            lr_gamma,
            iter_per_epoch,
            total_epochs,
            warmup_iter,
            flat_epochs,
            no_aug_epochs,
            scheduler_type="cosine",
    ):
        if scheduler_type != "cosine":
            raise ValueError(f"Unsupported scheduler_type: {scheduler_type}")

        self.base_lrs = [
            group.get("initial_lr", group["lr"])
            for group in optimizer.param_groups
        ]
        self.min_lrs = [base_lr * lr_gamma for base_lr in self.base_lrs]

        total_iter = int(iter_per_epoch * total_epochs)
        no_aug_iter = int(iter_per_epoch * no_aug_epochs)
        flat_iter = int(iter_per_epoch * flat_epochs)

        print(
            "[FlatCosineLRScheduler]",
            "base_lrs=", self.base_lrs,
            "min_lrs=", self.min_lrs,
            "total_iter=", total_iter,
            "warmup_iter=", warmup_iter,
            "flat_iter=", flat_iter,
            "no_aug_iter=", no_aug_iter,
        )

        self.lr_func = partial(
            flat_cosine_schedule,
            total_iter,
            warmup_iter,
            flat_iter,
            no_aug_iter,
        )

    def step(self, current_iter, optimizer):
        for i, group in enumerate(optimizer.param_groups):
            group["lr"] = self.lr_func(
                current_iter,
                self.base_lrs[i],
                self.min_lrs[i],
            )
        return optimizer
