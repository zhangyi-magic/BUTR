"""
Copied from D-FINE (https://github.com/Peterande/D-FINE)
Copyright(c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import random
from copy import deepcopy

import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
from ..core import register

torchvision.disable_beta_transforms_warning()

__all__ = [
    "DataLoader",
    "BaseCollateFunction",
    "BatchImageCollateFunction",
    "batch_image_collate_fn",
]


@register()
class DataLoader(data.DataLoader):
    __inject__ = ["dataset", "collate_fn"]

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for n in ["dataset", "batch_size", "num_workers", "drop_last", "collate_fn"]:
            format_string += "\n"
            format_string += "    {0}: {1}".format(n, getattr(self, n))
        format_string += "\n)"
        return format_string

    def set_epoch(self, epoch):
        self._epoch = epoch
        self.dataset.set_epoch(epoch)
        self.collate_fn.set_epoch(epoch)

    @property
    def epoch(self):
        return self._epoch if hasattr(self, "_epoch") else -1

    @property
    def shuffle(self):
        return self._shuffle

    @shuffle.setter
    def shuffle(self, shuffle):
        assert isinstance(shuffle, bool), "shuffle must be a boolean"
        self._shuffle = shuffle


@register()
def batch_image_collate_fn(items):
    """only batch image"""
    return torch.cat([x[0][None] for x in items], dim=0), [x[1] for x in items]


class BaseCollateFunction(object):
    def set_epoch(self, epoch):
        self._epoch = epoch

    @property
    def epoch(self):
        return self._epoch if hasattr(self, "_epoch") else -1

    def __call__(self, items):
        raise NotImplementedError("")


def generate_scales(base_size, base_size_repeat, window_size):
    """修改后的generate_scales函数确保处理后尺寸满足(size/4)%window_size==0"""
    if isinstance(base_size, (list, tuple)):
        h, w = base_size
        step = 4 * window_size
        # 调整h和w为step的倍数
        h = (h // step) * step or step
        w = (w // step) * step or step

        # 生成h的scales
        h_min = int(h * 0.75)
        start_low_h = ((h_min + step - 1) // step) * step
        low_to_base_h = list(range(start_low_h, h + 1, step))
        h_max = int(h * 1.25)
        end_high_h = (h_max // step) * step
        high_part_h = list(range(end_high_h, h - 1, -step)) if end_high_h >= h + step else []
        h_scales = low_to_base_h + [h] * base_size_repeat + high_part_h

        # 生成w的scales
        w_min = int(w * 0.75)
        start_low_w = ((w_min + step - 1) // step) * step
        low_to_base_w = list(range(start_low_w, w + 1, step))
        w_max = int(w * 1.25)
        end_high_w = (w_max // step) * step
        high_part_w = list(range(end_high_w, w - 1, -step)) if end_high_w >= w + step else []
        w_scales = low_to_base_w + [w] * base_size_repeat + high_part_w

        # 生成笛卡尔积组合
        scales = []
        for h_scale in h_scales:
            for w_scale in w_scales:
                scales.append((h_scale, w_scale))
        return scales
    else:
        # 处理方形尺寸
        step = 4 * window_size
        base_size = (base_size // step) * step or step
        base_min = int(base_size * 0.75)
        start_low = ((base_min + step - 1) // step) * step
        low_to_base = list(range(start_low, base_size + 1, step))
        base_max = int(base_size * 1.25)
        end_high = (base_max // step) * step
        high_part = list(range(end_high, base_size - 1, -step)) if end_high >= base_size + step else []
        scales = low_to_base + [base_size] * base_size_repeat + high_part
        return [(s, s) for s in scales]


@register()
class BatchImageCollateFunction(BaseCollateFunction):
    """Optional batch MixUp followed by multi-scale resizing for all detectors."""

    def __init__(
            self,
            stop_epoch=None,
            ema_restart_decay=0.9999,
            base_size=(640, 640),
            base_size_repeat=None,
            mwas_window_size=20,  # 新增window_size参数
            mixup_prob=0.0,
            mixup_epochs=(0, 0),
    ) -> None:
        super().__init__()
        self.base_size = base_size
        if isinstance(self.base_size, str):
            try:
                self.base_size = int(self.base_size)
            except ValueError:
                try:
                    self.base_size = self.base_size.strip('()[]')
                    self.base_size = tuple(int(x.strip()) for x in self.base_size.split(','))
                except:
                    raise ValueError(f"Cannot convert base_size string '{self.base_size}' to valid size format")
        self.window_size = mwas_window_size  # 初始化window_size
        self.scales = (
            generate_scales(self.base_size, base_size_repeat, self.window_size)
            if base_size_repeat is not None
            else None
        )
        self.stop_epoch = stop_epoch if stop_epoch is not None else 100000000
        self.ema_restart_decay = ema_restart_decay
        self.mixup_prob = mixup_prob
        self.mixup_epochs = mixup_epochs

        if mixup_prob > 0:
            print(f"     ### Using MixUp with Prob@{mixup_prob} in {mixup_epochs} epochs ### ")

        if stop_epoch is not None:
            print("     ### Multi-scale Training until {} epochs ### ".format(self.stop_epoch))
            print("     ### Multi-scales@ {} ###        ".format(self.scales))

    def __call__(self, items):
        images = torch.cat([x[0][None] for x in items], dim=0)
        targets = [x[1] for x in items]
        images, targets = self.apply_mixup(images, targets)
        return self._resize_batch(images, targets)

    def _resize_batch(self, images, targets):
        if self.scales is not None and self.epoch < self.stop_epoch:
            sz = random.choice(self.scales)
            images = F.interpolate(images, size=sz)
            if "masks" in targets:
                for tg in targets:
                    tg["masks"] = F.interpolate(tg["masks"], size=sz, mode="nearest")

        return images, targets


    def apply_mixup(self, images, targets):
        if (
            self.mixup_prob > 0
            and self.mixup_epochs[0] <= self.epoch < self.mixup_epochs[-1]
            and random.random() < self.mixup_prob
        ):
            beta = round(random.uniform(0.45, 0.55), 6)
            images = images.roll(shifts=1, dims=0).mul_(1.0 - beta).add_(images.mul(beta))
            shifted_targets = targets[-1:] + targets[:-1]
            updated_targets = deepcopy(targets)
            for i in range(len(targets)):
                for key in ("boxes", "labels", "area"):
                    updated_targets[i][key] = torch.cat([targets[i][key], shifted_targets[i][key]], dim=0)
                updated_targets[i]["mixup"] = torch.tensor(
                    [beta] * len(targets[i]["labels"])
                    + [1.0 - beta] * len(shifted_targets[i]["labels"]),
                    dtype=torch.float32,
                )
            targets = updated_targets
        return images, targets
