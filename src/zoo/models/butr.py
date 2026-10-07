"""
Dome-DETR: Dome-DETR: DETR with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object Detection
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import torch.nn as nn
from ...core import register

__all__ = [
    "BUTR",
    "DOME",
]


@register()
class BUTR(nn.Module):
    __inject__ = [
        "backbone",
        "encoder",
        "decoder",
    ]

    def __init__(
            self,
            backbone: nn.Module,
            encoder: nn.Module,
            decoder: nn.Module
    ):
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, targets=None, epoch=None):
        H, W = x.shape[-2:]
        # img_inputs = x.clone()
        x = self.backbone(x)
        x = self.encoder(x, targets, H, W)
        x = self.decoder(x, targets, epoch)
        return x

    def deploy(self):
        self.eval()
        for m in self.modules():
            if hasattr(m, "convert_to_deploy"):
                m.convert_to_deploy()
        return self


DOME = BUTR
register(name="DOME", force=True)(DOME)
