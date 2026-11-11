"""
Copied from D-FINE (https://github.com/Peterande/D-FINE)
Copyright(c) 2024 The D-FINE Authors. All Rights Reserved.
"""

from typing import Any, Dict, List, Optional

import PIL
import PIL.Image
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F

from ...core import register
from .._misc import (
    BoundingBoxes,
    Image,
    Mask,
    SanitizeBoundingBoxes,
    Video,
    _boxes_keys,
    convert_to_tv_tensor,
)

torchvision.disable_beta_transforms_warning()


RandomPhotometricDistort = register()(T.RandomPhotometricDistort)
RandomZoomOut = register()(T.RandomZoomOut)
RandomHorizontalFlip = register()(T.RandomHorizontalFlip)
RandomVerticalFlip = register()(T.RandomVerticalFlip)  # ⭐ 新增：竖直翻转
Resize = register()(T.Resize)
# ToImageTensor = register()(T.ToImageTensor)
# ConvertDtype = register()(T.ConvertDtype)
# PILToTensor = register()(T.PILToTensor)
SanitizeBoundingBoxes = register(name="SanitizeBoundingBoxes")(SanitizeBoundingBoxes)
RandomCrop = register()(T.RandomCrop)
Normalize = register()(T.Normalize)


@register()
class EmptyTransform(T.Transform):
    def __init__(
        self,
    ) -> None:
        super().__init__()

    def forward(self, *inputs):
        inputs = inputs if len(inputs) > 1 else inputs[0]
        return inputs


@register()
class PadToSize(T.Pad):
    _transformed_types = (
        PIL.Image.Image,
        Image,
        Video,
        Mask,
        BoundingBoxes,
    )

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        sp = F.get_spatial_size(flat_inputs[0])
        h, w = self.size[1] - sp[0], self.size[0] - sp[1]
        self.padding = [0, 0, w, h]
        return dict(padding=self.padding)

    def __init__(self, size, fill=0, padding_mode="constant") -> None:
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        super().__init__(0, fill, padding_mode)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        fill = self._fill[type(inpt)]
        padding = params["padding"]
        return F.pad(inpt, padding=padding, fill=fill, padding_mode=self.padding_mode)  # type: ignore[arg-type]

    def __call__(self, *inputs: Any) -> Any:
        outputs = super().forward(*inputs)
        if len(outputs) > 1 and isinstance(outputs[1], dict):
            outputs[1]["padding"] = torch.tensor(self.padding)
        return outputs


@register()
class RandomIoUCrop(T.RandomIoUCrop):
    def __init__(
        self,
        min_scale: float = 0.3,
        max_scale: float = 1,
        min_aspect_ratio: float = 0.5,
        max_aspect_ratio: float = 2,
        sampler_options: Optional[List[float]] = None,
        trials: int = 40,
        p: float = 1.0,
    ):
        super().__init__(
            min_scale, max_scale, min_aspect_ratio, max_aspect_ratio, sampler_options, trials
        )
        self.p = p

    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]

        return super().forward(*inputs)


@register()
class ConvertBoxes(torch.nn.Module):
    def __init__(self, fmt="", normalize=False) -> None:
        super().__init__()
        self.fmt = fmt
        self.normalize = normalize

    def forward(self, sample):
        img, target, *rest = sample

        boxes = target.get("boxes", None)
        if not isinstance(boxes, BoundingBoxes):
            if rest:
                return (img, target, *rest)
            return img, target

        spatial_size = getattr(boxes, _boxes_keys[1])

        if self.fmt:
            in_fmt = boxes.format.value.lower()
            out = torchvision.ops.box_convert(boxes, in_fmt=in_fmt, out_fmt=self.fmt.lower())
            boxes = convert_to_tv_tensor(
                out, key="boxes", box_format=self.fmt.upper(), spatial_size=spatial_size
            )

        if self.normalize:
            scale = torch.tensor(spatial_size[::-1], device=boxes.device, dtype=boxes.dtype).tile(2)[None]
            boxes = boxes / scale

        target["boxes"] = boxes

        if rest:
            return (img, target, *rest)
        return img, target



@register()
class ConvertPILImage(torch.nn.Module):
    def __init__(self, dtype="float32", scale=True) -> None:
        super().__init__()
        self.dtype = dtype
        self.scale = scale

    def forward(self, sample):
        # 兼容 (img, target) 或 (img, target, dataset/extra)
        img, target, *rest = sample

        if isinstance(img, PIL.Image.Image):
            img = F.pil_to_tensor(img)
            if self.dtype == "float32":
                img = img.float()
            if self.scale:
                img = img / 255.0
            img = Image(img)

        if rest:
            return (img, target, *rest)
        return img, target




@register()
class RandomRotate90(T.Transform):
    """
    随机按 90° 的整数倍旋转（0, 90, 180, 270）。
    - p: 触发概率
    使用 torchvision.transforms.v2.functional.rotate，
    自动支持 Image / Video / Mask / BoundingBoxes 等类型。
    """

    _transformed_types = (
        PIL.Image.Image,
        Image,
        Video,
        Mask,
        BoundingBoxes,
    )

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        # 只在一次 forward 中采样一个角度，应用到该 batch 的所有相关输入上
        if torch.rand(1) >= self.p:
            return {"do_rotate": False, "angle": 0}

        # {0, 90, 180, 270}
        k = int(torch.randint(0, 4, (1,)).item())
        angle = k * 90
        return {"do_rotate": True, "angle": angle}

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if not params["do_rotate"] or params["angle"] % 360 == 0:
            return inpt

        angle = params["angle"]
        # 对不同类型，F.rotate 会自动处理（包括 tv_tensors.BoundingBoxes）
        # 默认 expand=False，保持同样的图像尺寸
        return F.rotate(inpt, angle)
