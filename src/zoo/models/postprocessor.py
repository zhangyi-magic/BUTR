"""
Copied from D-FINE (https://github.com/Peterande/D-FINE)
Copyright(c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from ...core import register

__all__ = ["DomePostProcessor"]


def mod(a, b):
    out = a - a // b * b
    return out


@register()
class DomePostProcessor(nn.Module):
    __share__ = ["num_classes", "use_focal_loss", "num_top_queries", "remap_mscoco_category"]

    def __init__(
        self,
        num_classes=80,
        use_focal_loss=True,
        num_top_queries=1500,
        remap_mscoco_category=False,
        score_thresh=0.1,
    ) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = int(num_classes)
        self.remap_mscoco_category = remap_mscoco_category
        self.deploy_mode = False
        self.score_thresh = score_thresh

    def extra_repr(self) -> str:
        return (
            f"use_focal_loss={self.use_focal_loss}, "
            f"num_classes={self.num_classes}, "
            f"num_top_queries={self.num_top_queries}"
        )

    def _forward_impl(self, logits: torch.Tensor, boxes: torch.Tensor,
                      orig_target_sizes: torch.Tensor):
        """
        logits: [B, Q, C]
        boxes:  [B, Q, 4] (cxcywh, normalized)
        orig_target_sizes: [B, 2] (width, height), matching CocoDetection.
        """
        # Preserve the dynamic-query protocol: retain one result per decoded query.
        num_top_queries = logits.shape[1]

        # cxcywh -> xyxy，并映射回原图尺度
        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")
        bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)  # [B, 1, 4] broadcast

        if self.use_focal_loss:
            scores = torch.sigmoid(logits)
            scores, index = torch.topk(scores.flatten(1), num_top_queries, dim=-1)
            # labels = index % self.num_classes
            labels = mod(index, self.num_classes)
            index = index // self.num_classes
            boxes = bbox_pred.gather(
                dim=1,
                index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]),
            )
        else:
            scores = F.softmax(logits, dim=-1)[:, :, :-1]
            scores, labels = scores.max(dim=-1)
            boxes = bbox_pred
            if scores.shape[1] > num_top_queries:
                scores, index = torch.topk(scores, num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=index)
                boxes = torch.gather(
                    boxes,
                    dim=1,
                    index=index.unsqueeze(-1).repeat(1, 1, boxes.shape[-1]),
                )

        if self.deploy_mode:
            return labels, boxes, scores

        if self.remap_mscoco_category:
            from ...data.dataset import mscoco_label2category
            labels = (
                torch.tensor(
                    [mscoco_label2category[int(x.item())] for x in labels.flatten()]
                )
                .to(boxes.device)
                .reshape(labels.shape)
            )

        results = []
        for lab, box, sco in zip(labels, boxes, scores):
            result = dict(labels=lab, boxes=box, scores=sco)
            results.append(result)

        return results

    def forward(self, outputs, orig_target_sizes: torch.Tensor):
        """
        outputs:
          - 旧版: dict, 包含 'pred_logits': [B, Q, C], 'pred_boxes': [B, Q, 4]
          - 现在: list[dict]，每个元素:
                {'pred_logits': [1, Q_i, C], 'pred_boxes': [1, Q_i, 4]}
        orig_target_sizes: [B, 2]
        """
        # 兼容旧结构：单个 dict，内部已经是 batched tensor
        if isinstance(outputs, dict):
            logits = outputs["pred_logits"]   # [B, Q, C]
            boxes = outputs["pred_boxes"]     # [B, Q, 4]
            return self._forward_impl(logits, boxes, orig_target_sizes)

        # 新结构：list of dict，每个样本单独处理
        assert isinstance(outputs, (list, tuple)), \
            f"Unexpected outputs type: {type(outputs)}"

        results = []
        if self.deploy_mode:
            if len(outputs) != 1:
                raise ValueError("Deploy inference for variable-length outputs requires batch size 1.")
            return self._forward_impl(outputs[0]["pred_logits"], outputs[0]["pred_boxes"], orig_target_sizes)
        for i, out in enumerate(outputs):
            logits = out["pred_logits"]   # [1, Q_i, C]
            boxes = out["pred_boxes"]     # [1, Q_i, 4]

            # 对应这张图的原始 size，形状 [1, 2]
            size_i = orig_target_sizes[i:i+1]

            # _forward_impl 返回的是一个 list，长度 = B（这里 B=1）
            res_i = self._forward_impl(logits, boxes, size_i)[0]
            results.append(res_i)

        return results

    def deploy(self):
        self.eval()
        self.deploy_mode = True
        return self
