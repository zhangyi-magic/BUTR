"""Historical loss-debugging implementation, retained for manual comparisons.

Modified from Dome-DETR and D-FINE.
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import copy

import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F

from src.misc.dist_utils import get_world_size, is_dist_available_and_initialized
from src.zoo.models.box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou
from src.zoo.models.dome_utils import bbox2distance


class DomeCriterion(nn.Module):
    """This class computes the loss for Dome-DETR."""

    __share__ = [
        "num_classes",
    ]
    __inject__ = [
        "matcher",
    ]

    def __init__(
        self,
        matcher,
        weight_dict,
        losses,
        alpha=0.2,
        gamma=2.0,
        num_classes=80,
        reg_max=32,
        boxes_weight_format=None,
        share_matched_indices=False,
        defe_density_map_weight=4,
        density_recall_penalty=0.3,
        mal_alpha=None,
        use_uni_set=True,
    ):
        """Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals.
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            num_classes: number of object categories, omitting the special no-object category.
            reg_max (int): Max number of the discrete bins in Dome-DETR.
            boxes_weight_format: format for boxes weight (iou, ).
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.boxes_weight_format = boxes_weight_format
        self.share_matched_indices = share_matched_indices
        self.alpha = alpha
        self.gamma = gamma
        self.fgl_targets, self.fgl_targets_dn = None, None
        self.own_targets, self.own_targets_dn = None, None
        self.reg_max = reg_max
        self.num_pos, self.num_neg = None, None
        self.defe_density_map_weight = defe_density_map_weight
        self.density_recall_penalty = density_recall_penalty
        self.mal_alpha = mal_alpha
        self.use_uni_set = use_uni_set

    def _process_list_output(self, outputs, key):
        """处理列表形式的输出，如果是列表则返回，否则包装成列表"""
        if key in outputs:
            if isinstance(outputs[key], list):
                return outputs[key]
            else:
                return [outputs[key]]
        return None

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, values=None):
        assert "pred_boxes" in outputs
        pred_boxes_list = self._process_list_output(outputs, "pred_boxes")
        src_logits_list = self._process_list_output(outputs, "pred_logits")
        # print("pred_boxes_list: ", len(pred_boxes_list), pred_boxes_list[0].shape, pred_boxes_list[1].shape)
        # print("src_logits_list: ", len(src_logits_list), src_logits_list[0].shape, src_logits_list[1].shape)

        total_loss = 0
        batch_size = len(src_logits_list)

        for b, anno in enumerate(targets):
            src_logits = src_logits_list[b]  # [1, num_queries, num_classes]
            src_boxes = pred_boxes_list[b]  # [1, num_queries, 4]
            if len(src_logits.shape) == 2:
                src_logits = src_logits.unsqueeze(0)
            if len(src_boxes.shape) == 2:
                src_boxes = src_boxes.unsqueeze(0)

            src_idx, tgt_idx = indices[b]
            # print("vfl meta: ", src_logits.shape, src_boxes.shape, src_idx, tgt_idx)

            # 计算IoU
            if values is None:
                src_boxes_matched = src_boxes[0, src_idx]  # [num_matched, 4]
                target_boxes_matched = anno["boxes"]  # [num_matched, 4]
                ious, _ = box_iou(
                    box_cxcywh_to_xyxy(src_boxes_matched), box_cxcywh_to_xyxy(target_boxes_matched)
                )
                ious = torch.diag(ious).detach()
            else:
                ious = values[b] if isinstance(values, list) and b < len(values) else values

            target_classes_o = anno["labels"]
            target_classes = torch.full(
                src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
            )

            target_classes[0, src_idx] = target_classes_o

            target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

            target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
            target_score_o[0, src_idx] = ious.to(target_score_o.dtype)
            target_score = target_score_o.unsqueeze(-1) * target

            pred_score = F.sigmoid(src_logits).detach()
            weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score

            loss = F.binary_cross_entropy_with_logits(
                src_logits, target_score, weight=weight, reduction="none"
            )
            loss = loss.mean(1).sum() * src_logits.shape[1] / max(len(src_idx), 1)
            total_loss += loss
        print("total loss for vfl: ", total_loss)

        return {"loss_vfl": total_loss / batch_size}

    def loss_boxes(self, outputs, targets, indices, num_boxes, boxes_weight=None):
        assert "pred_boxes" in outputs
        print("=== 进入loss_boxes ===")

        # 强制同步
        torch.cuda.synchronize()

        pred_boxes_list = self._process_list_output(outputs, "pred_boxes")
        print("outputs: ", len(outputs["pred_logits"]), "targets: ", len(targets[0]["labels"]))
        total_bbox_loss = 0
        total_giou_loss = 0
        batch_size = len(pred_boxes_list)
        print("batch_size", batch_size)
        total_matched_boxes = 0

        # 确保所有索引都在正确的设备上
        device = pred_boxes_list[0].device if len(pred_boxes_list) > 0 else torch.device("cuda")
        indices = [(src_idx.to(device), tgt_idx.to(device)) for src_idx, tgt_idx in indices]

        for b, anno in enumerate(targets):
            src_boxes = pred_boxes_list[b]  # [1, num_queries, 4]
            src_idx, tgt_idx = indices[b]

            if len(src_boxes.shape) == 2:
                src_boxes = src_boxes.unsqueeze(0)
            print(
                "boxes shape1: ", src_boxes.shape, anno["boxes"].shape, src_idx.shape, tgt_idx.shape
            )

            print("src_boxes_matched: ", src_boxes.shape, min(src_idx), max(src_idx))
            src_boxes_matched = src_boxes[0, src_idx]  # [num_matched, 4]

            print("anno: ", anno["boxes"].shape)
            target_boxes_matched = anno["boxes"][tgt_idx]  # [num_matched, 4]

            print("boxes shape2: ", src_boxes_matched.shape, target_boxes_matched.shape)
            # print("devices: ", src_boxes_matched.device, target_boxes_matched.device)

            # 检查转换后的框坐标
            src_boxes_xyxy = box_cxcywh_to_xyxy(src_boxes_matched)
            target_boxes_xyxy = box_cxcywh_to_xyxy(target_boxes_matched)
            print("boxes shape3: ", src_boxes_xyxy.shape, target_boxes_xyxy.shape)

            num_matched = len(src_boxes_matched)
            total_matched_boxes += num_matched
            torch.cuda.synchronize()
            loss_bbox = F.l1_loss(src_boxes_matched, target_boxes_matched, reduction="sum")

            total_bbox_loss += loss_bbox

            # GIoU loss
            loss_giou = 1 - torch.diag(generalized_box_iou(src_boxes_xyxy, target_boxes_xyxy))
            if boxes_weight is not None:
                if isinstance(boxes_weight, list) and b < len(boxes_weight):
                    loss_giou = loss_giou * boxes_weight[b]
                else:
                    loss_giou = loss_giou * boxes_weight
            total_giou_loss += loss_giou.sum()
        if total_matched_boxes > 0:
            total_bbox_loss = total_bbox_loss / total_matched_boxes
            total_giou_loss = total_giou_loss / total_matched_boxes
        else:
            total_bbox_loss = torch.tensor(0.0, device=pred_boxes_list[0].device)
            total_giou_loss = torch.tensor(0.0, device=pred_boxes_list[0].device)
        print("total loss for boxes: ", total_bbox_loss, total_giou_loss)

        return {"loss_bbox": total_bbox_loss, "loss_giou": total_giou_loss}

    def loss_local(self, outputs, targets, indices, num_boxes, T=5):
        """Compute Fine-Grained Localization (FGL) Loss and Decoupled Distillation Focal (DDF) Loss."""
        losses = {}

        pred_corners_list = self._process_list_output(outputs, "pred_corners")
        pred_boxes_list = self._process_list_output(outputs, "pred_boxes")
        ref_points_list = self._process_list_output(outputs, "ref_points")

        if pred_corners_list is not None:
            total_fgl_loss = 0
            total_ddf_loss = 0
            batch_size = len(pred_corners_list)
            total_matched_corners = 0

            # 为DDF损失收集全局信息
            all_pred_corners = []
            all_teacher_corners = []
            all_teacher_logits = []
            all_masks = []
            all_ious = []

            for b in range(batch_size):
                pred_corners = pred_corners_list[b]  # [1, num_queries, 4*(reg_max+1)]
                pred_boxes = pred_boxes_list[b]  # [1, num_queries, 4]
                ref_points = ref_points_list[b] if ref_points_list else None  # [1, num_queries, 2]

                device = pred_corners.device
                indices = [(src_idx.to(device), tgt_idx.to(device)) for src_idx, tgt_idx in indices]

                # 为DDF损失收集数据
                all_pred_corners.append(pred_corners.reshape(-1, (self.reg_max + 1)))
                if "teacher_corners" in outputs and "teacher_logits" in outputs:
                    teacher_corners_list = self._process_list_output(outputs, "teacher_corners")
                    teacher_logits_list = self._process_list_output(outputs, "teacher_logits")
                    all_teacher_corners.append(
                        teacher_corners_list[b].reshape(-1, (self.reg_max + 1))
                    )
                    all_teacher_logits.append(teacher_logits_list[b])

                if indices[b] is not None and len(indices[b][0]) > 0:
                    src_idx, tgt_idx = indices[b]
                    target_boxes = targets[b]["boxes"][tgt_idx]  # [num_matched, 4]

                    # 正确的索引方式：在查询维度上索引
                    pred_corners_matched = pred_corners[0, src_idx]  # [num_matched, 4*(reg_max+1)]
                    pred_boxes_matched = pred_boxes[0, src_idx]  # [num_matched, 4]
                    ref_points_matched = (
                        ref_points[0, src_idx] if ref_points is not None else None
                    )  # [num_matched, 2]

                    # FGL loss
                    if ref_points_matched is not None:
                        with torch.no_grad():
                            # 为每个样本单独计算目标角点
                            target_corners, weight_right, weight_left = bbox2distance(
                                ref_points_matched,
                                box_cxcywh_to_xyxy(target_boxes),
                                self.reg_max,
                                outputs["reg_scale"],
                                outputs["up"],
                            )

                        ious = torch.diag(
                            box_iou(
                                box_cxcywh_to_xyxy(pred_boxes_matched),
                                box_cxcywh_to_xyxy(target_boxes),
                            )[0]
                        )
                        weight_targets = ious.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()

                        fgl_loss = self.unimodal_distribution_focal_loss(
                            pred_corners_matched.reshape(-1, (self.reg_max + 1)),
                            target_corners,
                            weight_right,
                            weight_left,
                            weight_targets,
                            avg_factor=len(pred_corners_matched),
                        )
                        total_fgl_loss += fgl_loss
                        total_matched_corners += len(pred_corners_matched)

                        # 为DDF损失收集数据
                        mask = torch.zeros(
                            pred_corners.shape[1], dtype=torch.bool, device=pred_corners.device
                        )
                        mask[src_idx] = True
                        all_masks.append(mask)
                        all_ious.append(ious)

            # 计算平均FGL损失
            if total_matched_corners > 0:
                losses["loss_fgl"] = total_fgl_loss / batch_size

            # DDF loss (如果有teacher输出)
            if (
                "teacher_corners" in outputs
                and "teacher_logits" in outputs
                and len(all_pred_corners) > 0
                and len(all_teacher_corners) > 0
            ):
                # 合并所有样本的数据
                pred_corners_flat = torch.cat(all_pred_corners)
                target_corners_flat = torch.cat(all_teacher_corners)
                teacher_logits_flat = torch.cat(all_teacher_logits)

                if not torch.equal(pred_corners_flat, target_corners_flat):
                    weight_targets_local = teacher_logits_flat.sigmoid().max(dim=-1)[0]

                    # 创建全局掩码
                    mask = (
                        torch.cat(all_masks)
                        if all_masks
                        else torch.tensor([], dtype=torch.bool, device=pred_corners_flat.device)
                    )
                    mask = mask.unsqueeze(-1).repeat(1, 1, 4).reshape(-1)

                    # 更新权重目标
                    if all_ious and len(all_ious) > 0:
                        # 计算每个样本的起始索引
                        start_idx = 0
                        for b, (sample_mask, sample_ious) in enumerate(zip(all_masks, all_ious)):
                            num_queries = all_pred_corners[b].shape[0]
                            matched_indices = torch.where(sample_mask)[0]

                            if len(matched_indices) > 0:
                                global_indices = start_idx + matched_indices
                                weight_targets_local[global_indices] = sample_ious.reshape_as(
                                    weight_targets_local[global_indices]
                                ).to(weight_targets_local.dtype)

                            start_idx += num_queries

                    weight_targets_local = (
                        weight_targets_local.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()
                    )

                    loss_match_local = (
                        weight_targets_local
                        * (T**2)
                        * (
                            nn.KLDivLoss(reduction="none")(
                                F.log_softmax(pred_corners_flat / T, dim=1),
                                F.softmax(target_corners_flat.detach() / T, dim=1),
                            )
                        ).sum(-1)
                    )

                    if "is_dn" not in outputs:
                        batch_scale = 8 / batch_size  # 根据批量大小调整
                        num_pos = (mask.sum() * batch_scale) ** 0.5
                        num_neg = ((~mask).sum() * batch_scale) ** 0.5
                    else:
                        num_pos, num_neg = self.num_pos, self.num_neg

                    loss_match_local1 = loss_match_local[mask].mean() if mask.any() else 0
                    loss_match_local2 = loss_match_local[~mask].mean() if (~mask).any() else 0
                    ddf_loss = (loss_match_local1 * num_pos + loss_match_local2 * num_neg) / (
                        num_pos + num_neg
                    )
                    total_ddf_loss += ddf_loss

                    losses["loss_ddf"] = total_ddf_loss

        return losses

    def _get_src_permutation_idx_single(self, indices_single, batch_idx, device):
        """单个样本的permutation indices - 简化版本"""
        src_idx, tgt_idx = indices_single
        batch_idx_tensor = torch.full_like(src_idx, batch_idx, device=device)
        return batch_idx_tensor, src_idx

    def _get_src_permutation_idx(self, indices):
        # 处理列表形式的indices
        if not indices or all(idx[0] is None for idx in indices):
            return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)

        batch_indices = []
        src_indices = []
        for i, (src, _) in enumerate(indices):
            if src is not None and len(src) > 0:
                batch_indices.append(torch.full_like(src, i))
                src_indices.append(src)

        if batch_indices:
            return torch.cat(batch_indices), torch.cat(src_indices)
        else:
            return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)

    def _get_tgt_permutation_idx(self, indices):
        # 处理列表形式的indices
        if not indices or all(idx[1] is None for idx in indices):
            return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)

        batch_indices = []
        tgt_indices = []
        for i, (_, tgt) in enumerate(indices):
            if tgt is not None and len(tgt) > 0:
                batch_indices.append(torch.full_like(tgt, i))
                tgt_indices.append(tgt)

        if batch_indices:
            return torch.cat(batch_indices), torch.cat(tgt_indices)
        else:
            return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)

    def _get_go_indices(self, indices, indices_aux_list):
        """Get a matching union set across all decoder layers."""
        print(f"Initial indices length: {len(indices)}")
        print(f"Number of aux indices lists: {len(indices_aux_list)}")

        for i, indices_aux in enumerate(indices_aux_list):
            print(f"Aux indices {i} length: {len(indices_aux)}")

        results = []
        for indices_aux in indices_aux_list:
            # 合并当前层和辅助层的匹配结果
            merged_indices = []
            for idx1, idx2 in zip(indices, indices_aux):
                if idx1[0] is not None and idx2[0] is not None:
                    merged_src = torch.cat([idx1[0], idx2[0]])
                    merged_tgt = torch.cat([idx1[1], idx2[1]])
                    merged_indices.append((merged_src, merged_tgt))
                elif idx1[0] is not None:
                    merged_indices.append(idx1)
                else:
                    merged_indices.append(idx2)
            indices = merged_indices
            print(f"After merging, indices length: {len(indices)}")

        # 为每个样本选择最佳匹配
        for idx in indices:
            if idx[0] is not None and len(idx[0]) > 0:
                ind_pairs = torch.stack([idx[0], idx[1]], dim=1)
                unique, counts = torch.unique(ind_pairs, return_counts=True, dim=0)
                count_sort_indices = torch.argsort(counts, descending=True)
                unique_sorted = unique[count_sort_indices]

                column_to_row = {}
                for pair in unique_sorted:
                    row_idx, col_idx = pair[0].item(), pair[1].item()
                    if row_idx not in column_to_row:
                        column_to_row[row_idx] = col_idx

                if column_to_row:
                    final_rows = torch.tensor(list(column_to_row.keys()), device=idx[0].device)
                    final_cols = torch.tensor(list(column_to_row.values()), device=idx[0].device)
                    results.append((final_rows.long(), final_cols.long()))
                else:
                    results.append(
                        (
                            torch.tensor([], device=idx[0].device),
                            torch.tensor([], device=idx[0].device),
                        )
                    )
            else:
                results.append(
                    (
                        torch.tensor(
                            [], device=indices[0][0].device if indices else torch.device("cpu")
                        ),
                        torch.tensor(
                            [], device=indices[0][0].device if indices else torch.device("cpu")
                        ),
                    )
                )

        print(f"Final results length: {len(results)}")
        return results

    def _clear_cache(self):
        self.fgl_targets, self.fgl_targets_dn = None, None
        self.own_targets, self.own_targets_dn = None, None
        self.num_pos, self.num_neg = None, None

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            "boxes": self.loss_boxes,
            "vfl": self.loss_labels_vfl,
            "local": self.loss_local,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, **kwargs):
        """修改后的forward函数，支持列表形式的输出"""
        B = len(targets)
        torch.cuda.synchronize()

        # 检索最后一层输出和目标之间的匹配
        indices = self.matcher(outputs, targets)["indices"]
        self._clear_cache()

        device = targets[0]["labels"].device

        # 计算目标框总数用于归一化
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        if "aux_outputs" in outputs:
            indices_aux_list, cached_indices, cached_indices_enc = [], [], []

            for b in range(B):
                indices_aux_per_sample = []

                for layer_output in outputs["aux_outputs"][b]:
                    single_output = {
                        "pred_logits": layer_output["pred_logits"],
                        "pred_boxes": layer_output["pred_boxes"],
                    }
                    single_target = [targets[b]]

                    indices_single = self.matcher(single_output, single_target)["indices"]
                    if len(indices_single) > 0:
                        indices_aux_per_sample.append(indices_single[0])
                    else:
                        indices_aux_per_sample.append(
                            (
                                torch.tensor([], dtype=torch.int64, device=device),
                                torch.tensor([], dtype=torch.int64, device=device),
                            )
                        )

                cached_indices.append(indices_aux_per_sample)
                indices_aux_list.append(indices_aux_per_sample)

            # 处理预输出 (pre_outputs) - 逐个样本处理
            if "pre_outputs" in outputs:
                indices_pre = []
                for b in range(B):
                    single_output = {
                        "pred_logits": outputs["pre_outputs"][b]["pred_logits"],
                        "pred_boxes": outputs["pre_outputs"][b]["pred_boxes"],
                    }
                    single_target = [targets[b]]

                    indices_single = self.matcher(single_output, single_target)["indices"]
                    if len(indices_single) > 0:
                        indices_pre.append(indices_single[0])
                    else:
                        indices_pre.append(
                            (
                                torch.tensor([], dtype=torch.int64, device=device),
                                torch.tensor([], dtype=torch.int64, device=device),
                            )
                        )
                # pre_outputs 其实和 aux_outputs 是一回事，只是 pre_ 是第0层的输出，在 decoder 中单独进行了处理。
                # 因此放到一起也算合理，但是后面的 enc_aux_outputs 就不是一回事了，因此 indices_enc_aux 是单独保存的。
                # 但是注意到 indices_enc_aux 也放到了 indices_aux_list 当中，这应该是为了创建一个跨所有层的统一匹配集合，用于某些特定的损失计算 indices_go
                cached_indices.append(indices_pre)
                indices_aux_list.append(indices_pre)

            # 处理编码器辅助输出 (enc_aux_outputs) - 逐个样本处理
            if "enc_aux_outputs" in outputs and outputs["enc_aux_outputs"]:
                indices_enc_aux = []
                for b in range(B):
                    single_output = {
                        "pred_logits": outputs["enc_aux_outputs"][b]["pred_logits"],
                        "pred_boxes": outputs["enc_aux_outputs"][b]["pred_boxes"],
                    }
                    single_target = [targets[b]]

                    indices_single = self.matcher(single_output, single_target)["indices"]
                    if len(indices_single) > 0:
                        indices_enc_aux.append(indices_single[0])
                    else:
                        indices_enc_aux.append(
                            (
                                torch.tensor([], dtype=torch.int64, device=device),
                                torch.tensor([], dtype=torch.int64, device=device),
                            )
                        )

                    cached_indices_enc.append(indices_enc_aux)
                    indices_aux_list.append(indices_enc_aux)

            # 计算全局匹配
            indices_go = self._get_go_indices(
                indices, indices_aux_list
            )  # 输入的是最后一层的预测匹配结果，以及 aux 和 enc_aux 预测的匹配结果
            print("indices_go", indices_go)

            # 计算全局匹配的框数量
            num_boxes_go = sum(len(x[0]) for x in indices_go if len(x[0]) > 0)
            num_boxes_go = torch.as_tensor([num_boxes_go], dtype=torch.float, device=device)
            if is_dist_available_and_initialized():
                torch.distributed.all_reduce(num_boxes_go)
            num_boxes_go = torch.clamp(num_boxes_go / get_world_size(), min=1).item()
        else:
            indices_go = indices
            num_boxes_go = num_boxes
        print("============================ 1 ==============================")
        # 计算所有请求的损失
        losses = {}
        print(self.losses)
        for loss in self.losses:
            use_uni_set = self.use_uni_set and (loss in ["boxes", "local"])
            print(loss, use_uni_set)
            indices_in = indices_go if use_uni_set else indices
            num_boxes_in = num_boxes_go if use_uni_set else num_boxes
            meta = self.get_loss_meta_info(loss, outputs, targets, indices_in)
            # print("enc_targets1: ", outputs, targets)
            l_dict = self.get_loss(loss, outputs, targets, indices_in, num_boxes_in, **meta)
            l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
            losses.update(l_dict)
        print("============================ 2 ==============================")
        print(
            "aux_outputs structure:",
            len(outputs["aux_outputs"]),
            len(outputs["aux_outputs"][0]),
            len(outputs["aux_outputs"][1]),
        )
        # 处理辅助损失
        if "aux_outputs" in outputs:
            # 获取辅助层的数量
            num_aux_layers = len(outputs["aux_outputs"][0]) if outputs["aux_outputs"] else 0
            batch_size_aux = len(outputs["aux_outputs"])

            print(
                f"Processing aux_outputs: batch_size={batch_size_aux}, num_layers={num_aux_layers}"
            )

            for b in range(B):
                for i, single_output in enumerate(outputs["aux_outputs"][b]):
                    if "local" in self.losses:
                        single_output["up"], single_output["reg_scale"] = (
                            outputs["up"],
                            outputs["reg_scale"],
                        )

                    single_target = [targets[b]]

                    for loss in self.losses:
                        use_uni_set = self.use_uni_set and (loss in ["boxes", "local"])
                        indices_in = indices_go[b] if use_uni_set else cached_indices[b][i]
                        num_boxes_in = num_boxes_go[b] if use_uni_set else num_boxes

                        meta = self.get_loss_meta_info(
                            loss, single_output, single_target, indices_in
                        )
                        l_dict = self.get_loss(
                            loss, single_output, single_target, indices_in, num_boxes_in, **meta
                        )

                        for k in list(l_dict.keys()):
                            if k in self.weight_dict:
                                loss_key = k + f"_aux_{i}"
                                loss_value = l_dict[k] * self.weight_dict[k]
                                # 累加到总损失
                                if loss_key not in losses:
                                    losses[loss_key] = 0
                                losses[loss_key] += loss_value

        print("=========================== 3 ==============================")
        # 处理预输出
        if "pre_outputs" in outputs:
            pre_outputs = outputs["pre_outputs"]
            indices_pre = []

            # pre_outputs 是一个字典，包含 pred_logits 和 pred_boxes 两个列表
            for b in range(B):
                # 为当前样本构建单样本输出字典
                single_output = {
                    "pred_logits": pre_outputs[b]["pred_logits"],  # 添加batch维度
                    "pred_boxes": pre_outputs[b]["pred_boxes"],
                }

                single_target = [targets[b]]
                indices_single = self.matcher(single_output, single_target)["indices"]

                if len(indices_single) > 0:
                    indices_pre.append(indices_single[0])
                else:
                    indices_pre.append(
                        (torch.tensor([], dtype=torch.int64), torch.tensor([], dtype=torch.int64))
                    )

            cached_indices.append(indices_pre)
            indices_aux_list.append(indices_pre)
            print(f"Added indices_pre with length {len(indices_pre)}")
        print("=========================== 4 ==============================")
        # 处理编码器辅助损失
        if "enc_aux_outputs" in outputs:
            assert "enc_meta" in outputs, ""
            class_agnostic = outputs["enc_meta"]["class_agnostic"]
            if class_agnostic:
                orig_num_classes = self.num_classes
                self.num_classes = 1
                enc_targets = copy.deepcopy(targets)
                for t in enc_targets:
                    t["labels"] = torch.zeros_like(t["labels"])
            else:
                enc_targets = targets
            print("=========================== 5 ==============================")
            for i, aux_outputs in enumerate(outputs["enc_aux_outputs"]):
                for loss in self.losses:
                    use_uni_set = self.use_uni_set and (loss == "boxes")
                    indices_in = indices_go if use_uni_set else cached_indices_enc[i]
                    num_boxes_in = num_boxes_go if use_uni_set else num_boxes
                    meta = self.get_loss_meta_info(loss, aux_outputs, enc_targets, indices_in)
                    torch.cuda.synchronize()

                    l_dict = self.get_loss(
                        loss, aux_outputs, enc_targets, indices_in, num_boxes_in, **meta
                    )
                    l_dict = {
                        k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict
                    }
                    l_dict = {k + f"_enc_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

            if class_agnostic:
                self.num_classes = orig_num_classes
        print("=========================== 6 ==============================")
        # 处理去噪训练损失
        if "dn_outputs" in outputs:
            assert "dn_meta" in outputs, ""
            indices_dn = self.get_cdn_matched_indices(outputs["dn_meta"], targets)
            dn_num_boxes = (
                num_boxes * outputs["dn_meta"][0]["dn_num_group"]
                if outputs["dn_meta"]
                else num_boxes
            )

            # 获取batch大小和去噪层数量
            batch_size_dn = len(outputs["dn_outputs"])
            num_dn_layers = len(outputs["dn_outputs"][0]) if batch_size_dn > 0 else 0

            print(f"Processing dn_outputs: batch_size={batch_size_dn}, num_layers={num_dn_layers}")

            # 对每个样本循环
            for b in range(batch_size_dn):
                # 对每个去噪层循环
                for layer_idx in range(num_dn_layers):
                    # 获取当前样本当前层的输出
                    layer_output = outputs["dn_outputs"][b][layer_idx]

                    if "local" in self.losses:
                        layer_output["is_dn"] = True
                        layer_output["up"] = outputs["up"]
                        layer_output["reg_scale"] = outputs["reg_scale"]

                    # 为当前样本计算损失
                    single_target = [targets[b]]
                    indices_in = [indices_dn[b]]  # 包装成列表

                    for loss in self.losses:
                        meta = self.get_loss_meta_info(
                            loss, layer_output, single_target, indices_in
                        )
                        l_dict = self.get_loss(
                            loss, layer_output, single_target, indices_in, dn_num_boxes, **meta
                        )
                        l_dict = {
                            k: l_dict[k] * self.weight_dict[k]
                            for k in l_dict
                            if k in self.weight_dict
                        }
                        l_dict = {k + f"_dn_{layer_idx}": v for k, v in l_dict.items()}
                        losses.update(l_dict)
            # 处理dn_pre_outputs
            if "dn_pre_outputs" in outputs:
                print("processing dn_pre_outputs ==============================")
                dn_pre_outputs = outputs["dn_pre_outputs"]
                batch_size_dn_pre = len(dn_pre_outputs)
                print(f"Processing dn_pre_outputs: batch_size={batch_size_dn_pre}")

                for b in range(batch_size_dn_pre):
                    # 为当前样本构建单样本输出字典
                    single_output = {
                        "pred_logits": dn_pre_outputs[b]["pred_logits"],
                        "pred_boxes": dn_pre_outputs[b]["pred_boxes"],
                    }

                    # 为当前样本计算损失
                    single_target = [targets[b]]
                    indices_in = [indices_dn[b]]  # 包装成列表

                    for loss in self.losses:
                        meta = self.get_loss_meta_info(
                            loss, single_output, single_target, indices_in
                        )
                        l_dict = self.get_loss(
                            loss, single_output, single_target, indices_in, dn_num_boxes, **meta
                        )
                        l_dict = {
                            k: l_dict[k] * self.weight_dict[k]
                            for k in l_dict
                            if k in self.weight_dict
                        }
                        l_dict = {k + "_dn_pre": v for k, v in l_dict.items()}
                        losses.update(l_dict)
        print("=========================== 7 ==============================")
        # 处理DEFE损失
        if "defe" in outputs:
            # 计算DEFE回归损失
            min_num_select = outputs["defe"]["min_num_select"]
            max_num_select = outputs["defe"]["max_num_select"]

            reg_targets = []
            for i in range(len(targets)):
                tgt_num = targets[i]["labels"].shape[0]
                if tgt_num < min_num_select:
                    tgt_num = min_num_select
                elif tgt_num > max_num_select:
                    tgt_num = max_num_select
                reg_targets.append((tgt_num - min_num_select) / (max_num_select - min_num_select))

            reg_targets = torch.tensor(reg_targets, dtype=torch.float32).to(
                outputs["defe"]["reg_value"].device
            )
            reg_value = outputs["defe"]["reg_value"]

            with torch.amp.autocast("cuda", dtype=torch.float16):
                diff = reg_value - reg_targets
                penalty_weights = torch.where(diff < 0, 2.0, 1.0).to(diff.device)
                defe_reg_loss = (penalty_weights * (diff**2)).mean()
            losses["defe_reg_loss"] = defe_reg_loss

            # 计算DEFE密度图损失
            density_map = outputs["defe"]["defe_feature"]
            gt_density_map = outputs["defe"]["gt_density_map"]
            with torch.amp.autocast("cuda", dtype=torch.float16):
                diff = density_map - gt_density_map
                underestimation_mask = (density_map < gt_density_map).float()
                penalty_weight = (
                    1 + self.density_recall_penalty * gt_density_map * underestimation_mask
                )
                defe_density_loss = (
                    penalty_weight * (diff**2)
                ).mean() * self.defe_density_map_weight
            losses["defe_density_loss"] = defe_density_loss
        print("defe_density_loss", losses["defe_density_loss"])
        print("=========================== 8 ==============================")
        # 处理NaN值
        losses = {k: torch.nan_to_num(v, nan=0.0) for k, v in losses.items()}
        if self.training:
            print(
                "out['defe']: ",
                outputs["defe"]["reg_value"].shape,
                outputs["defe"]["density_map"].shape,
                outputs["defe"]["defe_feature"].shape,
                outputs["defe"]["density_map_pooled"].shape,
                outputs["defe"]["defe_window_mask"].shape,
            )
        return losses

    def get_loss_meta_info(self, loss, outputs, targets, indices):
        if self.boxes_weight_format is None:
            return {}

        pred_boxes_list = self._process_list_output(outputs, "pred_boxes")
        ious_list = []

        for b in range(len(pred_boxes_list)):
            if len(indices) > b and indices[b] is not None and len(indices[b][0]) > 0:
                idx = self._get_src_permutation_idx_single(indices[b], b, pred_boxes_list[b].device)
                src_boxes = pred_boxes_list[b][idx]
                target_boxes = targets[b]["boxes"][indices[b][1]]

                if self.boxes_weight_format == "iou":
                    iou, _ = box_iou(
                        box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes)
                    )
                    iou = torch.diag(iou)
                elif self.boxes_weight_format == "giou":
                    iou = torch.diag(
                        generalized_box_iou(
                            box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes)
                        )
                    )
                else:
                    raise AttributeError()
                ious_list.append(iou)
            else:
                ious_list.append(torch.tensor([], device=pred_boxes_list[b].device))

        if loss in ("boxes",):
            meta = {"boxes_weight": ious_list}
        elif loss in ("vfl", "mal"):
            meta = {"values": ious_list}
        else:
            meta = {}

        return meta

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        """get_cdn_matched_indices - 支持列表形式的dn_meta，每个样本单独处理"""
        if not dn_meta:
            return []
        dn_match_indices = []

        # 遍历每个样本
        for i in range(len(targets)):
            # 获取当前样本的dn_meta
            if i < len(dn_meta):
                sample_dn_meta = dn_meta[i]
                # 从元组中提取张量
                dn_positive_idx = sample_dn_meta["dn_positive_idx"][0]  # 注意这里
                dn_num_group = sample_dn_meta["dn_num_group"]
                print(
                    f"Sample {i}: dn_positive_idx shape = {dn_positive_idx.shape}, dn_num_group = {dn_num_group}"
                )
            else:
                # 如果没有对应的dn_meta，添加空匹配
                dn_match_indices.append(
                    (
                        torch.zeros(0, dtype=torch.int64, device=targets[0]["labels"].device),
                        torch.zeros(0, dtype=torch.int64, device=targets[0]["labels"].device),
                    )
                )
                continue

            num_gt = len(targets[i]["labels"])
            print(f"Sample {i}: num_gt = {num_gt}")

            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=targets[0]["labels"].device)
                gt_idx = gt_idx.tile(dn_num_group)
                print(f"Sample {i}: gt_idx shape = {gt_idx.shape}")

                # 确保dn_positive_idx的长度与gt_idx匹配
                if len(dn_positive_idx) == len(gt_idx):
                    dn_match_indices.append((dn_positive_idx, gt_idx))
                    print(f"Sample {i}: Added valid match indices")
                else:
                    # 如果不匹配，添加空匹配
                    print(
                        f"Sample {i}: Length mismatch - dn_positive_idx: {len(dn_positive_idx)}, gt_idx: {len(gt_idx)}"
                    )
                    dn_match_indices.append(
                        (
                            torch.zeros(0, dtype=torch.int64, device=targets[0]["labels"].device),
                            torch.zeros(0, dtype=torch.int64, device=targets[0]["labels"].device),
                        )
                    )
            else:
                # 如果没有目标，添加空匹配
                print(f"Sample {i}: No targets")
                dn_match_indices.append(
                    (
                        torch.zeros(0, dtype=torch.int64, device=targets[0]["labels"].device),
                        torch.zeros(0, dtype=torch.int64, device=targets[0]["labels"].device),
                    )
                )
        return dn_match_indices

    def feature_loss_function(self, fea, target_fea):
        loss = (fea - target_fea) ** 2 * ((fea > 0) | (target_fea > 0)).float()
        return torch.abs(loss)

    def unimodal_distribution_focal_loss(
        self, pred, label, weight_right, weight_left, weight=None, reduction="sum", avg_factor=None
    ):
        dis_left = label.long()
        dis_right = dis_left + 1

        loss = F.cross_entropy(pred, dis_left, reduction="none") * weight_left.reshape(
            -1
        ) + F.cross_entropy(pred, dis_right, reduction="none") * weight_right.reshape(-1)

        if weight is not None:
            weight = weight.float()
            loss = loss * weight

        if avg_factor is not None:
            loss = loss.sum() / avg_factor
        elif reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()

        return loss

    def get_gradual_steps(self, outputs):
        if "aux_outputs" in outputs:
            num_layers = len(outputs["aux_outputs"]) + 1
        else:
            num_layers = 1
        step = 0.5 / (num_layers - 1) if num_layers > 1 else 0
        opt_list = [0.5 + step * i for i in range(num_layers)] if num_layers > 1 else [1]
        return opt_list
