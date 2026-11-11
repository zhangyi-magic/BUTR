"""
Dome-DETR: Dome-DETR: DETR with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object Detection
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.
"""

import contextlib
import copy
import os

import numpy as np
import torch
import torch.distributed as dist

from ...core import register

__all__ = [
    "AitodCocoEvaluator",
]

_CPU_PROCESS_GROUP = None


def is_main_process():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def get_cpu_process_group():
    global _CPU_PROCESS_GROUP

    if (not dist.is_available()) or (not dist.is_initialized()):
        return None
    if dist.get_world_size() == 1:
        return None

    if _CPU_PROCESS_GROUP is None:
        _CPU_PROCESS_GROUP = dist.new_group(backend="gloo")

    return _CPU_PROCESS_GROUP


def all_gather_cpu(data):
    if (not dist.is_available()) or (not dist.is_initialized()):
        return [data]

    world_size = dist.get_world_size()
    if world_size == 1:
        return [data]

    data_list = [None] * world_size
    dist.all_gather_object(data_list, data, group=get_cpu_process_group())
    return data_list


@register()
class AitodCocoEvaluator(object):
    def __init__(self, coco_gt, iou_types):
        global COCO, COCOeval_faster, mask_util
        try:
            from faster_coco_eval_aitod import COCO, COCOeval_faster
            import faster_coco_eval_aitod.core.mask as mask_util
        except ImportError as exc:
            raise ImportError(
                "AI-TOD evaluation requires faster-coco-eval-aitod; "
                "install the dependencies in requirements.txt on the evaluation machine."
            ) from exc
        assert isinstance(iou_types, (list, tuple))
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt: COCO = coco_gt
        self.iou_types = iou_types

        self.coco_eval = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = COCOeval_faster(
                coco_gt, iouType=iou_type, print_function=print, separate_eval=True
            )

        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}

    def cleanup(self):
        self.coco_eval = {}
        for iou_type in self.iou_types:
            self.coco_eval[iou_type] = COCOeval_faster(
                self.coco_gt, iouType=iou_type, print_function=print, separate_eval=True
            )
        self.img_ids = []
        self.eval_imgs = {k: [] for k in self.iou_types}

    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)
            coco_eval = self.coco_eval[iou_type]

            # suppress pycocotools prints
            with open(os.devnull, "w") as devnull:
                with contextlib.redirect_stdout(devnull):
                    coco_dt = self.coco_gt.loadRes(results) if results else COCO()
                    coco_eval.cocoDt = coco_dt
                    coco_eval.params.imgIds = list(img_ids)
                    coco_eval.evaluate()

            self.eval_imgs[iou_type].append(
                np.array(coco_eval._evalImgs_cpp).reshape(
                    len(coco_eval.params.catIds),
                    len(coco_eval.params.areaRng),
                    len(coco_eval.params.imgIds),
                )
            )

    def synchronize_between_processes(self):
        merged_img_ids = None

        for iou_type in self.iou_types:
            img_ids, eval_imgs = merge(self.img_ids, self.eval_imgs[iou_type])

            if merged_img_ids is None:
                merged_img_ids = img_ids

            coco_eval = self.coco_eval[iou_type]
            coco_eval.params.imgIds = img_ids
            coco_eval._paramsEval = copy.deepcopy(coco_eval.params)
            coco_eval._evalImgs_cpp = eval_imgs

        self.img_ids = merged_img_ids

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self):
        for iou_type, coco_eval in self.coco_eval.items():
            print("IoU metric: {}".format(iou_type))
            coco_eval.summarize()

    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        elif iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        elif iou_type == "keypoints":
            return self.prepare_for_coco_keypoint(predictions)
        else:
            raise ValueError("Unknown iou type {}".format(iou_type))

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results

    def prepare_for_coco_segmentation(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"]
            labels = prediction["labels"]
            masks = prediction["masks"]

            masks = masks > 0.5

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            rles = [
                mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0]
                for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "segmentation": rle,
                        "score": scores[k],
                    }
                    for k, rle in enumerate(rles)
                ]
            )
        return coco_results

    def prepare_for_coco_keypoint(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            keypoints = prediction["keypoints"]
            keypoints = keypoints.flatten(start_dim=1).tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "keypoints": keypoint,
                        "score": scores[k],
                    }
                    for k, keypoint in enumerate(keypoints)
                ]
            )
        return coco_results


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def merge(img_ids, eval_imgs):
    all_img_ids = all_gather_cpu(img_ids)
    all_eval_imgs = all_gather_cpu(eval_imgs)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.extend(p)

    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, axis=2)

    # keep only unique images and keep eval_imgs aligned
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
    merged_eval_imgs = merged_eval_imgs[:, :, idx]

    return merged_img_ids.tolist(), merged_eval_imgs.ravel().tolist()
