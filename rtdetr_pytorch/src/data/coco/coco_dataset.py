"""
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

COCO dataset which returns image_id for evaluation.
Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
import numpy as np
import torch
import torch.utils.data
from collections import defaultdict
import os

import torchvision
torchvision.disable_beta_transforms_warning()

from torchvision import tv_tensors as datapoints

from pycocotools import mask as coco_mask

from src.core import register

__all__ = ['CocoDetection']


@register
class CocoDetection(torchvision.datasets.CocoDetection):
    __inject__ = ['transforms']
    __share__ = ['remap_mscoco_category', 'fraction']
    
    def __init__(self, img_folder, ann_file, transforms, return_masks, remap_mscoco_category=False, fraction=0.01):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks, remap_mscoco_category)
        self.img_folder = img_folder
        self.ann_file = ann_file
        self.return_masks = return_masks
        self.remap_mscoco_category = remap_mscoco_category

        # 获取原始数据集大小
        before_size = len(self.ids)
        
        # 统计每个类别的bbox数量
        category_bbox_counts = defaultdict(int)
        image_to_bboxes = defaultdict(list)  # 存储每个图像中的bbox信息
        
        for img_id in self.ids:
            # 验证图像文件是否存在
            img_info = self.coco.loadImgs(img_id)
            if not img_info:
                print(f"Warning: Image ID {img_id} not found in COCO dataset")
                continue
                
            img_path = os.path.join(self.img_folder, img_info[0]['file_name'])
            if not os.path.exists(img_path):
                print(f"Warning: Image file not found: {img_path}")
                continue
            
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            
            # 统计该图像中每个类别的bbox数量
            img_bbox_counts = defaultdict(int)
            for ann in anns:
                if 'iscrowd' not in ann or ann['iscrowd'] == 0:
                    cat_id = ann['category_id']
                    img_bbox_counts[cat_id] += 1
                    category_bbox_counts[cat_id] += 1
            
            # 存储图像ID和其包含的bbox信息
            if img_bbox_counts:
                image_to_bboxes[img_id] = img_bbox_counts
        
        # 计算目标每个类别的bbox数量
        target_bbox_per_category = int(min(category_bbox_counts.values()) * fraction)
        print(f"Target bbox count per category: {target_bbox_per_category}")
        
        # 为每个类别选择图像，直到达到目标bbox数量
        selected_image_ids = set()
        category_bbox_selected = defaultdict(int)
        
        # 按类别循环，直到所有类别都达到目标数量
        while True:
            all_categories_complete = True
            for cat_id, total_bboxes in category_bbox_counts.items():
                if category_bbox_selected[cat_id] >= target_bbox_per_category:
                    continue
                    
                all_categories_complete = False
                
                # 找到包含该类别且未被选中的图像
                available_images = [
                    img_id for img_id, bbox_counts in image_to_bboxes.items()
                    if img_id not in selected_image_ids and cat_id in bbox_counts
                ]
                
                if not available_images:
                    continue
                    
                # 选择包含最多该类别bbox的图像
                selected_img_id = max(
                    available_images,
                    key=lambda x: image_to_bboxes[x][cat_id]
                )
                
                selected_image_ids.add(selected_img_id)
                category_bbox_selected[cat_id] += image_to_bboxes[selected_img_id][cat_id]
                
                # 更新其他类别的计数
                for other_cat_id, count in image_to_bboxes[selected_img_id].items():
                    if other_cat_id != cat_id:
                        category_bbox_selected[other_cat_id] += count
            
            if all_categories_complete:
                break
        
        # 更新self.ids
        self.ids = list(selected_image_ids)
        print(f"{self.ann_file}, before_size: {before_size}, sample size: {len(self.ids)}")
        
        # 打印每个类别的bbox数量
        print("\nCategory bbox distribution in sampled dataset:")
        for cat_id, count in sorted(category_bbox_selected.items()):
            cat_name = self.coco.cats[cat_id]['name']
            print(f"  {cat_name} ({cat_id}): {count} bboxes")
        
        # 打印每个类别的图像数量
        category_image_counts = defaultdict(int)
        for img_id in self.ids:
            for cat_id in image_to_bboxes[img_id].keys():
                category_image_counts[cat_id] += 1
        
        print("\nCategory image distribution in sampled dataset:")
        for cat_id, count in sorted(category_image_counts.items()):
            cat_name = self.coco.cats[cat_id]['name']
            print(f"  {cat_name} ({cat_id}): {count} images")

    def __getitem__(self, idx):
        try:
            img_id = self.ids[idx]
            img_info = self.coco.loadImgs(img_id)
            if not img_info:
                raise ValueError(f"Image ID {img_id} not found in COCO dataset")
                
            img_path = os.path.join(self.img_folder, img_info[0]['file_name'])
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image file not found: {img_path}")
                
            img, target = super(CocoDetection, self).__getitem__(idx)
            image_id = self.ids[idx]
            target = {'image_id': image_id, 'annotations': target}
            img, target = self.prepare(img, target)

            # ['boxes', 'masks', 'labels']:
            if 'boxes' in target:
                target['boxes'] = datapoints.BoundingBoxes(
                    target['boxes'], 
                    format=datapoints.BoundingBoxFormat.XYXY,
                    canvas_size=img.size[::-1]) # h w

            if 'masks' in target:
                target['masks'] = datapoints.Mask(target['masks'])

            if self._transforms is not None:
                img, target = self._transforms(img, target)
                
            return img, target
        except Exception as e:
            print(f"Error loading image at index {idx}: {str(e)}")
            # 返回一个空样本
            return None, None

    def extra_repr(self) -> str:
        s = f' img_folder: {self.img_folder}\n ann_file: {self.ann_file}\n'
        s += f' return_masks: {self.return_masks}\n'
        if hasattr(self, '_transforms') and self._transforms is not None:
            s += f' transforms:\n   {repr(self._transforms)}'

        return s


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False, remap_mscoco_category=False):
        self.return_masks = return_masks
        self.remap_mscoco_category = remap_mscoco_category

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        if self.remap_mscoco_category:
            classes = [mscoco_category2label[obj["category_id"]] for obj in anno]
        else:
            classes = [obj["category_id"] for obj in anno]
            
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(w), int(h)])
        target["size"] = torch.as_tensor([int(w), int(h)])
    
        return image, target


mscoco_category2name = {
    1: 'person',
    2: 'bicycle',
    3: 'car',
    4: 'motorcycle',
    5: 'airplane',
    6: 'bus',
    7: 'train',
    8: 'truck',
    9: 'boat',
    10: 'traffic light',
    11: 'fire hydrant',
    13: 'stop sign',
    14: 'parking meter',
    15: 'bench',
    16: 'bird',
    17: 'cat',
    18: 'dog',
    19: 'horse',
    20: 'sheep',
    21: 'cow',
    22: 'elephant',
    23: 'bear',
    24: 'zebra',
    25: 'giraffe',
    27: 'backpack',
    28: 'umbrella',
    31: 'handbag',
    32: 'tie',
    33: 'suitcase',
    34: 'frisbee',
    35: 'skis',
    36: 'snowboard',
    37: 'sports ball',
    38: 'kite',
    39: 'baseball bat',
    40: 'baseball glove',
    41: 'skateboard',
    42: 'surfboard',
    43: 'tennis racket',
    44: 'bottle',
    46: 'wine glass',
    47: 'cup',
    48: 'fork',
    49: 'knife',
    50: 'spoon',
    51: 'bowl',
    52: 'banana',
    53: 'apple',
    54: 'sandwich',
    55: 'orange',
    56: 'broccoli',
    57: 'carrot',
    58: 'hot dog',
    59: 'pizza',
    60: 'donut',
    61: 'cake',
    62: 'chair',
    63: 'couch',
    64: 'potted plant',
    65: 'bed',
    67: 'dining table',
    70: 'toilet',
    72: 'tv',
    73: 'laptop',
    74: 'mouse',
    75: 'remote',
    76: 'keyboard',
    77: 'cell phone',
    78: 'microwave',
    79: 'oven',
    80: 'toaster',
    81: 'sink',
    82: 'refrigerator',
    84: 'book',
    85: 'clock',
    86: 'vase',
    87: 'scissors',
    88: 'teddy bear',
    89: 'hair drier',
    90: 'toothbrush'
}

mscoco_category2label = {k: i for i, k in enumerate(mscoco_category2name.keys())}
mscoco_label2category = {v: k for k, v in mscoco_category2label.items()}