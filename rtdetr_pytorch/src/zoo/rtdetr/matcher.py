"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
Modules to compute the matching cost and solve the corresponding LSAP.

by lyuwenyu
"""
import numpy as np
import torch
import torch.nn.functional as F 

from scipy.optimize import linear_sum_assignment
from torch import nn

from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou

from src.core import register
from src.zoo.loss.wasserstein_loss import WassersteinLoss


# TODO 从onetomany入手，增加训练阶段的one to many, 预测阶段不需要one to many

@register
class HungarianMatcher(nn.Module):
    def __init__(self, weight_dict, use_focal_loss=False, alpha=0.25, gamma=2.0, group_detr=1, o2m=4):
        super().__init__()
        # 分类损失、边界框损失、iou损失
        self.group_detr = group_detr
        self.cost_class = weight_dict['cost_class']
        self.cost_bbox = weight_dict['cost_bbox']
        self.cost_giou = weight_dict['cost_giou']
        # self.wasserstein_loss = WassersteinLoss()

        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma

        assert self.cost_class != 0 or self.cost_bbox != 0 or self.cost_giou != 0, "all costs cant be 0"
        self.o2m = o2m  # one-to-many 的倍数

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # 1. 打平，合并batch_size和num_queries维度, softmax预测结果
        # We flatten to compute the cost matrices in a batch
        if self.use_focal_loss:
            out_prob = F.sigmoid(outputs["pred_logits"].flatten(0, 1))
        else:
            out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)

        # 2. 合并batch_size和num_queries维度,
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # 3. 获取target 标签和bbox
        # Also concat the target labels and boxes
        if out_prob.shape[-1] == 2:
            tgt_ids = torch.cat([v["labels"] for v in targets])
            tgt_ids = torch.ones(tgt_ids.shape).int()
            tgt_bbox = torch.cat([v["boxes"] for v in targets])
        else:
            tgt_ids = torch.cat([v["labels"] for v in targets])
            tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # 4. 计算focal_loss
        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        if self.use_focal_loss:
            out_prob = out_prob[:, tgt_ids]
            neg_cost_class = (1 - self.alpha) * (out_prob**self.gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = self.alpha * ((1 - out_prob)**self.gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class - neg_cost_class
        else:
            cost_class = -out_prob[:, tgt_ids]

        # 5. 计算box l1 loss
        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # 6. 计算giou_loss
        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        # cost_giou = self.wasserstein_loss(out_bbox, tgt_bbox) + cost_giou

        # 7. 计算最终的二分匹配质量分数
        # Compute quality loss
        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        # 8. 获取目标框的数量，将成本矩阵C按照size大小进行分割，每个分割对应一个目标的预测框和真实框的成本矩阵
        # 对每个分割的成本矩阵c[i]使用匈牙利算法(linear_sum_assigment)进行匹配，这个算法会返回一个最优的匹配结果，通常是最小化总成本
        # 每一列代表某个目标与所有query计算出来的成本大小
        # sizes = [len(v["boxes"]) for v in targets]
        # indice包含每个目标框的匹配结果
        # indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

        # 5. 支持将查询分为多个组进行匹配, 每个组求最优解，相当于每个target对应多个最佳的query, 但只是局部最佳
        sizes = [len(v["boxes"]) for v in targets]

        # 对每个图像分别进行匹配
        if self.training and self.o2m > 0:
            indices = []
            for i, c in enumerate(C.split(sizes, -1)):
                c_i = c[i]  # shape: [num_queries, num_targets]

                all_src_indices = []
                all_tgt_indices = []

                # 为每个target找到最好的o2m个queries
                for tgt_idx in range(c_i.shape[1]):
                    target_costs = c_i[:, tgt_idx]  # 所有queries对这个target的成本

                    # 找到成本最小的o2m个queries
                    topk_values, topk_indices = torch.topk(target_costs,
                                                           min(self.o2m, len(target_costs)),
                                                           largest=False)

                    # 过滤成本过高的匹配
                    valid_mask = topk_values < 10.0
                    valid_indices = topk_indices[valid_mask]

                    for src_idx in valid_indices:
                        all_src_indices.append(src_idx.item())
                        all_tgt_indices.append(tgt_idx)

                indices.append((torch.as_tensor(all_src_indices, dtype=torch.int64),
                                torch.as_tensor(all_tgt_indices, dtype=torch.int64)))
            
            return indices
        else:
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
