import torch
from torch import nn


def wasserstein_loss(pred, target, eps=1e-7, mode='exp', gamma=1, constant=12.8):
    """
    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (m, 4).
    """
    # 计算所有预测框和所有目标框之间的中心点距离
    center1 = (pred[:, :2] + pred[:, 2:]) / 2  # (n, 2)
    center2 = (target[:, :2] + target[:, 2:]) / 2  # (m, 2)
    
    # 计算所有预测框和所有目标框之间的中心点距离矩阵
    # 使用广播机制计算所有组合
    center_distance = torch.sum((center1.unsqueeze(1) - center2.unsqueeze(0)) ** 2, dim=2) + eps  # (n, m)

    # 计算宽度和高度
    w1 = pred[:, 2] - pred[:, 0] + eps  # (n,)
    h1 = pred[:, 3] - pred[:, 1] + eps  # (n,)
    w2 = target[:, 2] - target[:, 0] + eps  # (m,)
    h2 = target[:, 3] - target[:, 1] + eps  # (m,)

    # 计算宽度和高度距离
    wh_distance = ((w1.unsqueeze(1) - w2.unsqueeze(0)) ** 2 + 
                  (h1.unsqueeze(1) - h2.unsqueeze(0)) ** 2) / 4  # (n, m)

    # 计算总的Wasserstein距离
    wasserstein_2 = center_distance + wh_distance  # (n, m)

    if mode == 'exp':
        normalized_wasserstein = torch.exp(-torch.sqrt(wasserstein_2) / constant)
        wloss = 1 - normalized_wasserstein
    elif mode == 'sqrt':
        wloss = torch.sqrt(wasserstein_2)
    elif mode == 'log':
        wloss = torch.log(wasserstein_2 + 1)
    elif mode == 'norm_sqrt':
        wloss = 1 - 1 / (gamma + torch.sqrt(wasserstein_2))
    elif mode == 'w2':
        wloss = wasserstein_2

    return wloss  # 返回 (n, m) 的损失矩阵


class WassersteinLoss(nn.Module):

    def __init__(self, eps=1e-6, reduction='mean', loss_weight=1.0, mode='exp', gamma=2, constant=12.8):
        super(WassersteinLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.mode = mode
        self.gamma = gamma
        self.constant = constant    # constant = 12.8 for AI-TOD

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        if weight is not None and not torch.any(weight > 0):
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # giou_loss of shape (n,)
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        
        # 计算损失矩阵
        loss_matrix = wasserstein_loss(
            pred,
            target,
            eps=self.eps,
            mode=self.mode,
            gamma=self.gamma,
            constant=self.constant)
        return  loss_matrix


if __name__ == '__main__':
    # 示例预测和目标张量
    # 示例预测和目标张量
    pred = torch.tensor([[0.0, 0.0, 1.0, 1.0],
                         [1.0, 1.0, 2.0, 2.0]], dtype=torch.float32)
    target = torch.tensor([[0.5, 0.5, 1.5, 1.5],
                           [1.5, 1.5, 2.5, 2.5]], dtype=torch.float32)

    # 使用 WassersteinLoss 类
    loss_fn = WassersteinLoss()
    loss_value_class = loss_fn(pred, target)
    print(f"Wasserstein Loss (Class): {loss_value_class}")