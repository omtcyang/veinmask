import torch
from torch import nn

from ..registry import LOSSES


@LOSSES.register_module
class DELoss(nn.Module):
    def __init__(self):
        super(DELoss, self).__init__()

    def forward(self, pred, target):
        """
         :param pred:  shape (N,36), N is nr_box
         :param target: shape (N,36)
         :return: loss
         """
        index = torch.ones(target.shape[0])
        target_min, _ = torch.min(target, dim=1)
        index[target_min == 1e-6] = 0

        pred = pred[index == 1]
        target = target[index == 1]
        target_min, _ = torch.min(target, dim=1)
        loss = torch.sum(torch.abs(pred - target) * torch.sqrt(target_min.unsqueeze(-1)) / (torch.sqrt(target) * target), dim=1)
        return 0.1*loss.mean()
