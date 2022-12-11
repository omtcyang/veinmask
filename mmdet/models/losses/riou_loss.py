import torch
from torch import nn

from ..registry import LOSSES


@LOSSES.register_module
class RIOULoss(nn.Module):
    def __init__(self):
        super(RIOULoss, self).__init__()

    def forward(self, pred, target, weight=None, avg_factor=None):
        """
         :param pred:  shape (N,36), N is nr_box
         :param target: shape (N,36)
         :return: loss
         """
        loss = torch.sum(torch.abs(pred - target), dim=1) / (torch.sum(target, dim=1) + 0.000000001)

        if weight is not None:
            loss = loss * weight
        if avg_factor is not None:
            loss = loss.sum() / avg_factor
        else:
            loss = loss.mean()
        return loss
