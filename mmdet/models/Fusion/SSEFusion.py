import torch
import torch.nn as nn

import torch.nn.functional as F

from mmdet.models.utils import ConvModule
from mmcv.cnn import constant_init, kaiming_init


class SSEFusion(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self._init_layers()

    def _init_layers(self):
        self.convL = nn.Conv2d(self.in_channels, 1, 3, 1, 1, bias=False)
        self.convT = nn.Conv2d(self.in_channels, 1, 3, 1, 1, bias=False)
        self.convR = nn.Conv2d(self.in_channels, 1, 3, 1, 1, bias=False)
        self.convB = nn.Conv2d(self.in_channels, 1, 3, 1, 1, bias=False)
        # self.fuse = nn.Conv2d(self.in_channels*2,self.in_channels,3,1,1)
        kaiming_init(self.convL)
        kaiming_init(self.convT)
        kaiming_init(self.convR)
        kaiming_init(self.convB)
        self.sigmoid = nn.Sigmoid()
        # kaiming_init(self.fuse)
        # self.activate = nn.ReLU(inplace=False)

    def normlize(self, left, top, right, bottom):
        sum = left + top + right + bottom + 0.0000001
        return left / sum, top / sum, right / sum, bottom / sum

    def forward(self, input):
        B, C, H, W = input.shape
        n_input = F.pad(input, pad=(1, 1, 1, 1))
        left_1 = n_input[:, :, 1:1 + H, :W]
        top_1 = n_input[:, :, :H, 1:1 + W]
        right_1 = n_input[:, :, 1:1 + H, 2:]
        bottom_1 = n_input[:, :, 2:, 1:1 + W]

        left_2 = self.sigmoid(self.convL(left_1))
        top_2 = self.sigmoid(self.convT(top_1))
        right_2 = self.sigmoid(self.convR(right_1))
        bottom_2 = self.sigmoid(self.convB(bottom_1))

        left_3, top_3, right_3, bottom_3 = self.normlize(left_2, top_2, right_2, bottom_2)

        sum_features = left_3 * left_1 + top_3 * top_1 + right_3 * right_1 + bottom_3 * bottom_1
        return torch.cat((input, sum_features), dim=1)
