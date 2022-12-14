import numpy as np
import torch
import math

from . import veinmask


def vein_sampler_preprocess(nums):
    """
    生成 预定义角度:thetas 及 预定义交点坐标:intersects
    """
    R_ = 9999.0

    intersects = []
    thetas = np.linspace(-1, 1, nums, endpoint=False) * math.pi
    for theta in thetas:
        # 先求解从中心点 center 分别射向 -180 到 +180 角度内长度为 R 的各个向量的端点坐标
        inter_x = R_ * math.cos(theta)
        inter_y = R_ * math.sin(theta)
        intersects.append([inter_x, inter_y])
    intersects = np.array(intersects)
    return thetas, intersects


def vein_generator(points, distance, upper, lower, thetas, coors):
    veinmask.veinmask(points, distance, upper, lower, thetas, coors)
