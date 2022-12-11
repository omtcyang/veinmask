#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time : 2022/9/13 10:52
@Author : HaoZhao Ma
@Email : haozhaoma@foxmail.com
@time: 2022/9/13 10:52
"""
import numpy as np
import math


def point_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1]))


def point_phi(start, end):
    return np.arctan2(end[1] - start[1], end[0] - start[0]) * 180 / np.pi


def get_headcoor(vein, centerpoint, heads, inf):
    """
    以 centerpoint 为起点，计算在 thetas[curNum] 方向下与 contour 的交点坐标
    """
    cptX, cptY = centerpoint[0], centerpoint[1]
    length = vein[:, cptY, cptX]
    X = np.cos(inf['thetas']) * length + cptX
    Y = np.sin(inf['thetas']) * length + cptY
    heads.extend(np.concatenate([[X], [Y]], axis=0).T.astype(np.int)[:, np.newaxis, :])


def get_indexdeeps(start, end, centerpoint, params, inf):
    """
    获取 deep 中心点与两个heads之间的角度索引序列
    """
    angle_range, angle_scale, thetasPhi = params['angle_range'], params['angle_scale'], inf['thetasPhi']

    idxDeeps = []
    curDelta, nxtDelta = start[0] - centerpoint, end[0] - centerpoint
    curPhi, nxtPhi = np.arctan2(curDelta[1], curDelta[0]) * 180 / np.pi, np.arctan2(nxtDelta[1], nxtDelta[0]) * 180 / np.pi

    # 当 curPhi 到 nxtPhi 之间的顺时针夹角小于 angle_range 度，则没有必要进行deepsearch
    if nxtPhi >= curPhi:
        phiDelta = nxtPhi - curPhi
    else:
        phiDelta = (180 - curPhi) + (nxtPhi - (-180))
    if phiDelta < 360 / params['num_directions'] * 2:
        return idxDeeps

    # ！bug！对深度搜索的角度范围进行缩小，避免深度搜索时穿过有效角度范围导致轮廓点分散化！bug！
    curPhi += angle_scale
    nxtPhi -= angle_scale
    if curPhi >= 180:
        curPhi -= 360
    if nxtPhi <= -180:
        nxtPhi += 360
    # ！bug！对深度搜索的角度范围进行缩小，避免深度搜索时穿过有效角度范围导致轮廓点分散化！bug！
    # 后加------------------------------------------
    if nxtPhi >= curPhi:
        phiDelta = nxtPhi - curPhi
    else:
        phiDelta = (180 - curPhi) + (nxtPhi - (-180))

    if phiDelta < 720 / params['num_directions']:
        return idxDeeps
    # --------------------------------------------
    # 计算有效角度范围索引序列
    try:
        curIdx = np.where(thetasPhi > curPhi)[0][0]
    except:
        curIdx = 0
    try:
        nxtIdx = np.where(thetasPhi < nxtPhi)[0][-1]
    except:
        nxtIdx = thetasPhi.shape[0] - 1

    if curIdx <= nxtIdx:
        idxDeeps = list(np.arange(curIdx, nxtIdx + 1))
    else:
        idxDeeps = list(np.concatenate([np.arange(curIdx, thetasPhi.shape[0]), np.arange(0, nxtIdx + 1)], 0))
    # 后加----------------------------------------------
    if len(idxDeeps) == params['num_directions']:
        return []
    # -----------------------------------------------
    return idxDeeps


def get_deepcoors(vein, heads, curNum, nxtNum, centerpoint, deeptails, params, inf):
    """
    以 centerpoint 为起点，计算在 thetas[curNum] 和 thetas[nxtNum] 方向下与 contour 的交点坐标
    """
    # 增加clip----------------------------
    # centerpoint = clip(centerpoint, vein.shape[2] - 1, vein.shape[1] - 1)
    # 增加clip----------------------------
    num_tails, is_deep = params['num_deeptails'], False
    # 求解 deepsearch 时有效的角度范围索引
    idxDeeps = get_indexdeeps(heads[curNum], heads[nxtNum], centerpoint, params, inf)
    deepsequence, deepheads = [], []
    # 当存在有效角度范围，开始进行deepsearch
    if len(idxDeeps) > 0:
        get_headcoor(vein, centerpoint, deepheads, inf)  # 采样在center point在num_directions方向下与轮廓的交点坐标
        for ids in range(len(idxDeeps) - 1):
            curIds, nxtIds = idxDeeps[ids], idxDeeps[ids + 1]
            is_deeptail, deeptailcoors, _ = get_tailcoors(vein, deepheads, curIds, nxtIds, centerpoint, deeptails, num_tails, is_deep, params,
                                                          inf)
            deepsequence.append(deepheads[curIds])
            if is_deeptail and (deeptailcoors.shape[0] > 0):
                deepsequence.append(deeptailcoors)
        deepsequence.append(deepheads[idxDeeps[-1]])
    deeptails.append(deepsequence)


def get_tailcoors(vein, heads, curNum, nxtNum, centerpoint, deeptails, num_tails, is_deep, params, inf):
    """
    以 centerpoint 为起点，计算在 thetas[curNum] 和 thetas[nxtNum] 方向下与 contour 的交点坐标
    """
    distance_threshold, thetas = params['distance_threshold'], inf['thetas']
    if num_tails < 1 or point_distance(heads[curNum][0], heads[nxtNum][0]) < distance_threshold:
        return False, [], []
    centerpoint_x, centerpoint_y = centerpoint[0], centerpoint[1]
    vein_length, theta_cur, theta_nxt = vein[:, centerpoint[1], centerpoint[0]][curNum], thetas[curNum], thetas[nxtNum]
    curcoors, nxtcoors, centers = [], [], []

    for nt in range(num_tails):
        alpha, beta = 1 / ((num_tails - nt + 1) * 2 - 1), 1 / ((num_tails - nt + 1) * 2 - 2)
        # pre
        # 更新 当前位置坐标(centerpoint_x0, centerpoint_y0)，并计算该位置下所有方向与轮廓的距离
        centerpoint_x0 = int(vein_length * alpha * math.cos(theta_cur) + centerpoint_x)
        centerpoint_y0 = int(vein_length * alpha * math.sin(theta_cur) + centerpoint_y)
        # 计算当前位置下，指定方向下与轮廓的交点坐标

        # 增加clip----------------------------
        centerpoint_x0, centerpoint_y0 = clip((centerpoint_x0, centerpoint_y0), vein.shape[2] - 1, vein.shape[1] - 1)
        # 增加clip----------------------------

        vein_length_0 = vein[:, centerpoint_y0, centerpoint_x0][nxtNum]
        x_0 = int(vein_length_0 * math.cos(theta_nxt) + centerpoint_x0)
        y_0 = int(vein_length_0 * math.sin(theta_nxt) + centerpoint_y0)
        x_0, y_0 = clip((x_0, y_0), vein.shape[2] - 1, vein.shape[1] - 1)
        # nxt
        # 更新 当前位置坐标(centerpoint_x1, centerpoint_y1)，并计算该位置下所有方向与轮廓的距离
        centerpoint_x1 = int(vein_length_0 * beta * math.cos(theta_nxt) + centerpoint_x0)
        centerpoint_y1 = int(vein_length_0 * beta * math.sin(theta_nxt) + centerpoint_y0)
        # 计算当前位置下，指定方向下与轮廓的交点坐标

        # 增加clip----------------------------
        centerpoint_x1, centerpoint_y1 = clip((centerpoint_x1, centerpoint_y1), vein.shape[2] - 1, vein.shape[1] - 1)
        # 增加clip----------------------------

        vein_length_1 = vein[:, centerpoint_y1, centerpoint_x1][curNum]
        x_1 = int(vein_length_1 * math.cos(theta_cur) + centerpoint_x1)
        y_1 = int(vein_length_1 * math.sin(theta_cur) + centerpoint_y1)
        x_1, y_1 = clip((x_1, y_1), vein.shape[2] - 1, vein.shape[1] - 1)

        # 更新中心点 和 length
        centerpoint_x, centerpoint_y, vein_length = centerpoint_x1, centerpoint_y1, vein_length_1

        # ----------------------------------------------------------------------------------------------------------
        if not is_deep:
            nxtcoors.insert(0, np.array([x_0, y_0]))
            curcoors.append(np.array([x_1, y_1]))
            if point_distance([x_0, y_0], [x_1, y_1]) < distance_threshold \
                    or vein_length_0 < distance_threshold \
                    or vein_length_1 < distance_threshold:
                break
        else:
            nxtcoors.insert(0, np.array([x_0, y_0]))
            curcoors.append(np.array([x_1, y_1]))
            thetasPhiCur, thetasPhiNxt = point_phi(centerpoint, heads[curNum][0]), point_phi(centerpoint, heads[nxtNum][0])
            thetasPhiDeep = point_phi(centerpoint, [centerpoint_x, centerpoint_y])
            if thetasPhiCur == 180:
                thetasPhiCur = -180
            if thetasPhiNxt == 180:
                thetasPhiNxt = -180
            if thetasPhiDeep == 180:
                thetasPhiDeep = -180

            # ！bug！当原始中心点和深度搜索的中心点角度在一定范围内，并且深度搜索的中心点不会在目标轮廓线上时，视为有效！bug！
            if judge_deep(thetasPhiDeep, thetasPhiCur, thetasPhiNxt, vein_length_0, vein_length_1, distance_threshold):
                centers.append([centerpoint_x, centerpoint_y])
                if nt == num_tails - 1:
                    get_deepcoors(vein, heads, curNum, nxtNum, centers[-1], deeptails, params, inf)
                    break
            else:
                try:
                    get_deepcoors(vein, heads, curNum, nxtNum, centers[-1], deeptails, params, inf)
                except:
                    deeptails.append([])
                    break
    if is_deep:
        try:
            tailcoors = deeptails[-1]
			if len(tailcoors)==0:
                tailcoors = np.concatenate(np.array([curcoors, nxtcoors]), 0)
            else:
            	tailcoors = np.concatenate(np.array(tailcoors), 0)
        except:
            tailcoors = np.concatenate(np.array([curcoors, nxtcoors]), 0)
    else:
        tailcoors = np.concatenate(np.array([curcoors, nxtcoors]), 0)
    return True, tailcoors, centers


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


def vein_generator(centerpoint, vein, params, inf):
    num_tails, num_directions, is_deep = params['num_shoallowtails'], params['num_directions'], True
    heads, tails, tailcs, deeptails = [], [], [], []
    get_headcoor(vein, centerpoint, heads, inf)
    sequence = []
    for nd in range(num_directions):
        if nd == num_directions - 1:
            curNum, nxtNum = nd, 0
        else:
            curNum, nxtNum = nd, nd + 1
        is_tail, tailcoors, tailcenters = get_tailcoors(vein, heads, curNum, nxtNum, centerpoint, deeptails, num_tails, is_deep, params, inf)
        sequence.append(heads[curNum])
        if is_tail:
            sequence.append(tailcoors)
    sequence = np.concatenate(np.array(sequence), 0)[None, :, :]
    return sequence


def clip(centerpoint, x, y):
    return np.clip(centerpoint[0], 0, x), np.clip(centerpoint[1], 0, y)


def judge_deep(thetasPhiDeep, thetasPhiCur, thetasPhiNxt, vein_length, vein_length1, distance_threshold):
    if thetasPhiCur > 0 and thetasPhiNxt < 0:
        if thetasPhiDeep >= 0:
            thetasPhiNxt += 360
        else:
            thetasPhiCur -= 360
    return not (thetasPhiDeep <= thetasPhiCur or thetasPhiDeep >= thetasPhiNxt) \
           and (vein_length > distance_threshold) \
           and (vein_length1 > distance_threshold)
