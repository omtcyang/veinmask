#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time : 2022/9/13 10:52
@Author : HaoZhao Ma
@Email : haozhaoma@foxmail.com
@time: 2022/9/13 10:52
"""
import numpy as np
cimport numpy as np
cimport cython
from libcpp cimport bool
from libc cimport math

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double point_distance(double p1_1, double p1_2, double p2_1, double p2_2):
    cdef double temp
    temp = ((p1_1 - p2_1) * (p1_1 - p2_1) + (p1_2 - p2_2) * (p1_2 - p2_2)) ** 0.5
    return temp

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double point_phi(double start_1, double start_2, double end_1, double end_2):
    cdef double temp
    temp = math.atan2(end_2 - start_2, end_1 - start_1) * 180.0 / math.pi
    return temp

@cython.boundscheck(False)
@cython.wraparound(False)
cdef get_headcoor(
        np.ndarray[float, ndim=3] vein,
        int x,
        int y,
        list heads,
        dict inf):
    """
    以 centerpoint 为起点，计算在 thetas[curNum] 方向下与 contour 的交点坐标
    """
    cdef np.ndarray[float, ndim=1] length
    cdef np.ndarray[double, ndim=1] X, Y
    length = vein[:, y, x]
    X = np.cos(inf['thetas']) * length + x
    Y = np.sin(inf['thetas']) * length + y
    heads.extend(np.concatenate([[X], [Y]], axis=0).T.astype(np.int)[:, np.newaxis, :])

@cython.boundscheck(False)
@cython.wraparound(False)
cdef list get_indexdeeps(
        np.ndarray[long, ndim=2] start,
        np.ndarray[long, ndim=2] end,
        list centerpoint,
        dict params,
        dict inf):
    """
    获取 deep 中心点与两个heads之间的角度索引序列
    """
    cdef int line, angle_range, angle_scale
    cdef np.ndarray[double, ndim=1] thetasPhi
    cdef float  curPhi, nxtPhi, phiDelta
    cdef list idxDeeps
    cdef np.ndarray[long, ndim=1] curDelta, nxtDelta
    angle_range, angle_scale, thetasPhi = params['angle_range'], params['angle_scale'], inf['thetasPhi']
    line = params['num_directions']

    idxDeeps = []
    curDelta, nxtDelta = start[0] - centerpoint, end[0] - centerpoint
    curPhi, nxtPhi = np.arctan2(curDelta[1], curDelta[0]) * 180 / np.pi, np.arctan2(nxtDelta[1], nxtDelta[0]) * 180 / np.pi

    # 当 curPhi 到 nxtPhi 之间的顺时针夹角小于 angle_range 度，则没有必要进行deepsearch
    if nxtPhi >= curPhi:
        phiDelta = nxtPhi - curPhi
    else:
        phiDelta = (180 - curPhi) + (nxtPhi - (-180))
    if phiDelta < 360 / line * 2:
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

    if phiDelta < 720 / line:
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
    if len(idxDeeps) == line:
        return []
    # -----------------------------------------------
    return idxDeeps

@cython.boundscheck(False)
@cython.wraparound(False)
def get_deepcoors(
        np.ndarray[float, ndim=3] vein,
        list heads,
        int curNum,
        int nxtNum,
        list centerpoint,
        list deeptails,
        dict params,
        dict inf):
    """
    以 centerpoint 为起点，计算在 thetas[curNum] 和 thetas[nxtNum] 方向下与 contour 的交点坐标
    """
    cdef int num_tails
    cdef bool is_deep
    cdef list idxDeeps, deepsequence, deepheads
    cdef int curIds, nxtIds
    num_tails, is_deep = params['num_deeptails'], False
    # 求解 deepsearch 时有效的角度范围索引
    idxDeeps = get_indexdeeps(heads[curNum], heads[nxtNum], centerpoint, params, inf)
    deepsequence, deepheads = [], []
    # 当存在有效角度范围，开始进行deepsearch
    if len(idxDeeps) > 0:
        get_headcoor(vein, centerpoint[0], centerpoint[1], deepheads, inf)

        # 采样在center point在num_directions方向下与轮廓的交点坐标
        for ids in range(len(idxDeeps) - 1):
            curIds, nxtIds = idxDeeps[ids], idxDeeps[ids + 1]
            is_deeptail, deeptailcoors, _ = get_tailcoors(vein, deepheads, curIds, nxtIds, centerpoint, deeptails, num_tails, is_deep, params,
                                                          inf)
            deepsequence.append(deepheads[curIds])
            if is_deeptail and (deeptailcoors.shape[0] > 0):
                deepsequence.append(deeptailcoors)
        deepsequence.append(deepheads[idxDeeps[-1]])
    deeptails.append(deepsequence)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef bool check(int x, int y, int h, int w):
    if x < 0 or y < 0 or x >= w or y >= h:
        return False
    return True

@cython.boundscheck(False)
@cython.wraparound(False)
cdef get_tailcoors(
        np.ndarray[float, ndim=3] vein,
        list heads,
        int curNum,
        int nxtNum,
        list centerpoint,
        list deeptails,
        int num_tails,
        bool is_deep,
        dict params,
        dict inf):
    """
    以 centerpoint 为起点，计算在 thetas[curNum] 和 thetas[nxtNum] 方向下与 contour 的交点坐标
    """
    cdef int distance_threshold, min_threshold, centerpoint_x, centerpoint_y, nt, centerpoint_x0, centerpoint_y0, x_0, y_0, centerpoint_x1, centerpoint_y1, x_1, y_1
    cdef np.ndarray[float, ndim=1] thetas
    cdef float alpha, beta, vein_length_0, vein_length_1
    distance_threshold, thetas, min_threshold = params['distance_threshold'], inf['thetas'], params['min_threshold']
    if num_tails < 1 or point_distance(heads[curNum][0][0], heads[curNum][0][1], heads[nxtNum][0][0], heads[nxtNum][0][1]) < distance_threshold:
        return False, [], []
    centerpoint_x, centerpoint_y = centerpoint[0], centerpoint[1]
    vein_length, theta_cur, theta_nxt = vein[:, centerpoint_y, centerpoint_x][curNum], thetas[curNum], thetas[nxtNum]
    cdef list curcoors, nxtcoors, centers
    curcoors, nxtcoors, centers = [], [], []

    for nt in range(num_tails):
        alpha, beta = 1 / ((num_tails - nt + 1) * 2 - 1), 1 / ((num_tails - nt + 1) * 2 - 2)
        # pre
        # 更新 当前位置坐标(centerpoint_x0, centerpoint_y0)，并计算该位置下所有方向与轮廓的距离
        centerpoint_x0 = int(vein_length * alpha * math.cos(theta_cur) + centerpoint_x)
        centerpoint_y0 = int(vein_length * alpha * math.sin(theta_cur) + centerpoint_y)
        # 计算当前位置下，指定方向下与轮廓的交点坐标

        # 增加clip----------------------------
        if not check(centerpoint_x0, centerpoint_y0, vein.shape[1], vein.shape[2]):
            break
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
        # centerpoint_x1, centerpoint_y1 = clip((centerpoint_x1, centerpoint_y1), vein.shape[2] - 1, vein.shape[1] - 1)
        if not check(centerpoint_x1, centerpoint_y1, vein.shape[1], vein.shape[2]):
            break
        # 增加clip----------------------------

        vein_length_1 = vein[:, centerpoint_y1, centerpoint_x1][curNum]
        if vein_length_1 > min_threshold:
            break

        x_1 = int(vein_length_1 * math.cos(theta_cur) + centerpoint_x1)
        y_1 = int(vein_length_1 * math.sin(theta_cur) + centerpoint_y1)
        x_1, y_1 = clip((x_1, y_1), vein.shape[2] - 1, vein.shape[1] - 1)

        # 更新中心点 和 length
        centerpoint_x, centerpoint_y, vein_length = centerpoint_x1, centerpoint_y1, vein_length_1

        # ----------------------------------------------------------------------------------------------------------
        if not is_deep:
            nxtcoors.insert(0, np.array([x_0, y_0]))
            curcoors.append(np.array([x_1, y_1]))
            if point_distance(x_0, y_0, x_1, y_1) < distance_threshold \
                    or vein_length_0 < distance_threshold \
                    or vein_length_1 < distance_threshold:
                break
        else:
            nxtcoors.insert(0, np.array([x_0, y_0]))
            curcoors.append(np.array([x_1, y_1]))
            thetasPhiCur = point_phi(centerpoint[0], centerpoint[1], heads[curNum][0][0], heads[curNum][0][1])
            thetasPhiNxt = point_phi(centerpoint[0], centerpoint[1], heads[nxtNum][0][0], heads[nxtNum][0][1])
            thetasPhiDeep = point_phi(centerpoint[0], centerpoint[1], centerpoint_x, centerpoint_y)
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
            # 修改
            if len(tailcoors) == 0:
                tailcoors = np.concatenate(np.array([curcoors, nxtcoors]), 0)
            else:
                tailcoors = np.concatenate(np.array(tailcoors), 0)
        except:
            tailcoors = np.concatenate(np.array([curcoors, nxtcoors]), 0)
    else:
        tailcoors = np.concatenate(np.array([curcoors, nxtcoors]), 0)
    return True, tailcoors, centers

@cython.boundscheck(False)
@cython.wraparound(False)
def vein_generator(centerpoint, vein, params, inf):
    cdef int num_tails, num_directions, nd, curNum, nxtNum
    cdef bool is_deep, is_tail
    cdef list heads, tails, tailcs, deeptails, tailcoors, tailcenters
    num_tails, num_directions, is_deep = params['num_shoallowtails'], params['num_directions'], True
    heads, tails, tailcs, deeptails = [], [], [], []
    get_headcoor(vein, centerpoint[0], centerpoint[1], heads, inf)
    sequence = []
    for nd in range(num_directions):
        if nd == num_directions - 1:
            curNum, nxtNum = nd, 0
        else:
            curNum, nxtNum = nd, nd + 1
        is_tail, tailcoors, tailcenters = get_tailcoors(vein, heads, curNum, nxtNum, centerpoint, deeptails, num_tails, is_deep, params, inf)
        sequence.append(heads[curNum])
        if is_tail and len(tailcoors) > 0:
            sequence.append(tailcoors)
    sequence = np.concatenate(np.array(sequence), 0)[None, :, :]
    return sequence

@cython.boundscheck(False)
@cython.wraparound(False)
def clip(centerpoint, x, y):
    return np.clip(centerpoint[0], 0, x), np.clip(centerpoint[1], 0, y)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef bool judge_deep(
        double thetasPhiDeep,
        double thetasPhiCur,
        double thetasPhiNxt,
        float vein_length,
        float vein_length1,
        int distance_threshold):
    if thetasPhiCur > 0 and thetasPhiNxt < 0:
        if thetasPhiDeep >= 0:
            thetasPhiNxt += 360
        else:
            thetasPhiCur -= 360
    return not (thetasPhiDeep <= thetasPhiCur or thetasPhiDeep >= thetasPhiNxt) \
           and (vein_length > distance_threshold) \
           and (vein_length1 > distance_threshold)

@cython.boundscheck(False)
@cython.wraparound(False)
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
