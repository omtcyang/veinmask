import cv2
import math
from pycocotools.coco import COCO
from .custom import CustomDataset
import os.path as osp
import warnings
import mmcv
import numpy as np
from mmcv.parallel import DataContainer as DC
import torch
from .registry import DATASETS
from .utils import random_scale, to_tensor, uniformsample, augment, transform_original_data, get_valid_polys

INF = 1e8
line = 12


@DATASETS.register_module
class Coco_Seg_Dataset(CustomDataset):
    CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic_light', 'fire_hydrant',
               'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat',
               'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket',
               'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush')

    def load_annotations(self, ann_file):
        print(ann_file)
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.getCatIds()
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }
        self.img_ids = self.coco.getImgIds()
        img_infos = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            img_infos.append(info)

        self.count = {0: 0,
                      1: 96 * 160,
                      2: 96 * 160 + 48 * 80,
                      3: 96 * 160 + 48 * 80 + 24 * 40,
                      4: 96 * 160 + 48 * 80 + 24 * 40 + 12 * 20}
        return img_infos

    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        return self._parse_ann_info(ann_info, self.with_mask)

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        for i, img_info in enumerate(self.img_infos):
            if self.img_ids[i] not in ids_with_ann:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _parse_ann_info(self, ann_info, with_mask=True):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, mask_polys, poly_lens.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        # Two formats are provided.
        # 1. mask: a binary map of the same size of the image.
        # 2. polys: each mask consists of one or several polys, each poly is a
        # list of float.

        self.debug = False

        if with_mask:
            gt_masks = []
            gt_mask_polys = []
            gt_poly_lens = []

        if self.debug:
            count = 0
            total = 0
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            # filter bbox < 10
            if self.debug:
                total += 1

            if ann['area'] <= 15 or (w < 10 and h < 10) or self.coco.annToMask(ann).sum() < 15:
                # print('filter, area:{},w:{},h:{}'.format(ann['area'],w,h))
                if self.debug:
                    count += 1
                continue

            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            if ann['iscrowd']:
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
            if with_mask:
                gt_masks.append(self.coco.annToMask(ann))
                mask_polys = [
                    p for p in ann['segmentation'] if len(p) >= 6
                ]  # valid polygons have >= 3 points (6 coordinates)
                poly_lens = [len(p) for p in mask_polys]
                gt_mask_polys.append(mask_polys)
                gt_poly_lens.extend(poly_lens)

        if self.debug:
            print('filter:', count / total)
        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        ann = dict(bboxes=gt_bboxes, labels=gt_labels, bboxes_ignore=gt_bboxes_ignore)

        if with_mask:
            ann['masks'] = gt_masks
            # poly format is not used in the current implementation
            ann['mask_polys'] = gt_mask_polys
            ann['poly_lens'] = gt_poly_lens
        return ann

    def prepare_train_img(self, idx):
        img_info = self.img_infos[idx]
        img = mmcv.imread(osp.join(self.img_prefix, img_info['filename']))

        ann = self.get_ann_info(idx)
        # gt_bboxes = ann['bboxes']
        gt_labels = ann['labels']

        # skip the image if there is no valid gt bbox
        if len(gt_labels) == 0 and self.skip_img_without_anno:
            warnings.warn('Skip the image "%s" that has no valid gt bbox' % osp.join(self.img_prefix, img_info['filename']))
            return None

        width, height = img.shape[1], img.shape[0]
        inp, trans_input, flip, center, scale = augment(img, self.img_scales[0][0], self.img_scales[0][1], [0.6, 1.4])

        instance_polys = transform_original_data(ann['masks'], flip, width, trans_input, [self.img_scales[0][0], self.img_scales[0][1]])
        instance_polys = get_valid_polys(instance_polys, [self.img_scales[0][0], self.img_scales[0][1]])

        gt_labels_ = []
        instance_polys_ = []
        gt_bboxes_ = []
        for i, label in enumerate(gt_labels):
            for instance in instance_polys[i]:
                gt_labels_.append(label)
                instance_polys_.append(instance)
                minx = np.min(instance[:, 0])
                miny = np.min(instance[:, 1])
                maxx = np.max(instance[:, 0])
                maxy = np.max(instance[:, 1])
                gt_bboxes_.append([minx, miny, maxx, maxy])
        gt_labels = np.array(gt_labels_)
        gt_bboxes = np.array(gt_bboxes_)
        instance_polys = instance_polys_

        if len(gt_bboxes) == 0 and self.skip_img_without_anno:
            # warnings.warn('Skip the image "%s" that has no valid gt bbox' % osp.join(self.img_prefix, img_info['filename']))
            return None

        img, img_shape, pad_shape, scale_factor = self.img_transform(inp,
                                                                     [self.img_scales[0][0], self.img_scales[0][1]],
                                                                     False,
                                                                     keep_ratio=self.resize_keep_ratio)
        img = img.copy()
        gt_masks = np.zeros((len(instance_polys), self.img_scales[0][0], self.img_scales[0][1]))
        for i in range(len(instance_polys)):
            instance_poly = instance_polys[i]
            cv2.drawContours(gt_masks[i], [instance_poly[:, np.newaxis, :].astype(np.int)], -1, 1, -1)
        ori_shape = (img_info['height'], img_info['width'], 3)
        img_meta = dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=flip)

        data = dict(
            img=DC(to_tensor(img), stack=True),
            img_meta=DC(img_meta, cpu_only=True),
            gt_bboxes=DC(to_tensor(gt_bboxes)))
        if self.with_label:
            data['gt_labels'] = DC(to_tensor(gt_labels))
        if self.with_mask:
            data['gt_masks'] = DC(gt_masks, cpu_only=True)

        # --------------------offline ray label generation-----------------------------
        self.center_sample = True
        self.use_mask_center = True
        self.radius = 1.5
        self.strides = [8, 16, 32, 64, 128]
        self.regress_ranges = ((-1, 64), (64, 128), (128, 256), (256, 512), (512, INF))
        featmap_sizes = self.get_featmap_size(pad_shape)
        self.featmap_sizes = featmap_sizes
        num_levels = len(self.strides)
        all_level_points = self.get_points(featmap_sizes)
        self.all_level_points = all_level_points
        self.num_points_per_level = [i.size()[0] for i in all_level_points]

        expanded_regress_ranges = \
            [all_level_points[i].new_tensor(self.regress_ranges[i])[None].expand_as(all_level_points[i]) for i in range(num_levels)]
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(all_level_points, 0)
        gt_masks = gt_masks[:len(gt_bboxes)]

        gt_bboxes = torch.Tensor(gt_bboxes)
        gt_labels = torch.Tensor(gt_labels)

        _labels, _sample_point, _bbox_targets, _mask_targets, centroidness = self.polar_target_single(gt_bboxes,
                                                                                                      gt_masks,
                                                                                                      gt_labels,
                                                                                                      concat_points,
                                                                                                      concat_regress_ranges)

        data['_gt_labels'] = DC(_labels)
        data['_sample_point'] = DC(_sample_point)
        data['_gt_bboxes'] = DC(_bbox_targets)
        data['_gt_masks'] = DC(_mask_targets)
        data['centroidness'] = DC(centroidness)
        # --------------------offline ray label generation-----------------------------
        return data

    def get_featmap_size(self, shape):
        h, w = shape[:2]
        featmap_sizes = []
        for i in self.strides:
            featmap_sizes.append([int(h / i), int(w / i)])
        return featmap_sizes

    def get_points(self, featmap_sizes):
        mlvl_points = []
        for i in range(len(featmap_sizes)):
            mlvl_points.append(self.get_points_single(featmap_sizes[i], self.strides[i]))
        return mlvl_points

    def get_points_single(self, featmap_size, stride):
        h, w = featmap_size
        x_range = torch.arange(0, w * stride, stride)
        y_range = torch.arange(0, h * stride, stride)
        y, x = torch.meshgrid(y_range, x_range)
        points = torch.stack((x.reshape(-1), y.reshape(-1)), dim=-1) + stride // 2
        return points.float()

    def polar_target_single(self, gt_bboxes, masks, gt_labels, points, regress_ranges):
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        if num_gts == 0:
            return gt_labels.new_zeros(num_points), gt_bboxes.new_zeros((num_points, 4))

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0] + 1) * (gt_bboxes[:, 3] - gt_bboxes[:, 1] + 1)
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        # xs ys 分别是points的x y坐标
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)
        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)  # feature map上所有点对于gtbox的上下左右距离 [num_pix, num_gt, 4]

        # mask targets 也按照这种写 同时labels 得从bbox中心修改成mask 重心
        mask_centers = []
        mask_contours = []
        # mask_counts = []
        # 第一步 先算重心  return [num_gt, 2]
        for mask in masks:
            cnt, contour = self.get_single_centerpoint(mask)
            contour = torch.Tensor(contour).float()
            y, x = cnt
            mask_centers.append([x, y])
            mask_contours.append(contour)
            # mask_counts.append(torch.Tensor(count).float())
        mask_centers = torch.Tensor(mask_centers).float()
        # 把mask_centers assign到不同的层上,根据regress_range和重心的位置
        mask_centers = mask_centers[None].expand(num_points, num_gts, 2)

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------
        # condition1: inside a gt bbox
        # 加入center sample
        # if self.center_sample:
        strides = [8, 16, 32, 64, 128]
        # if self.use_mask_center:
        inside_gt_bbox_mask = self.get_mask_sample_region(gt_bboxes,
                                                          mask_centers,
                                                          strides,
                                                          self.num_points_per_level,
                                                          xs,
                                                          ys,
                                                          radius=self.radius)

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (max_regress_distance >= regress_ranges[..., 0]) & (max_regress_distance <= regress_ranges[..., 1])

        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = 0  # [num_gt] 介于0-20

        bbox_targets = bbox_targets[range(num_points), min_area_inds]
        pos_inds = labels.nonzero().reshape(-1)

        mask_targets = torch.zeros(num_points, line).float()
        pos_mask_ids = min_area_inds[pos_inds]
        # 下面计算centroidness----------------------------------------------------
        mask2point_dict = {}
        for index, maskId in enumerate(pos_mask_ids):
            if int(maskId) not in mask2point_dict.keys():
                mask2point_dict[int(maskId)] = []
            mask2point_dict[int(maskId)].append(index)

        centroidness = torch.zeros_like(labels)

        for maskId in mask2point_dict.keys():
            index = np.array(mask2point_dict[maskId])
            index = pos_inds[index]
            temp_points = points[index].permute(1, 0).unsqueeze(0)
            temp_contour = np.repeat(mask_contours[maskId].unsqueeze(-1), len(index), axis=-1)
            temp_points = np.repeat(temp_points, temp_contour.shape[0], axis=0)
            power = np.power(temp_points - temp_contour, 2)
            dist = np.sqrt(power[:, 0, :] + power[:, 1, :]).T
            min_dist, _ = torch.min(dist, dim=-1)

            temp_center = np.repeat(mask_centers[0, maskId].reshape(2, 1).flip(dims=(0, 1)).unsqueeze(0),
                                    len(index),
                                    axis=0)
            power = np.power(points[index].unsqueeze(-1) - temp_center, 2)
            center_dist = np.sqrt(power[:, 0, :] + power[:, 1, :])[:, 0]
            centroidness[index] = min_dist / (min_dist + center_dist + 0.00000001)
            # 验证归一化（省略）----------------------------------

        # -----------------------------------------------------------------------------

        sample_point = torch.zeros_like(labels)
        for p, id in zip(pos_inds, pos_mask_ids):
            x, y = points[p]
            pos_mask_contour = mask_contours[id]
            # bug, 采样中心点仅在bbox内，但不在mask之内，进行重新标定
            area_ci = np.sum(masks[id])
            length_ci = cv2.arcLength(pos_mask_contour.numpy(), True)
            d = area_ci * (1 - 0.8 * 0.8) / (length_ci + 0.000001)
            temple1 = np.ones(masks[id].shape, np.uint8)
            temple2 = np.zeros(masks[id].shape, np.uint8)
            cv2.drawContours(temple2, [pos_mask_contour.numpy()[:, np.newaxis, :].astype(np.int)], 0, 1, int(d))
            temple2 = (temple1 - temple2) * masks[id]
            if temple2[int(y), int(x)] == 0:
                labels[p] = 0
                continue

            dists, coords = self.get_num_coordinates(x, y, pos_mask_contour, line)
            mask_targets[p] = dists
            self.random_sample_within_mask(p, x, y, pos_mask_contour, mask_targets, strides, sample_point, temple2)
        return labels, sample_point, bbox_targets, mask_targets, centroidness

    def get_mask_sample_region(self, gt_bb, mask_center, strides, num_points_per, gt_xs, gt_ys, radius=1):
        center_y = mask_center[..., 0]
        center_x = mask_center[..., 1]
        center_gt = gt_bb.new_zeros(gt_bb.shape)
        # no gt
        if center_x[..., 0].sum() == 0:
            return gt_xs.new_zeros(gt_xs.shape, dtype=torch.uint8)
        beg = 0
        for level, n_p in enumerate(num_points_per):
            end = beg + n_p
            stride = strides[level] * radius
            xmin = center_x[beg:end] - stride
            ymin = center_y[beg:end] - stride
            xmax = center_x[beg:end] + stride
            ymax = center_y[beg:end] + stride
            # limit sample region in gt
            center_gt[beg:end, :, 0] = torch.where(xmin > gt_bb[beg:end, :, 0], xmin, gt_bb[beg:end, :, 0])
            center_gt[beg:end, :, 1] = torch.where(ymin > gt_bb[beg:end, :, 1], ymin, gt_bb[beg:end, :, 1])
            center_gt[beg:end, :, 2] = torch.where(xmax > gt_bb[beg:end, :, 2], gt_bb[beg:end, :, 2], xmax)
            center_gt[beg:end, :, 3] = torch.where(ymax > gt_bb[beg:end, :, 3], gt_bb[beg:end, :, 3], ymax)
            beg = end

        left = gt_xs - center_gt[..., 0]
        right = center_gt[..., 2] - gt_xs
        top = gt_ys - center_gt[..., 1]
        bottom = center_gt[..., 3] - gt_ys
        center_bbox = torch.stack((left, top, right, bottom), -1)
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0  # 上下左右都>0 就是在bbox里面
        return inside_gt_bbox_mask

    def get_centerpoint(self, lis):
        area = 0.0
        x, y = 0.0, 0.0
        a = len(lis)
        for i in range(a):
            lat = lis[i][0]
            lng = lis[i][1]
            if i == 0:
                lat1 = lis[-1][0]
                lng1 = lis[-1][1]
            else:
                lat1 = lis[i - 1][0]
                lng1 = lis[i - 1][1]
            fg = (lat * lng1 - lng * lat1) / 2.0
            area += fg
            x += fg * (lat + lat1) / 3.0
            y += fg * (lng + lng1) / 3.0
        x = x / (area + 0.00001)
        y = y / (area + 0.00001)
        return [int(x), int(y)]

    def get_center(self, instance):
        """
        获取 instance 中心点坐标
        """
        points = np.array(np.where(instance == 1)).transpose((1, 0))[:, ::-1]
        centerpoint = []
        xmin, xmax, ymin, ymax = np.min(points[:, 0]), np.max(points[:, 0]), np.min(points[:, 1]), np.max(points[:, 1])
        dx, dy = xmax - xmin, ymax - ymin
        if dx >= dy:
            x = int(xmin + dx / 2)
            y_pts = points[points[:, 0] == x][:, 1]
            if y_pts.shape[0] > 0:
                y_pts = np.sort(y_pts)
                y = y_pts[len(y_pts) // 2]
                centerpoint = [int(x), int(y)]
        else:
            y = int(ymin + dy / 2)
            x_pts = points[points[:, 1] == y][:, 0]
            if x_pts.shape[0] > 0:
                x_pts = np.sort(x_pts)
                x = x_pts[len(x_pts) // 2]
                centerpoint = [int(x), int(y)]
        return centerpoint

    def get_single_centerpoint(self, mask):
        contour, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        count = contour[0][:, 0, :]
        try:
            center = self.get_centerpoint(count)
        except:
            x, y = count.mean(axis=0)
            center = [int(x), int(y)]
        flag = 0
        if not (0 <= center[1] < mask.shape[0] and 0 <= center[0] < mask.shape[1]):
            center = self.get_center(mask)
            flag = 1
        if flag == 0 and mask[center[1], center[0]] == 0:
            center = self.get_center(mask)
        return center, uniformsample(count, len(count) * 10)

    def get_num_coordinates(self, c_x, c_y, pos_mask_contour, num):
        ct = pos_mask_contour
        x = ct[:, 0] - c_x
        y = ct[:, 1] - c_y
        angle = torch.atan2(x, y) * 180 / np.pi
        angle[angle < 0] += 360
        angle = angle.int()
        dist = torch.sqrt(x ** 2 + y ** 2)
        angle, idx = torch.sort(angle)
        dist = dist[idx]
        new_coordinate = {}
        for i in np.linspace(0, 360, num, endpoint=False):
            i = int(i)
            if i in angle:
                d = dist[angle == i].min()
                new_coordinate[i] = d
            elif i + 1 in angle:
                d = dist[angle == i + 1].min()
                new_coordinate[i] = d
            elif i - 1 in angle:
                d = dist[angle == i - 1].min()
                new_coordinate[i] = d
            elif i + 2 in angle:
                d = dist[angle == i + 2].min()
                new_coordinate[i] = d
            elif i - 2 in angle:
                d = dist[angle == i - 2].min()
                new_coordinate[i] = d
            elif i + 3 in angle:
                d = dist[angle == i + 3].min()
                new_coordinate[i] = d
            elif i - 3 in angle:
                d = dist[angle == i - 3].min()
                new_coordinate[i] = d
        distances = torch.zeros(num)
        for i, a in enumerate(np.linspace(0, 360, num, endpoint=False)):
            a = int(a)
            if not a in new_coordinate.keys():
                new_coordinate[a] = torch.tensor(1e-6)
                distances[i] = 1e-6
            else:
                distances[i] = new_coordinate[a]
        return distances, new_coordinate

    def random_sample_within_mask(self, pid, x, y, pos_mask_contour, mask_targets, strides, sample_point_labels, temple):
        ceil = self.get_ceiling(pid)
        dist, _ = self.get_num_coordinates(x, y, pos_mask_contour, 2 * line)
        sample_dist = np.array([dist.numpy() / 10 * i for i in range(1, 10)])
        stride = strides[ceil]
        angles = np.linspace(0, 2, 2 * line, endpoint=False) * math.pi
        x = (sample_dist * np.sin(angles) + x.numpy()) / stride
        y = (sample_dist * np.cos(angles) + y.numpy()) / stride
        sample_point = np.concatenate([[x.reshape(-1)], [y.reshape(-1)]], axis=0).astype(np.int).T
        sample_point = np.array(list(set([tuple(t) for t in sample_point])))
        sample_point[:, 0] = np.clip(sample_point[:, 0], 0, self.featmap_sizes[ceil][1] - 1)
        sample_point[:, 1] = np.clip(sample_point[:, 1], 0, self.featmap_sizes[ceil][0] - 1)
        for s_point in sample_point:
            x, y = s_point
            index = y * self.featmap_sizes[ceil][1] + x
            x, y = self.all_level_points[ceil][index]
            if temple[int(y), int(x)] == 0:
                continue
            sample_point_labels[self.count[ceil] + index] = 1
            temp_dist, _ = self.get_num_coordinates(x, y, pos_mask_contour, line)
            mask_targets[self.count[ceil] + index] = temp_dist

    def get_ceiling(self, pid):
        if pid < 15360:
            return 0
        elif pid < 19200:
            return 1
        elif pid < 20160:
            return 2
        elif pid < 20400:
            return 3
        return 4

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data
