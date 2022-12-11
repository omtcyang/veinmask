import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from mmdet.core import distance2bbox, force_fp32, multi_apply, multiclass_nms, multiclass_nms_with_mask, vein_generator, \
    vein_sampler_preprocess
from mmdet.ops import ModulatedDeformConvPack
from ..Fusion.SSEFusion import SSEFusion
from ..builder import build_loss
from ..registry import HEADS
from ..utils import ConvModule, Scale, bias_init_with_prob, build_norm_layer
import numpy as np
import torch.nn.functional as F
import time

INF = 1e8
line = 12

thetas, intersects = vein_sampler_preprocess(line)

inf = {'thetas': thetas,
       'thetasPhi': thetas * 180 / np.pi
       }

def distance2maskUpsample(topk_inds, distances, points, stride=8, scale_factor=(2, 1.7286), distance=125):
    c_y = topk_inds // distances.shape[2]
    c_x = topk_inds % distances.shape[2]
    points = points.reshape([distances.shape[1], distances.shape[2], 2])
    points = points[c_y, c_x]
    upper = stride * 8
    if stride == 128:
        upper = 999999
    distances = torch.abs(F.interpolate(torch.Tensor(distances).unsqueeze(0), scale_factor=stride, mode='bilinear', align_corners=False))[0].cuda()
    mask = []
    distances = distances.permute((1, 2, 0))
    line = distances.shape[2]
    H, W = distances.shape[:2]
    coors = torch.full((H, W, (3 * line + 1) * 2 * line), -10000.0).cuda()
    vein_generator(torch.tensor(points).cuda(), distances, upper, int(distance), torch.FloatTensor(inf['thetas']).cuda(), coors)
    for x, y in points.astype(np.int):
        sequence = []
        temp = coors[y, x]
        sequence = temp[temp!=-10000].reshape(-1, 2).cpu().numpy()
        sequence[:, 0] = np.clip(sequence[:, 0], 0, W-1)
        sequence[:, 1] = np.clip(sequence[:, 1], 0, H-1)
        mask.append((sequence / scale_factor[:2]))
    return mask


# inf = {'thetas': thetas,
#        'thetasPhi': thetas * 180 / np.pi,
#        'intersects': intersects}

# def distance2maskUpsample(topk_inds, distances, points, stride=8, scale_factor=(2, 1.7286), distance=125):
#     c_y = topk_inds // distances.shape[2]
#     c_x = topk_inds % distances.shape[2]
#     points = points.reshape([distances.shape[1], distances.shape[2], 2])
#     points = points[c_y, c_x]
#     upper = stride * 8
#     if stride == 128:
#         upper = 999999
#     params = {'num_directions': line,
#               'num_shoallowtails': 1,
#               'num_deeptails': 1,
#               'angle_range': 180,
#               'angle_scale': 10,
#               'min_threshold': upper,
#               'distance_threshold': int(distance)}
#     distances = torch.abs(F.interpolate(torch.Tensor(distances).unsqueeze(0), scale_factor=stride, mode='bilinear', align_corners=False))[
#         0].numpy()
#     mask = []
#     for x, y in points.astype(np.int):
#         sequence = vein_generator((x, y), distances, params, inf)[0]
#         sequence[:, 0] = np.clip(sequence[:, 0], 0, distances.shape[2] - 1)
#         sequence[:, 1] = np.clip(sequence[:, 1], 0, distances.shape[1] - 1)
#         print()
#         print(sequence)
#         # 下面进行映射回原始尺寸的大小
#         mask.append((sequence / scale_factor[:2]))
#     return mask


def distance2mask(topk_inds, distances, points, stride=8, scale_factor=(2, 1.7286)):
    c_y = topk_inds // distances.shape[2]
    c_x = topk_inds % distances.shape[2]
    si = np.sin(thetas)
    co = np.cos(thetas)
    points = points.reshape([distances.shape[1], distances.shape[2], 2])
    points = points[c_y, c_x]
    dist = distances[:, c_y, c_x]
    co = (((dist.T * co).T + points[:, 0]) / scale_factor[0]).astype(np.int).T
    si = (((dist.T * si).T + points[:, 1]) / scale_factor[1]).astype(np.int).T
    mask = np.concatenate(
        [co[:, :, np.newaxis], si[:, :, np.newaxis]], axis=-1)
    return mask


@HEADS.register_module
class PolarMask_Head(nn.Module):

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 stacked_convs=4,
                 strides=(4, 8, 16, 32, 64),
                 regress_ranges=((-1, 64), (64, 128),
                                 (128, 256), (256, 512), (512, INF)),
                 use_dcn=False,
                 mask_nms=False,
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(type='IoULoss', loss_weight=1.0),
                 loss_mask=dict(type='RIOULoss'),
                 loss_sample_points=dict(type='RIOULoss'),
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 distance=125):
        super(PolarMask_Head, self).__init__()

        self.num_classes = num_classes
        self.cls_out_channels = num_classes - 1
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.regress_ranges = regress_ranges
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_mask = build_loss(loss_mask)
        self.loss_centerness = build_loss(loss_centerness)
        self.loss_sample_point = build_loss(loss_sample_points)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False
        # xez add for polarmask
        self.use_dcn = use_dcn
        self.mask_nms = mask_nms
        # debug vis img
        self.vis_num = 1000
        self.count = 0
        self.distance = distance
        # # test
        # self.angles = torch.range(0, 350, 18).cuda() / 180 * math.pi
        self._init_layers()

    def _init_layers(self):
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.mask_convs = nn.ModuleList()
        self.SSEFusion = SSEFusion(self.in_channels)
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            if not self.use_dcn:
                self.cls_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        bias=self.norm_cfg is None))
                self.reg_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        bias=self.norm_cfg is None))
                if i == 0:
                    self.mask_convs.append(
                        ConvModule(
                            chn * 2,
                            self.feat_channels,
                            3,
                            stride=1,
                            padding=1,
                            conv_cfg=self.conv_cfg,
                            norm_cfg=self.norm_cfg,
                            bias=self.norm_cfg is None))
                else:
                    self.mask_convs.append(
                        ConvModule(
                            chn,
                            self.feat_channels,
                            3,
                            stride=1,
                            padding=1,
                            conv_cfg=self.conv_cfg,
                            norm_cfg=self.norm_cfg,
                            bias=self.norm_cfg is None))
            else:
                self.cls_convs.append(
                    ModulatedDeformConvPack(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        dilation=1,
                        deformable_groups=1,
                    ))
                if self.norm_cfg:
                    self.cls_convs.append(build_norm_layer(
                        self.norm_cfg, self.feat_channels)[1])
                self.cls_convs.append(nn.ReLU(inplace=True))

                self.reg_convs.append(
                    ModulatedDeformConvPack(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        dilation=1,
                        deformable_groups=1,
                    ))
                if self.norm_cfg:
                    self.reg_convs.append(build_norm_layer(
                        self.norm_cfg, self.feat_channels)[1])
                self.reg_convs.append(nn.ReLU(inplace=True))

                self.mask_convs.append(
                    ModulatedDeformConvPack(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        dilation=1,
                        deformable_groups=1,
                    ))
                if self.norm_cfg:
                    self.mask_convs.append(build_norm_layer(
                        self.norm_cfg, self.feat_channels)[1])
                self.mask_convs.append(nn.ReLU(inplace=True))

        self.polar_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.polar_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)
        self.polar_mask = nn.Conv2d(self.feat_channels, line, 3, padding=1)
        self.polar_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.scales_bbox = nn.ModuleList([Scale(1.0) for _ in self.strides])
        self.scales_mask = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def init_weights(self):
        if not self.use_dcn:
            for m in self.cls_convs:
                normal_init(m.conv, std=0.01)
            for m in self.reg_convs:
                normal_init(m.conv, std=0.01)
            for m in self.mask_convs:
                normal_init(m.conv, std=0.01)
        else:
            pass

        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.polar_cls, std=0.01, bias=bias_cls)
        normal_init(self.polar_reg, std=0.01)
        normal_init(self.polar_mask, std=0.01)
        normal_init(self.polar_centerness, std=0.01)

    def forward(self, feats):
        return multi_apply(self.forward_single, feats, self.scales_bbox, self.scales_mask)

    def forward_single(self, x, scale_bbox, scale_mask):
        cls_feat = x
        reg_feat = x
        mask_feat = self.SSEFusion(x)

        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        cls_score = self.polar_cls(cls_feat)
        centerness = self.polar_centerness(cls_feat)

        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)
        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16
        temp = self.polar_reg(reg_feat)
        bbox_pred = scale_bbox(temp).float().exp()

        for mask_layer in self.mask_convs:
            mask_feat = mask_layer(mask_feat)
        mask_pred = scale_mask(self.polar_mask(mask_feat)).float().exp()

        return cls_score, bbox_pred, centerness, mask_pred

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'mask_preds', 'centernesses'))
    def loss(self,
             cls_scores,
             bbox_preds,
             centernesses,
             mask_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             cfg,
             gt_masks,
             gt_bboxes_ignore=None,
             extra_data=None):
        assert len(cls_scores) == len(bbox_preds) == len(
            centernesses) == len(mask_preds)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.get_points(
            featmap_sizes, bbox_preds[0].dtype, bbox_preds[0].device)

        # ground truth-------------------------------------------
        labels, sample_points, bbox_targets, mask_targets, centroidness = self.polar_target(
            all_level_points, extra_data)
        # ground truth-------------------------------------------

        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and centerness
        flatten_cls_scores = [cls_score.permute(
            0, 2, 3, 1).reshape(-1, self.cls_out_channels) for cls_score in cls_scores]
        flatten_bbox_preds = [bbox_pred.permute(
            0, 2, 3, 1).reshape(-1, 4) for bbox_pred in bbox_preds]
        flatten_centerness = [centerness.permute(
            0, 2, 3, 1).reshape(-1) for centerness in centernesses]
        flatten_mask_preds = [mask_pred.permute(
            0, 2, 3, 1).reshape(-1, line) for mask_pred in mask_preds]

        flatten_cls_scores = torch.cat(flatten_cls_scores)  # [num_pixel, 20]
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)  # [num_pixel, 4]
        flatten_centerness = torch.cat(flatten_centerness)  # [num_pixel]
        flatten_mask_preds = torch.cat(flatten_mask_preds)  # [num_pixel, 12]

        flatten_labels = torch.cat(labels).long()  # [num_pixel]
        flatten_sample_points = torch.cat(sample_points).long()
        flatten_bbox_targets = torch.cat(bbox_targets)  # [num_pixel, 4]
        flatten_mask_targets = torch.cat(mask_targets)  # [num_pixel, 12]
        flatten_centroidness_targets = torch.cat(centroidness)

        pos_inds = flatten_labels.nonzero().reshape(-1)
        sample_points_ind = flatten_sample_points.nonzero().reshape(-1)
        num_pos = len(pos_inds)

        loss_cls = self.loss_cls(flatten_cls_scores, flatten_labels,
                                 avg_factor=num_pos + num_imgs)  # avoid num_pos is 0

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]
        pos_mask_preds = flatten_mask_preds[pos_inds]
        pos_mask_sample_point_preds = flatten_mask_preds[sample_points_ind]

        if num_pos > 0:
            pos_bbox_targets = flatten_bbox_targets[pos_inds]
            pos_mask_targets = flatten_mask_targets[pos_inds]
            pos_mask_sample_point_targets = flatten_mask_targets[sample_points_ind]
            pos_centerness_targets = flatten_centroidness_targets[pos_inds]
            # pos_centerness_targets = self.polar_centerness_target(pos_mask_targets)

            flatten_points = torch.cat(
                [points.repeat(num_imgs, 1) for points in all_level_points])  # [num_pixel,2]
            pos_points = flatten_points[pos_inds]
            pos_decoded_bbox_preds = distance2bbox(pos_points, pos_bbox_preds)
            pos_decoded_target_preds = distance2bbox(
                pos_points, pos_bbox_targets)

            # centerness weighted iou loss
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                weight=pos_centerness_targets,
                avg_factor=pos_centerness_targets.sum())
            loss_mask = self.loss_mask(pos_mask_preds,
                                       pos_mask_targets,
                                       weight=pos_centerness_targets,
                                       avg_factor=pos_centerness_targets.sum())
            loss_sample_points = self.loss_sample_point(
                pos_mask_sample_point_preds, pos_mask_sample_point_targets)
            loss_centerness = self.loss_centerness(
                pos_centerness, pos_centerness_targets)
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_mask = pos_mask_preds.sum()
            loss_sample_points = pos_mask_sample_point_preds.sum()
            loss_centerness = pos_centerness.sum()

        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_mask=loss_mask,
            loss_sample_points=loss_sample_points,
            loss_centerness=loss_centerness)

    def get_points(self, featmap_sizes, dtype, device):
        """Get points according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            dtype (torch.dtype): Type of points.
            device (torch.device): Device of points.

        Returns:
            tuple: points of each image.
        """
        mlvl_points = []
        for i in range(len(featmap_sizes)):
            mlvl_points.append(
                self.get_points_single(featmap_sizes[i], self.strides[i], dtype, device))
        return mlvl_points

    def get_points_single(self, featmap_size, stride, dtype, device):
        h, w = featmap_size
        x_range = torch.arange(
            0, w * stride, stride, dtype=dtype, device=device)
        y_range = torch.arange(
            0, h * stride, stride, dtype=dtype, device=device)
        y, x = torch.meshgrid(y_range, x_range)
        points = torch.stack(
            (x.reshape(-1), y.reshape(-1)), dim=-1) + stride // 2
        return points

    def polar_target(self, points, extra_data):
        assert len(points) == len(self.regress_ranges)

        num_levels = len(points)

        labels_list, sample_points_list, bbox_targets_list, mask_targets_list, centroidness_list = extra_data.values()

        # split to per img, per level
        num_points = [center.size(0) for center in points]
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        sample_points_list = [sample_points.split(
            num_points, 0) for sample_points in sample_points_list]
        bbox_targets_list = [bbox_targets.split(
            num_points, 0) for bbox_targets in bbox_targets_list]
        mask_targets_list = [mask_targets.split(
            num_points, 0) for mask_targets in mask_targets_list]
        centroidness_list = [centroidness.split(
            num_points, 0) for centroidness in centroidness_list]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_sample_points = []
        concat_lvl_bbox_targets = []
        concat_lvl_mask_targets = []
        concat_lvl_centroidness = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            concat_lvl_sample_points.append(
                torch.cat([sample_points[i] for sample_points in sample_points_list]))
            concat_lvl_bbox_targets.append(
                torch.cat([bbox_targets[i] for bbox_targets in bbox_targets_list]))
            concat_lvl_mask_targets.append(
                torch.cat([mask_targets[i] for mask_targets in mask_targets_list]))
            concat_lvl_centroidness.append(
                torch.cat([centroidness[i] for centroidness in centroidness_list]))

        return concat_lvl_labels, concat_lvl_sample_points, concat_lvl_bbox_targets, concat_lvl_mask_targets, concat_lvl_centroidness

    # def polar_centerness_target(self, pos_mask_targets):
    #     # only calculate pos centerness targets, otherwise there may be nan
    #     centerness_targets = (pos_mask_targets.min(dim=-1)[0] / pos_mask_targets.max(dim=-1)[0])
    #     return torch.sqrt(centerness_targets)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   centernesses,
                   mask_preds,
                   img_metas,
                   cfg,
                   rescale=None):
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = self.get_points(
            featmap_sizes, bbox_preds[0].dtype, bbox_preds[0].device)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [cls_scores[i][img_id].detach()
                              for i in range(num_levels)]
            bbox_pred_list = [bbox_preds[i][img_id].detach()
                              for i in range(num_levels)]
            centerness_pred_list = [centernesses[i]
                                    [img_id].detach() for i in range(num_levels)]
            mask_pred_list = [mask_preds[i][img_id].detach()
                              for i in range(num_levels)]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            det_bboxes = self.get_bboxes_single(cls_score_list,
                                                bbox_pred_list,
                                                mask_pred_list,
                                                centerness_pred_list,
                                                mlvl_points,
                                                img_shape,
                                                scale_factor,
                                                cfg,
                                                rescale)
            result_list.append(det_bboxes)
        return result_list

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          mask_preds,
                          centernesses,
                          mlvl_points,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_masks = []
        mlvl_centerness = []
        i = 0
        for cls_score, bbox_pred, mask_pred, centerness, points in zip(cls_scores, bbox_preds, mask_preds, centernesses, mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(
                1, 2, 0).reshape(-1, self.cls_out_channels).sigmoid()
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)

            # # 下面进行拷贝工作-------------
            points_copy = points.clone().cpu().numpy()
            mask_pred_copy = np.zeros_like(mask_pred.clone().cpu().numpy())
            temp = mask_pred.cpu().numpy()[::-1]
            count = int(0.75 * line + 1)
            mask_pred_copy[:count] = temp[(line - count):]
            mask_pred_copy[count:] = temp[:(line - count)]
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores * centerness[:, None]).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                centerness = centerness[topk_inds]
                topk_inds = topk_inds.cpu().numpy()
            else:
                topk_inds = np.arange(scores.shape[0])

            # --------------------------------------
            # 测试-----------------------------------
            index = scores.max(dim=1)[0] > cfg.score_thr
            points = points[index]
            bbox_pred = bbox_pred[index]
            scores = scores[index]
            centerness = centerness[index]
            topk_inds = topk_inds[index.cpu()]
            # 测试-----------------------------------
            # --------------------------------------

            bboxes = distance2bbox(points, bbox_pred, max_shape=img_shape)
            a = [8]
            if self.strides[i] in a:
                masks = distance2mask(
                    topk_inds, mask_pred_copy, points_copy, self.strides[i], scale_factor)
            else:
                masks = distance2maskUpsample(
                    topk_inds, mask_pred_copy, points_copy, self.strides[i], scale_factor, self.distance)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)
            mlvl_masks.extend(masks)
            i += 1

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            _mlvl_bboxes = mlvl_bboxes / mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)
        mlvl_centerness = torch.cat(mlvl_centerness)

        centerness_factor = 0.5
        '''2 origin bbox->nms, performance same to mask->min_bbox'''
        det_bboxes, det_labels, det_masks = multiclass_nms_with_mask(
            _mlvl_bboxes,
            mlvl_scores,
            mlvl_masks,
            cfg.score_thr,
            cfg.nms,
            cfg.max_per_img,
            score_factors=mlvl_centerness + centerness_factor)

        return det_bboxes, det_labels, det_masks
