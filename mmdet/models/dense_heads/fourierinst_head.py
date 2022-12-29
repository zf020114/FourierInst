# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, Scale, bias_init_with_prob, normal_init
from mmcv.ops import deform_conv2d
from mmcv.runner import force_fp32

from mmdet.core import (anchor_inside_flags, build_assigner, distance2bbox,
                        images_to_levels, multi_apply, reduce_mean, unmap)
from mmdet.core.utils import filter_scores_and_topk,select_single_mlvl
from mmdet.models.utils import sigmoid_geometric_mean
from ..builder import HEADS, build_loss
from .atss_head import ATSSHead
from mmdet.core.bbox.transforms_rotate import distance2rbox, \
                poly2obb_np_mask,rotated_box_to_poly,rbbox2result,\
                    rotated_box_to_bbox,distance2fourier,fourier_to_bbox,fourier_to_ploy
from mmcv.ops import batched_nms

@HEADS.register_module()
class FourierInstHead(ATSSHead):
    """TOODHead used in `TOOD: Task-aligned One-stage Object Detection.

    <https://arxiv.org/abs/2108.07755>`_.

    TOOD uses Task-aligned head (T-head) and is optimized by Task Alignment
    Learning (TAL).

    Args:
        num_dcn (int): Number of deformable convolution in the head.
            Default: 0.
        anchor_type (str): If set to `anchor_free`, the head will use centers
            to regress bboxes. If set to `anchor_based`, the head will
            regress bboxes based on anchors. Default: `anchor_free`.
        initial_loss_cls (dict): Config of initial loss.

    Example:
        >>> self = TOODHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_score, bbox_pred = self.forward(feats)
        >>> assert len(cls_score) == len(self.scales)
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 num_dcn=0,
                 anchor_type='anchor_free',
                 fourier_degree=10,
                 sample_num = 50,
                 point_loss = False,
                 reg_format='mask',
                 show_result = False,
                 alpha=0.2,
                 initial_loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     activated=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                loss_mask=dict(type='L1Loss', loss_weight=1.0),
                 **kwargs):
        assert anchor_type in ['anchor_free', 'anchor_based']
        self.num_dcn = num_dcn
        self.anchor_type = anchor_type
        self.epoch = 0  # which would be update in SetEpochInfoHook!
        #TODO insert
        self.fourier_degree = fourier_degree
        self.reg_format = reg_format
        self.all_level_points =None
        if self.reg_format=='mask':
            self.reg_channel =  (2*self.fourier_degree+1)*2
        elif self.reg_format=='bbox':
            self.reg_channel =  4
        elif self.reg_format=='bbox+mask':
            self.reg_channel =  (2*self.fourier_degree+1)*2+4
        else:
            raise NotImplementedError(
                    'Unknown reg_format  type.' )
        self.tar_channel = 5 + (2*self.fourier_degree+1)*2
        self.sample_num = sample_num
        self. point_loss = point_loss
        self.show_result = show_result
        self.loss_alpha = alpha
        
        super(FourierInstHead, self).__init__(num_classes, in_channels, **kwargs)

        if self.train_cfg:
            self.initial_epoch = self.train_cfg.initial_epoch
            self.initial_assigner = build_assigner(
                self.train_cfg.initial_assigner)
            self.initial_loss_cls = build_loss(initial_loss_cls)
            self.loss_mask = build_loss(loss_mask) 
            self.assigner = self.initial_assigner
            self.alignment_assigner = build_assigner(self.train_cfg.assigner)
            self.alpha = self.train_cfg.alpha
            self.beta = self.train_cfg.beta

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.inter_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            if i < self.num_dcn:
                conv_cfg = dict(type='DCNv2', deform_groups=4)
            else:
                conv_cfg = self.conv_cfg
            chn = self.in_channels if i == 0 else self.feat_channels
            self.inter_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg))

        self.cls_decomp = TaskDecomposition(self.feat_channels,
                                            self.stacked_convs,
                                            self.stacked_convs * 8,
                                            self.conv_cfg, self.norm_cfg)
        self.reg_decomp = TaskDecomposition(self.feat_channels,
                                            self.stacked_convs,
                                            self.stacked_convs * 8,
                                            self.conv_cfg, self.norm_cfg)

        self.tood_cls = nn.Conv2d(
            self.feat_channels,
            self.num_base_priors * self.cls_out_channels,
            3,
            padding=1)
        # self.tood_reg = nn.Conv2d(
        #     self.feat_channels, self.num_base_priors * self.reg_channel, 3, padding=1)
        # self.reg_offset_module = nn.Sequential(
        #     nn.Conv2d(self.feat_channels * self.stacked_convs,
        #               self.feat_channels // 4, 1), nn.ReLU(inplace=True),
        #     nn.Conv2d(self.feat_channels // 4,  self.reg_channel * 2, 3, padding=1))
        if self.reg_format=='bbox+mask':
                self.tood_reg = nn.Conv2d(
                    self.feat_channels, self.num_base_priors * 4, 3, padding=1)#4
                self.tood_mask = nn.Conv2d(
                        self.feat_channels, self.num_base_priors * (self.reg_channel-4), 3, padding=1)#4
                self.reg_offset_module = nn.Sequential(
                    nn.Conv2d(self.feat_channels * self.stacked_convs,
                            self.feat_channels // 4, 1), nn.ReLU(inplace=True),
                    nn.Conv2d(self.feat_channels // 4,  4 * 2, 3, padding=1))
                self.mask_offset_module = nn.Sequential(
                            nn.Conv2d(self.feat_channels * self.stacked_convs,
                                    self.feat_channels // 4, 1), nn.ReLU(inplace=True),
                            nn.Conv2d(self.feat_channels // 4, (self.reg_channel-4) * 2, 3, padding=1))
        else:
            self.tood_reg = nn.Conv2d(
                self.feat_channels, self.num_base_priors * self.reg_channel, 3, padding=1)#4
            self.reg_offset_module = nn.Sequential(
                nn.Conv2d(self.feat_channels * self.stacked_convs,
                        self.feat_channels // 4, 1), nn.ReLU(inplace=True),
                nn.Conv2d(self.feat_channels // 4,  self.reg_channel * 2, 3, padding=1))
            
        self.cls_prob_module = nn.Sequential(
            nn.Conv2d(self.feat_channels * self.stacked_convs,
                      self.feat_channels // 4, 1), nn.ReLU(inplace=True),
            nn.Conv2d(self.feat_channels // 4, 1, 3, padding=1))


        self.scales = nn.ModuleList(
            [Scale(1.0) for _ in self.prior_generator.strides])

    def init_weights(self):
        """Initialize weights of the head."""
        bias_cls = bias_init_with_prob(0.01)
        for m in self.inter_convs:
            normal_init(m.conv, std=0.01)
        for m in self.cls_prob_module:
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        for m in self.reg_offset_module:
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)
        normal_init(self.cls_prob_module[-1], std=0.01, bias=bias_cls)

        self.cls_decomp.init_weights()
        self.reg_decomp.init_weights()

        normal_init(self.tood_cls, std=0.01, bias=bias_cls)
        normal_init(self.tood_reg, std=0.01)
        if self.reg_format=='bbox+mask':
            normal_init(self.tood_mask, std=0.01)
            for m in self.mask_offset_module:
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.001)
                    
    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_masks=None,
                      gt_semantic_seg=None,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        outs = self(x)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes,  img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_masks, gt_semantic_seg, gt_labels,img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(
                *outs, img_metas=img_metas, cfg=proposal_cfg)
            return losses, proposal_list
        
    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually a tuple of classification scores and bbox prediction
                cls_scores (list[Tensor]): Classification scores for all scale
                    levels, each is a 4D-tensor, the channels number is
                    num_anchors * num_classes.
                bbox_preds (list[Tensor]): Decoded box for all scale levels,
                    each is a 4D-tensor, the channels number is
                    num_anchors * 4. In [tl_x, tl_y, br_x, br_y] format.
        """
        cls_scores = []
        bbox_preds = []
        for idx, (x, scale, stride) in enumerate(
                zip(feats, self.scales, self.prior_generator.strides)):
            b, c, h, w = x.shape
            anchor = self.prior_generator.single_level_grid_priors(
                (h, w), idx, device=x.device)
            anchor = torch.cat([anchor for _ in range(b)])
            # extract task interactive features
            inter_feats = []
            for inter_conv in self.inter_convs:
                x = inter_conv(x)
                inter_feats.append(x)
            feat = torch.cat(inter_feats, 1)

            # task decomposition
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_feat = self.cls_decomp(feat, avg_feat)
            reg_feat = self.reg_decomp(feat, avg_feat)

            # cls prediction and alignment
            cls_logits = self.tood_cls(cls_feat)
            cls_prob = self.cls_prob_module(feat)
            cls_score = sigmoid_geometric_mean(cls_logits, cls_prob)

            # reg prediction and alignment
            if self.reg_format=='bbox':
                reg_dist = scale(self.tood_reg(reg_feat).exp()).float()*stride[0]
                reg_offset = self.reg_offset_module(feat)

                reg_dist = self.deform_sampling(reg_dist.contiguous(),
                                                reg_offset.contiguous())
                reg_dist = reg_dist.permute(0, 2, 3, 1).reshape(-1, self.reg_channel)
                bbox_pred = distance2bbox(
                            self.anchor_center(anchor) ,#/ stride[0],
                            reg_dist).reshape(b, h, w, 4).permute(0, 3, 1, 2)  # (b, c, h, w)
            elif self.reg_format=='mask':
                reg_dist = scale(self.tood_reg(reg_feat)).float()*stride[0]
                reg_offset = self.reg_offset_module(feat)

                reg_dist = self.deform_sampling(reg_dist.contiguous(),
                                                reg_offset.contiguous())
                reg_dist = reg_dist.permute(0, 2, 3, 1).reshape(-1, self.reg_channel)
                bbox_pred =  distance2fourier( self.anchor_center(anchor) ,#/ stride[0]
                                                                reg_dist,self.fourier_degree).reshape(b, h, w, self.reg_channel).permute(0, 3, 1, 2)  # (b, c, h, w)
            elif self.reg_format=='bbox+mask':
                reg_dist = scale(self.tood_reg(reg_feat).exp()).float()*stride[0]
                reg_offset = self.reg_offset_module(feat)
                reg_dist = self.deform_sampling(reg_dist.contiguous(),
                                                reg_offset.contiguous())
                reg_dist = reg_dist.permute(0, 2, 3, 1).reshape(-1,4)
                bbox_pred = distance2bbox(
                            self.anchor_center(anchor) ,#/ stride[0],
                            reg_dist).reshape(b, h, w, 4).permute(0, 3, 1, 2)  # (b, c, h, w)
                
                reg_mask = scale(self.tood_mask(reg_feat)).float()*stride[0]
                mask_offset = self.mask_offset_module(feat)
                reg_mask = self.deform_sampling(reg_mask.contiguous(),
                                            mask_offset.contiguous())
                reg_mask = reg_mask.permute(0, 2, 3, 1).reshape(-1, self.reg_channel-4)
                reg_mask =  distance2fourier( self.anchor_center(anchor) ,#/ stride[0]
                                                                    reg_mask,self.fourier_degree).reshape(b, h, w, self.reg_channel-4).permute(0, 3, 1, 2)  # (b, c, h, w)
                bbox_pred = torch.cat((bbox_pred,reg_mask),dim=1)
                
            else:
                raise NotImplementedError(
                    f'Unknown anchor type: {self.anchor_type}.'
                    f'Please use `anchor_free` or `anchor_based`.')
           
            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred)
        return tuple(cls_scores), tuple(bbox_preds)


    def get_poly_gt_reg(self, gt_pts, anchor_pts):
        gt_reg = gt_pts.new_zeros([gt_pts.size(0), gt_pts.size(1)*2])
        anchor_pts_repeat = anchor_pts[:, :2].repeat(1, self.sample_num)
        offset_reg = gt_pts - anchor_pts_repeat
        br_reg = offset_reg >= 0 
        tl_reg = offset_reg < 0
        tlbr_inds = torch.stack([tl_reg, br_reg], -1).reshape(-1, gt_pts.size(1)*2)
        gt_reg[tlbr_inds] = torch.abs(offset_reg.reshape(-1))
        gt_reg[~tlbr_inds] = self.loss_alpha*torch.abs(offset_reg.reshape(-1))

        xl_reg = gt_reg[..., 0::4]
        xr_reg = gt_reg[..., 1::4]
        yt_reg = gt_reg[..., 2::4]
        yb_reg = gt_reg[..., 3::4]
        yx_gt_reg = torch.stack([yt_reg, yb_reg, xl_reg, xr_reg], -1).reshape(-1, gt_pts.size(1)*2)

        xl_inds = tlbr_inds[..., 0::4]
        xr_inds = tlbr_inds[..., 1::4]
        yt_inds = tlbr_inds[..., 2::4]
        yb_inds = tlbr_inds[..., 3::4]
        yx_inds = torch.stack([yt_inds, yb_inds, xl_inds, xr_inds], -1).reshape(-1, gt_pts.size(1)*2)

        return yx_gt_reg, yx_inds


    def loss_single(self, anchors, cls_score, bbox_pred, labels, label_weights,
                    bbox_targets, mask_targets, alignment_metrics, stride):#TODO insert mask_targets
        """Compute loss of a single scale level.

        Args:
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Decoded bboxes for each scale
                level with shape (N, num_anchors * 4, H, W).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors).
            bbox_targets (Tensor): BBox regression targets of each anchor with
                shape (N, num_total_anchors, 4).
            alignment_metrics (Tensor): Alignment metrics with shape
                (N, num_total_anchors).
            stride (tuple[int]): Downsample stride of the feature map.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert stride[0] == stride[1], 'h stride is not equal to w stride!'
        anchors = anchors.reshape(-1, 4)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(
            -1, self.cls_out_channels).contiguous()
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, self.reg_channel)#TODO change 4
        bbox_targets = bbox_targets.reshape(-1, 4)
        mask_targets = mask_targets.reshape(-1, self.tar_channel) #TODO insert
        labels = labels.reshape(-1)
        alignment_metrics = alignment_metrics.reshape(-1)
        label_weights = label_weights.reshape(-1)
        targets = labels if self.epoch < self.initial_epoch else (
            labels, alignment_metrics)
        cls_loss_func = self.initial_loss_cls \
            if self.epoch < self.initial_epoch else self.loss_cls

        loss_cls = cls_loss_func(
            cls_score, targets, label_weights, avg_factor=1.0)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().squeeze(1)

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]
            pos_anchors = anchors[pos_inds]

            #TODO insert
            pos_mask_targets = mask_targets[pos_inds][:,5:]
            # pos_rbox_targets = mask_targets[pos_inds][:,:5]
            
            # regression loss
            pos_bbox_weight = self.centerness_target(
                pos_anchors, pos_bbox_targets
            ) if self.epoch < self.initial_epoch else alignment_metrics[
                pos_inds]

            if  self.reg_format=='bbox':
                loss_mask=0
                loss_bbox = self.loss_bbox(
                    pos_bbox_pred,
                    pos_bbox_targets,
                    weight=pos_bbox_weight,
                    avg_factor=1.0)
            elif self.reg_format=='mask':
                # pos_decode_mask_pred = pos_bbox_pred
                # pos_mask_weight = pos_bbox_weight[:,None].repeat((1,self.sample_num*2))
                pos_mask_weight = pos_bbox_weight[:,None].repeat((1,self.sample_num*4))
                pos_decode_mask_pred = fourier_to_ploy(pos_bbox_pred,\
                                                            self.fourier_degree,self.sample_num)
                a=pos_decode_mask_pred
                pos_decode_bbox_pred = torch.stack([a[:, :,0].min(1)[0],a[:, :,1].min(1)[0],a[:,:, 0].max(1)[0],a[:, :,1].max(1)[0]],-1)
                pos_decode_mask_pred =pos_decode_mask_pred.reshape(-1,2*self.sample_num)
                pos_decode_mask_targets = fourier_to_ploy(pos_mask_targets,\
                                                            self.fourier_degree,self.sample_num).reshape(-1,2*self.sample_num)
              
                pos_anchors_center = self.anchor_center(pos_anchors)
                poly_gt_reg, poly_yx_inds = self.get_poly_gt_reg(pos_decode_mask_targets,
                                                                        pos_anchors_center)
                poly_pred_reg, _ = self.get_poly_gt_reg(pos_decode_mask_pred,
                                                                        pos_anchors_center)
                
                # normalize_term = 4* stride[0]
                loss_mask = self.loss_mask(poly_pred_reg ,
                                                    poly_gt_reg ,
                                                    pos_mask_weight,
                                                    avg_factor = 1.0,
                                                    anchor_pts = pos_anchors[:,:-1] ,
                                                    bbox_gt = pos_bbox_targets ,
                                                    bbox_pred = pos_decode_bbox_pred,)
                # loss_mask = self.loss_mask(
                #     pos_decode_mask_pred,
                #     pos_decode_mask_targets,
                #     weight=pos_mask_weight,
                #     avg_factor=1.0)/stride[0]/self.sample_num
                
                loss_bbox = self.loss_bbox(
                    pos_decode_bbox_pred,
                    pos_bbox_targets,
                    weight=pos_bbox_weight,
                    avg_factor=1.0)
                # pos_decode_mask_pred = fourier_to_ploy(pos_bbox_pred,\
                #                                             self.fourier_degree,self.sample_num)
                # a=pos_decode_mask_pred
                # pos_decode_bbox_pred = torch.stack([a[:, :,0].min(1)[0],a[:, :,1].min(1)[0],a[:,:, 0].max(1)[0],a[:, :,1].max(1)[0]],-1)
                # loss_bbox = self.loss_bbox(
                #     pos_decode_bbox_pred,
                #     pos_bbox_targets,
                #     weight=pos_bbox_weight,
                #     avg_factor=1.0)
                # if self. point_loss:
                #     pos_decode_mask_targets = fourier_to_ploy(pos_mask_targets,\
                #                                                 self.fourier_degree,self.sample_num).reshape(-1,self.sample_num*2)
                #     pos_decode_mask_pred = pos_decode_mask_pred.reshape(-1,2*self.sample_num)
                #     pos_bbox_weight = pos_bbox_weight[:,None].repeat((1,self.sample_num*2))
                #     loss_mask = self.loss_mask(
                #         pos_decode_mask_pred,
                #         pos_decode_mask_targets,
                #         weight=pos_bbox_weight,
                #         avg_factor=1.0)/stride[0]
                # else:
                #     pos_bbox_weight = pos_bbox_weight[:,None].repeat((1,self.reg_channel))
                #     loss_mask = self.loss_mask(
                #         pos_bbox_pred,
                #         pos_mask_targets,
                #         weight=pos_bbox_weight,
                #         avg_factor=1.0)/stride[0]

            elif self.reg_format=='bbox+mask':
                pos_decode_bbox_pred = pos_bbox_pred[:,:4]
                pos_decode_mask_pred = pos_bbox_pred[:,4:]
                loss_bbox = self.loss_bbox(
                    pos_decode_bbox_pred,
                    pos_bbox_targets,
                    weight=pos_bbox_weight,
                    avg_factor=1.0)
                if self.point_loss:
                    pos_mask_weight = pos_bbox_weight[:,None].repeat((1,self.sample_num*2))
                    pos_decode_mask_pred = fourier_to_ploy(pos_decode_mask_pred,\
                                                                self.fourier_degree,self.sample_num).reshape(-1,2*self.sample_num)
                    pos_decode_mask_targets = fourier_to_ploy(pos_mask_targets,\
                                                                self.fourier_degree,self.sample_num).reshape(-1,2*self.sample_num)
                    loss_mask = self.loss_mask(
                        pos_decode_mask_pred,
                        pos_decode_mask_targets,
                        weight=pos_mask_weight,
                        avg_factor=1.0)/stride[0]
                else:
                    pos_mask_weight = pos_bbox_weight[:,None].repeat((1,self.reg_channel-4))
                    loss_mask = self.loss_mask(
                        pos_decode_mask_pred,
                        pos_mask_targets,
                        weight=pos_mask_weight,
                        avg_factor=1.0)/stride[0]
                    
        else:
            loss_bbox = bbox_pred.sum() * 0
            loss_mask = bbox_pred.sum() * 0
            pos_bbox_weight = bbox_targets.new_tensor(0.)

        return loss_cls, loss_bbox, loss_mask,alignment_metrics.sum(
        ), pos_bbox_weight.sum()

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_masks, 
             gt_segs,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Decoded box for each scale
                level with shape (N, num_anchors * 4, H, W) in
                [tl_x, tl_y, br_x, br_y] format.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (list[Tensor] | None): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        num_imgs = len(img_metas)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1

        flatten_cls_scores = torch.cat([
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                  self.cls_out_channels)
            for cls_score in cls_scores
        ], 1)
        flatten_bbox_preds = torch.cat([
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.reg_channel) #* stride[0]
            for bbox_pred, stride in zip(bbox_preds,
                                         self.prior_generator.strides)
        ], 1)


        cls_reg_targets = self.get_targets(
            flatten_cls_scores,
            flatten_bbox_preds,
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            gt_masks, gt_segs,#TODO insert
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        (anchor_list, labels_list, label_weights_list, bbox_targets_list,mask_targets_list,#TODO insert
         alignment_metrics_list) = cls_reg_targets

        losses_cls, losses_bbox,loss_mask,\
            cls_avg_factors, bbox_avg_factors = multi_apply(
                self.loss_single,
                anchor_list,
                cls_scores,
                bbox_preds,
                labels_list,
                label_weights_list,
                bbox_targets_list,
                mask_targets_list, #TODO insert
                alignment_metrics_list,
                self.prior_generator.strides)

        cls_avg_factor = reduce_mean(sum(cls_avg_factors)).clamp_(min=1).item()
        losses_cls = list(map(lambda x: x / cls_avg_factor, losses_cls))

        bbox_avg_factor = reduce_mean(
            sum(bbox_avg_factors)).clamp_(min=1).item()
        losses_bbox = list(map(lambda x: x / bbox_avg_factor, losses_bbox))
        # return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)
        if self.reg_format=='bbox+mask' or self.reg_format=='mask' :
            losses_mask = list(map(lambda x: x / bbox_avg_factor, loss_mask))
            return dict(loss_cls=losses_cls, loss_bbox=losses_bbox,losses_mask=losses_mask)
        else:
            return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)

    def get_targets(self,
                    cls_scores,
                    bbox_preds,
                    anchor_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    gt_masks_list, gt_segs_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    label_channels=1,
                    unmap_outputs=True):#TODO insert                gt_masks_list, gt_segs_list,
        """Compute regression and classification targets for anchors in
        multiple images.

        Args:
            cls_scores (Tensor): Classification predictions of images,
                a 3D-Tensor with shape [num_imgs, num_priors, num_classes].
            bbox_preds (Tensor): Decoded bboxes predictions of one image,
                a 3D-Tensor with shape [num_imgs, num_priors, 4] in [tl_x,
                tl_y, br_x, br_y] format.
            anchor_list (list[list[Tensor]]): Multi level anchors of each
                image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, 4).
            valid_flag_list (list[list[Tensor]]): Multi level valid flags of
                each image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, )
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            img_metas (list[dict]): Meta info of each image.
            gt_bboxes_ignore_list (list[Tensor]): Ground truth bboxes to be
                ignored.
            gt_labels_list (list[Tensor]): Ground truth labels of each box.
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple: a tuple containing learning targets.

                - anchors_list (list[list[Tensor]]): Anchors of each level.
                - labels_list (list[Tensor]): Labels of each level.
                - label_weights_list (list[Tensor]): Label weights of each
                  level.
                - bbox_targets_list (list[Tensor]): BBox targets of each level.
                - norm_alignment_metrics_list (list[Tensor]): Normalized
                  alignment metrics of each level.
        """
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        num_level_anchors_list = [num_level_anchors] * num_imgs

        # concat all level anchors and flags to a single tensor
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            anchor_list[i] = torch.cat(anchor_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        # anchor_list: list(b * [-1, 4])

        if self.epoch < self.initial_epoch:
            (all_anchors, all_labels, all_label_weights, all_bbox_targets,all_mask_targets,#TODO insert
             all_bbox_weights, pos_inds_list, neg_inds_list) = multi_apply(
                 self._get_atss_target_single,
                 anchor_list,
                 valid_flag_list,
                 num_level_anchors_list,
                 gt_bboxes_list,
                 gt_masks_list, gt_segs_list,#TODOinsert
                 gt_bboxes_ignore_list,
                 gt_labels_list,
                 img_metas,
                 label_channels=label_channels,
                 unmap_outputs=unmap_outputs)
            all_assign_metrics = [
                weight[..., 0] for weight in all_bbox_weights
            ]
        else:
            (all_anchors, all_labels, all_label_weights, all_bbox_targets,all_mask_targets, #TODO insert
             all_assign_metrics) = multi_apply(
                 self._get_target_single,
                 cls_scores,
                 bbox_preds,
                 anchor_list,
                 valid_flag_list,
                 gt_bboxes_list,
                 gt_masks_list, gt_segs_list,
                 gt_bboxes_ignore_list,
                 gt_labels_list,
                 img_metas,
                 label_channels=label_channels,
                 unmap_outputs=unmap_outputs)
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None

        # split targets to a list w.r.t. multiple levels
        anchors_list = images_to_levels(all_anchors, num_level_anchors)
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        mask_targets_list = images_to_levels(all_mask_targets,
                                             num_level_anchors)#TODO insert
        norm_alignment_metrics_list = images_to_levels(all_assign_metrics,
                                                       num_level_anchors)

        return (anchors_list, labels_list, label_weights_list,
                bbox_targets_list, 
                mask_targets_list, #TODO insert
                norm_alignment_metrics_list)

    def _get_atss_target_single(self,
                           flat_anchors,
                           valid_flags,
                           num_level_anchors,
                           gt_bboxes,
                           gt_masks, gt_segs,
                           gt_bboxes_ignore,
                           gt_labels,
                           img_meta,
                           label_channels=1,
                           unmap_outputs=True):

        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None, ) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        num_level_anchors_inside = self.get_num_level_anchors_inside(
            num_level_anchors, inside_flags)
        assign_result = self.assigner.assign(anchors, num_level_anchors_inside,
                                             gt_bboxes, gt_bboxes_ignore,
                                             gt_labels)

        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_bboxes)
        
        #TODO insert
        device = gt_bboxes.device
        gt_masks = torch.tensor(gt_masks,dtype =gt_bboxes.dtype ).to(device)
        sampling_result_mask = self.sampler.sample(assign_result, anchors,
                                              gt_masks)
        
        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        mask_targets = torch.zeros((anchors.size()[0],gt_masks.size()[1]),dtype = anchors.dtype,device = device)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            # if self.reg_decoded_bbox:
            pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            # else:
            #     pos_bbox_targets = self.bbox_coder.encode(
            #         sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            pos_mask_targets = sampling_result_mask.pos_gt_bboxes
            mask_targets[pos_inds, :] = pos_mask_targets
            # pos_mask_targets = sampling_result_mask.pos_gt_bboxes
            
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class since v2.5.0
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            anchors = unmap(anchors, num_total_anchors, inside_flags)
            labels = unmap(
                labels, num_total_anchors, inside_flags, fill=self.num_classes)
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            mask_targets = unmap(mask_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

        return (anchors, labels, label_weights, bbox_targets, mask_targets,bbox_weights,
                pos_inds, neg_inds)


    def _get_target_single(self,
                           cls_scores,
                           bbox_preds,
                           flat_anchors,
                           valid_flags,
                           gt_bboxes,
                           gt_masks, gt_segs,
                           gt_bboxes_ignore,
                           gt_labels,
                           img_meta,
                           label_channels=1,
                           unmap_outputs=True):
        """Compute regression, classification targets for anchors in a single
        image.

        Args:
            cls_scores (list(Tensor)): Box scores for each image.
            bbox_preds (list(Tensor)): Box energies / deltas for each image.
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors ,4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            img_meta (dict): Meta info of the image.
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple: N is the number of total anchors in the image.
                anchors (Tensor): All anchors in the image with shape (N, 4).
                labels (Tensor): Labels of all anchors in the image with shape
                    (N,).
                label_weights (Tensor): Label weights of all anchor in the
                    image with shape (N,).
                bbox_targets (Tensor): BBox targets of all anchors in the
                    image with shape (N, 4).
                norm_alignment_metrics (Tensor): Normalized alignment metrics
                    of all priors in the image with shape (N,).
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None, ) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]
        
        #TODO insert 如果这是mask回归，则将mask 转为 正框
        # if self.reg_format =='mask':
        #     bbox_preds = fourier_to_bbox(bbox_preds,self.fourier_degree,50)
        # elif self.reg_format =='rbox':
        #     bbox_preds = rotated_box_to_bbox(bbox_preds)
        # assign gt and sample anchors
        #TODO insert
        if self.reg_format=='bbox' or self.reg_format=='bbox+mask':
            bbox_preds = bbox_preds[:,:4]
        elif  self.reg_format=='mask' :
            pos_decode_mask_pred = fourier_to_ploy(bbox_preds,\
                                                        self.fourier_degree,self.sample_num)
            a=pos_decode_mask_pred
            bbox_preds = torch.stack([a[:, :,0].min(1)[0],a[:, :,1].min(1)[0],a[:,:, 0].max(1)[0],a[:, :,1].max(1)[0]],-1)
            
        assign_result = self.alignment_assigner.assign(
            cls_scores[inside_flags, :], bbox_preds[inside_flags, :], anchors,
            gt_bboxes, gt_bboxes_ignore, gt_labels, self.alpha, self.beta)
        assign_ious = assign_result.max_overlaps
        assign_metrics = assign_result.assign_metrics

        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_bboxes)
        # TODO insert
        device = gt_bboxes.device
        gt_masks = torch.tensor(gt_masks,dtype =gt_bboxes.dtype ).to(device)
        sampling_result_mask = self.sampler.sample(assign_result, anchors,
                                              gt_masks)
        mask_targets = torch.zeros((anchors.size()[0],gt_masks.size()[1]),dtype = anchors.dtype,device = device)
        
        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)
        norm_alignment_metrics = anchors.new_zeros(
            num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            # point-based
            pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            #TODO insert
            pos_mask_targets = sampling_result_mask.pos_gt_bboxes
            mask_targets[pos_inds, :] = pos_mask_targets
            
            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class since v2.5.0
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        class_assigned_gt_inds = torch.unique(
            sampling_result.pos_assigned_gt_inds)
        for gt_inds in class_assigned_gt_inds:
            gt_class_inds = pos_inds[sampling_result.pos_assigned_gt_inds ==
                                     gt_inds]
            pos_alignment_metrics = assign_metrics[gt_class_inds]
            pos_ious = assign_ious[gt_class_inds]
            pos_norm_alignment_metrics = pos_alignment_metrics / (
                pos_alignment_metrics.max() + 10e-8) * pos_ious.max()
            norm_alignment_metrics[gt_class_inds] = pos_norm_alignment_metrics

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            anchors = unmap(anchors, num_total_anchors, inside_flags)
            labels = unmap(
                labels, num_total_anchors, inside_flags, fill=self.num_classes)
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            mask_targets = unmap(mask_targets, num_total_anchors, inside_flags)#TODOinsert 
            norm_alignment_metrics = unmap(norm_alignment_metrics,
                                           num_total_anchors, inside_flags)
        return (anchors, labels, label_weights, bbox_targets,mask_targets, #TODO insert
                norm_alignment_metrics)
    
    

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   score_factors=None,
                   img_metas=None,
                   cfg=None,
                   rescale=False,
                   with_nms=True,
                   **kwargs):
        """Transform network outputs of a batch into bbox results.

        Note: When score_factors is not None, the cls_scores are
        usually multiplied by it then obtain the real score used in NMS,
        such as CenterNess in FCOS, IoU branch in ATSS.

        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            score_factors (list[Tensor], Optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, num_priors * 1, H, W). Default None.
            img_metas (list[dict], Optional): Image meta info. Default None.
            cfg (mmcv.Config, Optional): Test / postprocessing configuration,
                if None, test_cfg would be used.  Default None.
            rescale (bool): If True, return boxes in original image space.
                Default False.
            with_nms (bool): If True, do nms before return boxes.
                Default True.

        Returns:
            list[list[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of
                the corresponding box.
        """
        assert len(cls_scores) == len(bbox_preds)

        if score_factors is None:
            # e.g. Retina, FreeAnchor, Foveabox, etc.
            with_score_factors = False
        else:
            # e.g. FCOS, PAA, ATSS, AutoAssign, etc.
            with_score_factors = True
            assert len(cls_scores) == len(score_factors)

        num_levels = len(cls_scores)

        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        # TODO change 为了减少推理过程中生成anchor的次数
        # mlvl_priors = self.prior_generator.grid_priors(
        #     featmap_sizes,
        #     dtype=cls_scores[0].dtype,
        #     device=cls_scores[0].device)
        if self.all_level_points is None or  featmap_sizes[0][0]*featmap_sizes[0][1] != self.all_level_points[0].shape[0] :
            self.all_level_points = self.prior_generator.grid_priors(
                featmap_sizes, dtype=cls_scores[0].dtype,
                device=cls_scores[0].device)
        mlvl_priors = self.all_level_points
        result_list = []

        for img_id in range(len(img_metas)):
            img_meta = img_metas[img_id]
            cls_score_list = select_single_mlvl(cls_scores, img_id)
            bbox_pred_list = select_single_mlvl(bbox_preds, img_id)
            if with_score_factors:
                score_factor_list = select_single_mlvl(score_factors, img_id)
            else:
                score_factor_list = [None for _ in range(num_levels)]

            results = self._get_bboxes_single(cls_score_list, bbox_pred_list,
                                              score_factor_list, mlvl_priors,
                                              img_meta, cfg, rescale, with_nms,
                                              **kwargs)
            result_list.append(results)
        return result_list
      
    def _get_bboxes_single(self,
                           cls_score_list,
                           bbox_pred_list,
                           score_factor_list,
                           mlvl_priors,
                           img_meta,
                           cfg,
                           rescale=False,
                           with_nms=True,
                           **kwargs):
        if self.reg_format=='bbox' or (self.reg_format=='bbox+mask' and self.training):
            return self._get_bboxes_single_rect(
                            cls_score_list,
                            bbox_pred_list,
                            score_factor_list,
                            mlvl_priors,
                            img_meta,
                            cfg,
                            rescale,
                            with_nms
                           )
        elif self.reg_format=='mask' or  (self.reg_format=='bbox+mask' and self.training==False):
            return self._get_bboxes_single_mask(
                            cls_score_list,
                            bbox_pred_list,
                            score_factor_list,
                            mlvl_priors,
                            img_meta,
                            cfg,
                            rescale,
                            with_nms,
                           )
        else:
            raise NotImplementedError(
                    f'Unknown anchor type: {self.anchor_type}.'
                    f'Please use `anchor_free` or `anchor_based`.')
            
           
    def _get_bboxes_single_mask(self,
                           cls_score_list,
                           bbox_pred_list,
                           score_factor_list,
                           mlvl_priors,
                           img_meta,
                           cfg,
                           rescale=False,
                           with_nms=True,
                           **kwargs):
        """Transform outputs of a single image into bbox predictions.

        Args:
            cls_score_list (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_priors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas from
                all scale levels of a single image, each item has shape
                (num_priors * 4, H, W).
            score_factor_list (list[Tensor]): Score factor from all scale
                levels of a single image, each item has shape
                (num_priors * 1, H, W).
            mlvl_priors (list[Tensor]): Each element in the list is
                the priors of a single level in feature pyramid. In all
                anchor-based methods, it has shape (num_priors, 4). In
                all anchor-free methods, it has shape (num_priors, 2)
                when `with_stride=True`, otherwise it still has shape
                (num_priors, 4).
            img_meta (dict): Image meta info.
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            tuple[Tensor]: Results of detected bboxes and labels. If with_nms
                is False and mlvl_score_factor is None, return mlvl_bboxes and
                mlvl_scores, else return mlvl_bboxes, mlvl_scores and
                mlvl_score_factor. Usually with_nms is False is used for aug
                test. If with_nms is True, then return the following format

                - det_bboxes (Tensor): Predicted bboxes with shape \
                    [num_bboxes, 5], where the first 4 columns are bounding \
                    box positions (tl_x, tl_y, br_x, br_y) and the 5-th \
                    column are scores between 0 and 1.
                - det_labels (Tensor): Predicted labels of the corresponding \
                    box with shape [num_bboxes].
        """
    
        cfg = self.test_cfg if cfg is None else cfg
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_labels = []
        for cls_score, bbox_pred, priors, stride in zip(
                cls_score_list, bbox_pred_list, mlvl_priors,
                self.prior_generator.strides):

            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, self.reg_channel) #* stride[0]
            scores = cls_score.permute(1, 2,
                                       0).reshape(-1, self.cls_out_channels)

            # After https://github.com/open-mmlab/mmdetection/pull/6268/,
            # this operation keeps fewer bboxes under the same `nms_pre`.
            # There is no difference in performance for most models. If you
            # find a slight drop in performance, you can set a larger
            # `nms_pre` than before.
            results = filter_scores_and_topk(
                scores, cfg.score_thr, nms_pre,
                dict(bbox_pred=bbox_pred, priors=priors))
            scores, labels, keep_idxs, filtered_results = results

            bboxes = filtered_results['bbox_pred']
            #TODO insert
            bboxes = fourier_to_ploy(bboxes,self.fourier_degree,self.sample_num)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)

        # return self._bbox_post_process(mlvl_scores, mlvl_labels, mlvl_bboxes,
        #                                img_meta['scale_factor'], cfg, rescale,
        #                                with_nms, None, **kwargs)
        return self._mask_post_process(mlvl_scores, mlvl_labels, mlvl_bboxes,
                                       img_meta['scale_factor'], cfg, rescale,
                                       with_nms, None, **kwargs)

    def _mask_post_process(self,
                           mlvl_scores,
                           mlvl_labels,
                           mlvl_bboxes,
                           scale_factor,
                           cfg,
                           rescale=False,
                           with_nms=True,
                           mlvl_score_factors=None,
                           **kwargs):
        """bbox post-processing method.

        The boxes would be rescaled to the original image scale and do
        the nms operation. Usually with_nms is False is used for aug test.

        Args:
            mlvl_scores (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_bboxes, ).
           mlvl_labels (list[Tensor]): Box class labels from all scale
                levels of a single image, each item has shape
                (num_bboxes, ).
            mlvl_bboxes (list[Tensor]): Decoded bboxes from all scale
                levels of a single image, each item has shape (num_bboxes, 4).
            scale_factor (ndarray, optional): Scale factor of the image arange
                as (w_scale, h_scale, w_scale, h_scale). 
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.
            mlvl_score_factors (list[Tensor], optional): Score factor from
                all scale levels of a single image, each item has shape
                (num_bboxes, ). Default: None.

        Returns:
            tuple[Tensor]: Results of detected bboxes and labels. If with_nms
                is False and mlvl_score_factor is None, return mlvl_bboxes and
                mlvl_scores, else return mlvl_bboxes, mlvl_scores and
                mlvl_score_factor. Usually with_nms is False is used for aug
                test. If with_nms is True, then return the following format

                - det_bboxes (Tensor): Predicted bboxes with shape \
                    [num_bboxes, 5], where the first 4 columns are bounding \
                    box positions (tl_x, tl_y, br_x, br_y) and the 5-th \
                    column are scores between 0 and 1.
                - det_labels (Tensor): Predicted labels of the corresponding \
                    box with shape [num_bboxes].
        """
        assert len(mlvl_scores) == len(mlvl_bboxes) == len(mlvl_labels)

        mlvl_bboxes = torch.cat(mlvl_bboxes).permute(0,2,1)
        if rescale:
            scale_factor_x =  mlvl_bboxes.new_tensor(scale_factor)[0]
            scale_factor_y =  mlvl_bboxes.new_tensor(scale_factor)[1]

            mlvl_bboxes[:,0] /= scale_factor_x
            mlvl_bboxes[:,1] /= scale_factor_y

        mlvl_scores = torch.cat(mlvl_scores)
        mlvl_labels = torch.cat(mlvl_labels)

        if mlvl_score_factors is not None:
            # TODO： Add sqrt operation in order to be consistent with
            #  the paper.
            mlvl_score_factors = torch.cat(mlvl_score_factors)
            mlvl_scores = mlvl_scores * mlvl_score_factors

        if with_nms:
            if mlvl_bboxes.numel() == 0:
                mlvl_bboxes = torch.zeros((0,4),dtype =mlvl_bboxes.dtype ).to( mlvl_bboxes.device)
                det_bboxes = torch.cat([mlvl_bboxes, mlvl_scores[:, None]], -1)
                det_masks = torch.zeros((0,2,self.sample_num),dtype =mlvl_bboxes.dtype ).to( mlvl_bboxes.device)
                return det_bboxes, mlvl_labels, det_masks
            
            a = mlvl_bboxes
            _mlvl_bboxes = torch.stack([a[:, 0].min(1)[0],a[:, 1].min(1)[0],a[:, 0].max(1)[0],a[:, 1].max(1)[0]],-1)
        
            det_bboxes, keep_idxs = batched_nms(_mlvl_bboxes, mlvl_scores,
                                                mlvl_labels, cfg.nms)
            det_bboxes = det_bboxes[:cfg.max_per_img]
            det_labels = mlvl_labels[keep_idxs][:cfg.max_per_img]
            det_masks = mlvl_bboxes[keep_idxs][:cfg.max_per_img]
            
            return det_bboxes, det_labels, det_masks
        else:
            return mlvl_bboxes, mlvl_scores, mlvl_labels

    def _get_bboxes_single_rect(self,
                           cls_score_list,
                           bbox_pred_list,
                           score_factor_list,
                           mlvl_priors,
                           img_meta,
                           cfg,
                           rescale=False,
                           with_nms=True,
                           **kwargs):
        """Transform outputs of a single image into bbox predictions.

        Args:
            cls_score_list (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_priors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas from
                all scale levels of a single image, each item has shape
                (num_priors * 4, H, W).
            score_factor_list (list[Tensor]): Score factor from all scale
                levels of a single image, each item has shape
                (num_priors * 1, H, W).
            mlvl_priors (list[Tensor]): Each element in the list is
                the priors of a single level in feature pyramid. In all
                anchor-based methods, it has shape (num_priors, 4). In
                all anchor-free methods, it has shape (num_priors, 2)
                when `with_stride=True`, otherwise it still has shape
                (num_priors, 4).
            img_meta (dict): Image meta info.
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            tuple[Tensor]: Results of detected bboxes and labels. If with_nms
                is False and mlvl_score_factor is None, return mlvl_bboxes and
                mlvl_scores, else return mlvl_bboxes, mlvl_scores and
                mlvl_score_factor. Usually with_nms is False is used for aug
                test. If with_nms is True, then return the following format

                - det_bboxes (Tensor): Predicted bboxes with shape \
                    [num_bboxes, 5], where the first 4 columns are bounding \
                    box positions (tl_x, tl_y, br_x, br_y) and the 5-th \
                    column are scores between 0 and 1.
                - det_labels (Tensor): Predicted labels of the corresponding \
                    box with shape [num_bboxes].
        """

        cfg = self.test_cfg if cfg is None else cfg
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_labels = []
        for cls_score, bbox_pred, priors, stride in zip(
                cls_score_list, bbox_pred_list, mlvl_priors,
                self.prior_generator.strides):

            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            
            if self.reg_format =='bbox' or self.reg_format=='bbox+mask':
                bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, self.reg_channel)[:,:4] #* stride[0]
            elif self.reg_format=='mask':
                bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, self.reg_channel)
                pos_decode_mask_pred = fourier_to_ploy(bbox_pred,\
                                                        self.fourier_degree,self.sample_num)
                a=pos_decode_mask_pred
                bbox_pred = torch.stack([a[:, :,0].min(1)[0],a[:, :,1].min(1)[0],a[:,:, 0].max(1)[0],a[:, :,1].max(1)[0]],-1)
                
            # bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4) #* stride[0]
            scores = cls_score.permute(1, 2,
                                       0).reshape(-1, self.cls_out_channels)

            # After https://github.com/open-mmlab/mmdetection/pull/6268/,
            # this operation keeps fewer bboxes under the same `nms_pre`.
            # There is no difference in performance for most models. If you
            # find a slight drop in performance, you can set a larger
            # `nms_pre` than before.
            results = filter_scores_and_topk(
                scores, cfg.score_thr, nms_pre,
                dict(bbox_pred=bbox_pred, priors=priors))
            scores, labels, keep_idxs, filtered_results = results

            bboxes = filtered_results['bbox_pred']

            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)

        return self._bbox_post_process(mlvl_scores, mlvl_labels, mlvl_bboxes,
                                       img_meta['scale_factor'], cfg, rescale,
                                       with_nms, None, **kwargs)
        

    def deform_sampling(self, feat, offset):
        """Sampling the feature x according to offset.

        Args:
            feat (Tensor): Feature
            offset (Tensor): Spatial offset for feature sampling
        """
        # it is an equivalent implementation of bilinear interpolation
        b, c, h, w = feat.shape
        weight = feat.new_ones(c, 1, 1, 1)
        y = deform_conv2d(feat, offset, weight, 1, 0, 1, c, c)
        return y

    def anchor_center(self, anchors):
        """Get anchor centers from anchors.

        Args:
            anchors (Tensor): Anchor list with shape (N, 4), "xyxy" format.

        Returns:
            Tensor: Anchor centers with shape (N, 2), "xy" format.
        """
        anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
        anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2
        return torch.stack([anchors_cx, anchors_cy], dim=-1)



class TaskDecomposition(nn.Module):
    """Task decomposition module in task-aligned predictor of TOOD.

    Args:
        feat_channels (int): Number of feature channels in TOOD head.
        stacked_convs (int): Number of conv layers in TOOD head.
        la_down_rate (int): Downsample rate of layer attention.
        conv_cfg (dict): Config dict for convolution layer.
        norm_cfg (dict): Config dict for normalization layer.
    """

    def __init__(self,
                 feat_channels,
                 stacked_convs,
                 la_down_rate=8,
                 conv_cfg=None,
                 norm_cfg=None):
        super(TaskDecomposition, self).__init__()
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.in_channels = self.feat_channels * self.stacked_convs
        self.norm_cfg = norm_cfg
        self.layer_attention = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels // la_down_rate, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                self.in_channels // la_down_rate,
                self.stacked_convs,
                1,
                padding=0), nn.Sigmoid())

        self.reduction_conv = ConvModule(
            self.in_channels,
            self.feat_channels,
            1,
            stride=1,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            bias=norm_cfg is None)

    def init_weights(self):
        for m in self.layer_attention.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)
        normal_init(self.reduction_conv.conv, std=0.01)

    def forward(self, feat, avg_feat=None):
        b, c, h, w = feat.shape
        if avg_feat is None:
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
        weight = self.layer_attention(avg_feat)

        # here we first compute the product between layer attention weight and
        # conv weight, and then compute the convolution between new conv weight
        # and feature map, in order to save memory and FLOPs.
        conv_weight = weight.reshape(
            b, 1, self.stacked_convs,
            1) * self.reduction_conv.conv.weight.reshape(
                1, self.feat_channels, self.stacked_convs, self.feat_channels)
        conv_weight = conv_weight.reshape(b, self.feat_channels,
                                          self.in_channels)
        feat = feat.reshape(b, self.in_channels, h * w)
        feat = torch.bmm(conv_weight, feat).reshape(b, self.feat_channels, h,
                                                    w)
        if self.norm_cfg is not None:
            feat = self.reduction_conv.norm(feat)
        feat = self.reduction_conv.activate(feat)

        return feat
    