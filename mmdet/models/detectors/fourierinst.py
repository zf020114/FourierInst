# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .single_stage import SingleStageDetector
import pycocotools.mask as maskUtils
from mmdet.core import bbox2result
import numpy as np
from mmdet.core.bbox.transforms_rotate import bbox_poly2result

@DETECTORS.register_module()
class FourierInst(SingleStageDetector):
    r"""Implementation of `TOOD: Task-aligned One-stage Object Detection.
    <https://arxiv.org/abs/2108.07755>`_."""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(FourierInst, self).__init__(backbone, neck, bbox_head, train_cfg,
                                   test_cfg, pretrained, init_cfg)

    def set_epoch(self, epoch):
        self.bbox_head.epoch = epoch


    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_masks,
                      gt_semantic_seg=None,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(SingleStageDetector, self).forward_train(img, img_metas)
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes, gt_masks, gt_semantic_seg,
                                              gt_labels, gt_bboxes_ignore)
        return losses
    
    def simple_test(self, img, img_metas, rescale=False):
        if self.bbox_head.reg_format=='bbox':
            return  self.simple_test_bbox( img, img_metas, rescale)
        elif self.bbox_head.reg_format=='mask' or self.bbox_head.reg_format=='mask+refine' :
            return  self.simple_test_mask( img, img_metas, rescale)
        else:
            print('Not in [bbox, rbox, mask]!')
            
    def simple_test_bbox(self, img, img_metas, rescale=False):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        feat = self.extract_feat(img)
        results_list = self.bbox_head.simple_test(
            feat, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def simple_test_mask(self, img, img_metas, rescale=False):
        feat = self.extract_feat(img)
        results_list = self.bbox_head.simple_test(
            feat, img_metas, rescale=rescale)
        if  self.bbox_head.show_result: 
            results = [
                self.bbox_mask2result(det_bboxes, det_masks, det_labels, self.bbox_head.num_classes, img_metas[0])
                for det_bboxes, det_labels, det_masks in results_list]
            bbox_results = [results[0][0]]
            mask_results = [results[0][1]]
            return list(zip(bbox_results, mask_results))#bbox_results, mask_results
        else:
            bbox_results = [
                bbox_poly2result(det_bboxes, det_masks, det_labels, 
                                    self.bbox_head.num_classes,
                                    self.bbox_head.sample_num)
                for det_bboxes, det_labels, det_masks  in results_list
            ]
            return bbox_results[0]

    # def simple_test_mask(self, img, img_metas, rescale=False):
    #     feat = self.extract_feat(img)
    #     results_list = self.bbox_head.simple_test(
    #         feat, img_metas, rescale=rescale)

    #     results = [
    #         self.bbox_mask2result(det_bboxes, det_masks, det_labels, self.bbox_head.num_classes, img_metas[0])
    #         for det_bboxes, det_labels, det_masks in results_list]

    #     bbox_results = [results[0][0]]
    #     mask_results = [results[0][1]]

    #     return list(zip(bbox_results, mask_results))#bbox_results, mask_results

    def bbox_mask2result(self, bboxes, masks, labels, num_classes, img_meta):
        '''bbox and mask 转成result mask要画图'''
        """Convert detection results to a list of numpy arrays.

        Args:
            bboxes (Tensor): shape (n, 5)
            masks (Tensor): shape (n, 2, 36)
            labels (Tensor): shape (n, )
            num_classes (int): class number, including background class

        Returns:
            list(ndarray): bbox results of each class 
        """
        ori_shape = img_meta['ori_shape']
        img_h, img_w, _ = ori_shape
        mask_results = [[] for _ in range(num_classes )]
        # im_mask_model = np.zeros((img_h, img_w), dtype=np.uint8)
        for i in range(masks.shape[0]):
            # im_mask = im_mask_model.copy()
            # mask = [masks[i].transpose(1,0).unsqueeze(1).int().data.cpu().numpy()]
            # im_mask = cv2.drawContours(im_mask, mask, -1,1,-1)
            mask_rles =[ masks[i].permute(1,0).reshape(-1).cpu().numpy()]
            rles = maskUtils.frPyObjects(mask_rles, img_h, img_w)
            rle = maskUtils.merge(rles)
            im_mask = maskUtils.decode(rle)

            label = labels[i]
            mask_results[label].append(im_mask)

        if bboxes.shape[0] == 0:
            bbox_results = [
                np.zeros((0, 5), dtype=np.float32) for i in range(num_classes )
            ]
            return bbox_results, mask_results
        else:
            bboxes = bboxes.cpu().numpy()
            labels = labels.cpu().numpy()
            bbox_results = [bboxes[labels == i, :] for i in range(num_classes )]
            return bbox_results, mask_results
