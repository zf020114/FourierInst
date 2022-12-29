# Copyright (c) OpenMMLab. All rights reserved.
import math

import cv2
import numpy as np
import torch
import pycocotools.mask as mask_util

def bbox_flip(bboxes, img_shape, direction='horizontal'):
    """Flip bboxes horizontally or vertically.

    Args:
        bboxes (Tensor): Shape (..., 5*k)
        img_shape (tuple): Image shape.
        direction (str): Flip direction, options are "horizontal", "vertical",
            "diagonal". Default: "horizontal"

    Returns:
        Tensor: Flipped bboxes.
    """
    version = 'oc'
    assert bboxes.shape[-1] % 5 == 0
    assert direction in ['horizontal', 'vertical', 'diagonal']
    flipped = bboxes.clone()
    if direction == 'horizontal':
        flipped[:, 0] = img_shape[1] - bboxes[:, 0] - 1
    elif direction == 'vertical':
        flipped[:, 1] = img_shape[0] - bboxes[:, 1] - 1
    else:
        flipped[:, 0] = img_shape[1] - bboxes[:, 0] - 1
        flipped[:, 1] = img_shape[0] - bboxes[:, 1] - 1
    if version == 'oc':
        rotated_flag = (bboxes[:, 4] != np.pi / 2)
        flipped[rotated_flag, 4] = np.pi / 2 - bboxes[rotated_flag, 4]
        flipped[rotated_flag, 2] = bboxes[rotated_flag, 3]
        flipped[rotated_flag, 3] = bboxes[rotated_flag, 2]
    else:
        flipped[:, 4] = norm_angle(np.pi - bboxes[:, 4], version)
    return flipped


def bbox_mapping_back(bboxes,
                      img_shape,
                      scale_factor,
                      flip,
                      flip_direction='horizontal'):
    """Map bboxes from testing scale to original image scale."""
    new_bboxes = bbox_flip(bboxes, img_shape,
                           flip_direction) if flip else bboxes
    new_bboxes[:, :4] = new_bboxes[:, :4] / new_bboxes.new_tensor(scale_factor)
    return new_bboxes.view(bboxes.shape)


def rbbox2result(bboxes, labels, num_classes):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (torch.Tensor): shape (n, 6)
        labels (torch.Tensor): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    if bboxes.shape[0] == 0:
        return [np.zeros((0, 6), dtype=np.float32) for _ in range(num_classes)]
    else:
        bboxes = bboxes.cpu().numpy()
        labels = labels.cpu().numpy()
        return [bboxes[labels == i, :] for i in range(num_classes)]


def rbbox2roi(bbox_list):
    """Convert a list of bboxes to roi format.

    Args:
        bbox_list (list[Tensor]): a list of bboxes corresponding to a batch
            of images.

    Returns:
        Tensor: shape (n, 6), [batch_ind, cx, cy, w, h, a]
    """
    rois_list = []
    for img_id, bboxes in enumerate(bbox_list):
        if bboxes.size(0) > 0:
            img_inds = bboxes.new_full((bboxes.size(0), 1), img_id)
            rois = torch.cat([img_inds, bboxes[:, :5]], dim=-1)
        else:
            rois = bboxes.new_zeros((0, 6))
        rois_list.append(rois)
    rois = torch.cat(rois_list, 0)
    return rois


def poly2obb(polys, version='oc'):
    """Convert polygons to oriented bounding boxes.

    Args:
        polys (torch.Tensor): [x0,y0,x1,y1,x2,y2,x3,y3]
        version (Str): angle representations.

    Returns:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
    """
    if version == 'oc':
        results = poly2obb_oc(polys)
    elif version == 'le135':
        results = poly2obb_le135(polys)
    elif version == 'le90':
        results = poly2obb_le90(polys)
    else:
        raise NotImplementedError
    return results


def poly2obb_np(polys, version='oc'):
    """Convert polygons to oriented bounding boxes.

    Args:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3]
        version (Str): angle representations.

    Returns:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle]
    """
    if version == 'oc':
        results = poly2obb_np_oc(polys)
    elif version == 'le135':
        results = poly2obb_np_le135(polys)
    elif version == 'le90':
        results = poly2obb_np_le90(polys)
    else:
        raise NotImplementedError
    return results


def obb2hbb(rbboxes, version='oc'):
    """Convert oriented bounding boxes to horizontal bounding boxes.

    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
        version (Str): angle representations.

    Returns:
        hbbs (torch.Tensor): [x_ctr,y_ctr,w,h,-pi/2]
    """
    if version == 'oc':
        results = obb2hbb_oc(rbboxes)
    elif version == 'le135':
        results = obb2hbb_le135(rbboxes)
    elif version == 'le90':
        results = obb2hbb_le90(rbboxes)
    else:
        raise NotImplementedError
    return results


def obb2poly(rbboxes, version='oc'):
    """Convert oriented bounding boxes to polygons.

    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
        version (Str): angle representations.

    Returns:
        polys (torch.Tensor): [x0,y0,x1,y1,x2,y2,x3,y3]
    """
    if version == 'oc':
        results = obb2poly_oc(rbboxes)
    elif version == 'le135':
        results = obb2poly_le135(rbboxes)
    elif version == 'le90':
        results = obb2poly_le90(rbboxes)
    else:
        raise NotImplementedError
    return results


def obb2poly_np(rbboxes, version='oc'):
    """Convert oriented bounding boxes to polygons.

    Args:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle]
        version (Str): angle representations.

    Returns:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3]
    """
    if version == 'oc':
        results = obb2poly_np_oc(rbboxes)
    elif version == 'le135':
        results = obb2poly_np_le135(rbboxes)
    elif version == 'le90':
        results = obb2poly_np_le90(rbboxes)
    else:
        raise NotImplementedError
    return results


def obb2xyxy(rbboxes, version='oc'):
    """Convert oriented bounding boxes to horizontal bounding boxes.

    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
        version (Str): angle representations.

    Returns:
        hbbs (torch.Tensor): [x_lt,y_lt,x_rb,y_rb]
    """
    if version == 'oc':
        results = obb2xyxy_oc(rbboxes)
    elif version == 'le135':
        results = obb2xyxy_le135(rbboxes)
    elif version == 'le90':
        results = obb2xyxy_le90(rbboxes)
    else:
        raise NotImplementedError
    return results


def hbb2obb(hbboxes, version='oc'):
    """Convert horizontal bounding boxes to oriented bounding boxes.

    Args:
        hbbs (torch.Tensor): [x_lt,y_lt,x_rb,y_rb]
        version (Str): angle representations.

    Returns:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
    """
    if version == 'oc':
        results = hbb2obb_oc(hbboxes)
    elif version == 'le135':
        results = hbb2obb_le135(hbboxes)
    elif version == 'le90':
        results = hbb2obb_le90(hbboxes)
    elif version =='mask':
         results = poly2obb_np_mask(hbboxes)
    else:
        raise NotImplementedError
    return results


def poly2obb_oc(polys):
    """Convert polygons to oriented bounding boxes.

    Args:
        polys (torch.Tensor): [x0,y0,x1,y1,x2,y2,x3,y3]

    Returns:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
    """
    points = torch.reshape(polys, [-1, 4, 2])
    cxs = torch.unsqueeze(torch.sum(points[:, :, 0], axis=1), axis=1) / 4.
    cys = torch.unsqueeze(torch.sum(points[:, :, 1], axis=1), axis=1) / 4.
    _ws = torch.unsqueeze(dist_torch(points[:, 0], points[:, 1]), axis=1)
    _hs = torch.unsqueeze(dist_torch(points[:, 1], points[:, 2]), axis=1)
    _thetas = torch.unsqueeze(
        torch.atan2(-(points[:, 1, 0] - points[:, 0, 0]),
                    points[:, 1, 1] - points[:, 0, 1]),
        axis=1)
    odd = torch.eq(torch.remainder((_thetas / (np.pi * 0.5)).floor_(), 2), 0)
    ws = torch.where(odd, _hs, _ws)
    hs = torch.where(odd, _ws, _hs)
    thetas = torch.remainder(_thetas, np.pi * 0.5)
    rbboxes = torch.cat([cxs, cys, ws, hs, thetas], axis=1)
    return rbboxes


def poly2obb_le135(polys):
    """Convert polygons to oriented bounding boxes.

    Args:
        polys (torch.Tensor): [x0,y0,x1,y1,x2,y2,x3,y3]

    Returns:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
    """
    polys = torch.reshape(polys, [-1, 8])
    pt1, pt2, pt3, pt4 = polys[..., :8].chunk(4, 1)
    edge1 = torch.sqrt(
        torch.pow(pt1[..., 0] - pt2[..., 0], 2) +
        torch.pow(pt1[..., 1] - pt2[..., 1], 2))
    edge2 = torch.sqrt(
        torch.pow(pt2[..., 0] - pt3[..., 0], 2) +
        torch.pow(pt2[..., 1] - pt3[..., 1], 2))
    angles1 = torch.atan2((pt2[..., 1] - pt1[..., 1]),
                          (pt2[..., 0] - pt1[..., 0]))
    angles2 = torch.atan2((pt4[..., 1] - pt1[..., 1]),
                          (pt4[..., 0] - pt1[..., 0]))
    angles = polys.new_zeros(polys.shape[0])
    angles[edge1 > edge2] = angles1[edge1 > edge2]
    angles[edge1 <= edge2] = angles2[edge1 <= edge2]
    angles = norm_angle(angles, 'le135')
    x_ctr = (pt1[..., 0] + pt3[..., 0]) / 2.0
    y_ctr = (pt1[..., 1] + pt3[..., 1]) / 2.0
    edges = torch.stack([edge1, edge2], dim=1)
    width, _ = torch.max(edges, 1)
    height, _ = torch.min(edges, 1)
    return torch.stack([x_ctr, y_ctr, width, height, angles], 1)


def poly2obb_le90(polys):
    """Convert polygons to oriented bounding boxes.

    Args:
        polys (torch.Tensor): [x0,y0,x1,y1,x2,y2,x3,y3]

    Returns:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
    """
    polys = torch.reshape(polys, [-1, 8])
    pt1, pt2, pt3, pt4 = polys[..., :8].chunk(4, 1)
    edge1 = torch.sqrt(
        torch.pow(pt1[..., 0] - pt2[..., 0], 2) +
        torch.pow(pt1[..., 1] - pt2[..., 1], 2))
    edge2 = torch.sqrt(
        torch.pow(pt2[..., 0] - pt3[..., 0], 2) +
        torch.pow(pt2[..., 1] - pt3[..., 1], 2))
    angles1 = torch.atan2((pt2[..., 1] - pt1[..., 1]),
                          (pt2[..., 0] - pt1[..., 0]))
    angles2 = torch.atan2((pt4[..., 1] - pt1[..., 1]),
                          (pt4[..., 0] - pt1[..., 0]))
    angles = polys.new_zeros(polys.shape[0])
    angles[edge1 > edge2] = angles1[edge1 > edge2]
    angles[edge1 <= edge2] = angles2[edge1 <= edge2]
    angles = norm_angle(angles, 'le90')
    x_ctr = (pt1[..., 0] + pt3[..., 0]) / 2.0
    y_ctr = (pt1[..., 1] + pt3[..., 1]) / 2.0
    edges = torch.stack([edge1, edge2], dim=1)
    width, _ = torch.max(edges, 1)
    height, _ = torch.min(edges, 1)
    return torch.stack([x_ctr, y_ctr, width, height, angles], 1)


def poly2obb_np_oc(poly):
    """Convert polygons to oriented bounding boxes.

    Args:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3]

    Returns:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle]
    """
    bboxps = np.array(poly).reshape((4, 2))
    rbbox = cv2.minAreaRect(bboxps)
    x, y, w, h, a = rbbox[0][0], rbbox[0][1], rbbox[1][0], rbbox[1][1], rbbox[
        2]
    if w < 2 or h < 2:
        return
    while not 0 < a <= 90:
        if a == -90:
            a += 180
        else:
            a += 90
            w, h = h, w
    a = a / 180 * np.pi
    assert 0 < a <= np.pi / 2
    return x, y, w, h, a


def poly2obb_np_le135(poly):
    """Convert polygons to oriented bounding boxes.

    Args:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3]

    Returns:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle]
    """
    poly = np.array(poly[:8], dtype=np.float32)
    pt1 = (poly[0], poly[1])
    pt2 = (poly[2], poly[3])
    pt3 = (poly[4], poly[5])
    pt4 = (poly[6], poly[7])
    edge1 = np.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) + (pt1[1] - pt2[1]) *
                    (pt1[1] - pt2[1]))
    edge2 = np.sqrt((pt2[0] - pt3[0]) * (pt2[0] - pt3[0]) + (pt2[1] - pt3[1]) *
                    (pt2[1] - pt3[1]))
    if edge1 < 2 or edge2 < 2:
        return
    width = max(edge1, edge2)
    height = min(edge1, edge2)
    angle = 0
    if edge1 > edge2:
        angle = np.arctan2(float(pt2[1] - pt1[1]), float(pt2[0] - pt1[0]))
    elif edge2 >= edge1:
        angle = np.arctan2(float(pt4[1] - pt1[1]), float(pt4[0] - pt1[0]))
    angle = norm_angle(angle, 'le135')
    x_ctr = float(pt1[0] + pt3[0]) / 2
    y_ctr = float(pt1[1] + pt3[1]) / 2
    return x_ctr, y_ctr, width, height, angle



def cv2OriBox(cv_rect):
        #如图，如代码，角度定义为：“顺时针”与“X轴正向”最近的边的夹角，角度范围为[0,90]，而非(0,90]
        cv_width,cv_height,angle=cv_rect[1][0],cv_rect[1][1],cv_rect[2]
        if cv_width>cv_height:
            cv_width,cv_height=cv_height,cv_width
            angle+=90
        return np.array([cv_rect[0][0],cv_rect[0][1],cv_width,cv_height,angle/180*np.pi]) 
        
def poly_to_rotated_box(poly,label,refine_angle_with_label=[]):#这里转换为np使用cv2的方式获取旋转框
    """
    polys:n*8
    poly:[x0,y0,x1,y1,x2,y2,x3,y3]
    to
    rrect:[x_ctr,y_ctr,w,h,angle]
    """


    # poly = np.array(poly).reshape((-1, 2))
    rect = cv2.minAreaRect(poly.reshape(-1,2))
    outpoly=cv2OriBox(rect)
    if len(refine_angle_with_label)>0:
        if label in refine_angle_with_label:# 目标是 storage-tank or roundabout
            outpoly=np.array([outpoly[0],outpoly[1],(outpoly[2]+outpoly[3])/2,(outpoly[2]+outpoly[3])/2,0]) 
    return outpoly


def poly2obb_np_mask(poly):
    """Convert polygons to oriented bounding boxes.

    Args:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3]

    Returns:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle]
    """
    bboxps = np.array(poly).reshape((-1, 2))
    rbbox = cv2.minAreaRect(bboxps)
    x, y, w, h, a = rbbox[0][0], rbbox[0][1], rbbox[1][0], rbbox[1][1], rbbox[
        2]
    if w < 2 or h < 2:
        return
    a = a / 180 * np.pi
    if w < h:
        w, h = h, w
        a += np.pi / 2
    while not np.pi / 2 > a >= -np.pi / 2:
        if a >= np.pi / 2:
            a -= np.pi
        else:
            a += np.pi
    assert np.pi / 2 > a >= -np.pi / 2
    return x, y, w, h, a


def poly2obb_np_le90(poly):
    """Convert polygons to oriented bounding boxes.

    Args:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3]

    Returns:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle]
    """
    bboxps = np.array(poly).reshape((4, 2))
    rbbox = cv2.minAreaRect(bboxps)
    x, y, w, h, a = rbbox[0][0], rbbox[0][1], rbbox[1][0], rbbox[1][1], rbbox[
        2]
    if w < 2 or h < 2:
        return
    a = a / 180 * np.pi
    if w < h:
        w, h = h, w
        a += np.pi / 2
    while not np.pi / 2 > a >= -np.pi / 2:
        if a >= np.pi / 2:
            a -= np.pi
        else:
            a += np.pi
    assert np.pi / 2 > a >= -np.pi / 2
    return x, y, w, h, a


def obb2poly_oc(rboxes):
    """Convert oriented bounding boxes to polygons.

    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]

    Returns:
        polys (torch.Tensor): [x0,y0,x1,y1,x2,y2,x3,y3]
    """
    x = rboxes[:, 0]
    y = rboxes[:, 1]
    w = rboxes[:, 2]
    h = rboxes[:, 3]
    a = rboxes[:, 4]
    cosa = torch.cos(a)
    sina = torch.sin(a)
    wx, wy = w / 2 * cosa, w / 2 * sina
    hx, hy = -h / 2 * sina, h / 2 * cosa
    p1x, p1y = x - wx - hx, y - wy - hy
    p2x, p2y = x + wx - hx, y + wy - hy
    p3x, p3y = x + wx + hx, y + wy + hy
    p4x, p4y = x - wx + hx, y - wy + hy
    return torch.stack([p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y], dim=-1)


def obb2poly_le135(rboxes):
    """Convert oriented bounding boxes to polygons.

    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]

    Returns:
        polys (torch.Tensor): [x0,y0,x1,y1,x2,y2,x3,y3]
    """
    N = rboxes.shape[0]
    if N == 0:
        return rboxes.new_zeros((rboxes.size(0), 8))
    x_ctr, y_ctr, width, height, angle = rboxes.select(1, 0), rboxes.select(
        1, 1), rboxes.select(1, 2), rboxes.select(1, 3), rboxes.select(1, 4)
    tl_x, tl_y, br_x, br_y = \
        -width * 0.5, -height * 0.5, \
        width * 0.5, height * 0.5
    rects = torch.stack([tl_x, br_x, br_x, tl_x, tl_y, tl_y, br_y, br_y],
                        dim=0).reshape(2, 4, N).permute(2, 0, 1)
    sin, cos = torch.sin(angle), torch.cos(angle)
    M = torch.stack([cos, -sin, sin, cos], dim=0).reshape(2, 2,
                                                          N).permute(2, 0, 1)
    polys = M.matmul(rects).permute(2, 1, 0).reshape(-1, N).transpose(1, 0)
    polys[:, ::2] += x_ctr.unsqueeze(1)
    polys[:, 1::2] += y_ctr.unsqueeze(1)
    return polys.contiguous()


def obb2poly_le90(rboxes):
    """Convert oriented bounding boxes to polygons.

    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]

    Returns:
        polys (torch.Tensor): [x0,y0,x1,y1,x2,y2,x3,y3]
    """
    N = rboxes.shape[0]
    if N == 0:
        return rboxes.new_zeros((rboxes.size(0), 8))
    x_ctr, y_ctr, width, height, angle = rboxes.select(1, 0), rboxes.select(
        1, 1), rboxes.select(1, 2), rboxes.select(1, 3), rboxes.select(1, 4)
    tl_x, tl_y, br_x, br_y = \
        -width * 0.5, -height * 0.5, \
        width * 0.5, height * 0.5
    rects = torch.stack([tl_x, br_x, br_x, tl_x, tl_y, tl_y, br_y, br_y],
                        dim=0).reshape(2, 4, N).permute(2, 0, 1)
    sin, cos = torch.sin(angle), torch.cos(angle)
    M = torch.stack([cos, -sin, sin, cos], dim=0).reshape(2, 2,
                                                          N).permute(2, 0, 1)
    polys = M.matmul(rects).permute(2, 1, 0).reshape(-1, N).transpose(1, 0)
    polys[:, ::2] += x_ctr.unsqueeze(1)
    polys[:, 1::2] += y_ctr.unsqueeze(1)
    return polys.contiguous()


def obb2hbb_oc(rbboxes):
    """Convert oriented bounding boxes to horizontal bounding boxes.

    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]

    Returns:
        hbbs (torch.Tensor): [x_ctr,y_ctr,w,h,pi/2]
    """
    w = rbboxes[:, 2::5]
    h = rbboxes[:, 3::5]
    a = rbboxes[:, 4::5]
    cosa = torch.cos(a)
    sina = torch.sin(a)
    hbbox_w = cosa * w + sina * h
    hbbox_h = sina * w + cosa * h
    hbboxes = rbboxes.clone().detach()
    hbboxes[:, 2::5] = hbbox_h
    hbboxes[:, 3::5] = hbbox_w
    hbboxes[:, 4::5] = np.pi / 2
    return hbboxes


def obb2hbb_le135(rotatex_boxes):
    """Convert oriented bounding boxes to horizontal bounding boxes.

    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]

    Returns:
        hbbs (torch.Tensor): [x_ctr,y_ctr,w,h,-pi/2]
    """
    polys = obb2poly_le135(rotatex_boxes)
    xmin, _ = polys[:, ::2].min(1)
    ymin, _ = polys[:, 1::2].min(1)
    xmax, _ = polys[:, ::2].max(1)
    ymax, _ = polys[:, 1::2].max(1)
    bboxes = torch.stack([xmin, ymin, xmax, ymax], dim=1)
    x_ctr = (bboxes[:, 2] + bboxes[:, 0]) / 2.0
    y_ctr = (bboxes[:, 3] + bboxes[:, 1]) / 2.0
    edges1 = torch.abs(bboxes[:, 2] - bboxes[:, 0])
    edges2 = torch.abs(bboxes[:, 3] - bboxes[:, 1])
    angles = bboxes.new_zeros(bboxes.size(0))
    inds = edges1 < edges2
    rotated_boxes = torch.stack((x_ctr, y_ctr, edges1, edges2, angles), dim=1)
    rotated_boxes[inds, 2] = edges2[inds]
    rotated_boxes[inds, 3] = edges1[inds]
    rotated_boxes[inds, 4] = np.pi / 2.0
    return rotated_boxes


def obb2hbb_le90(obboxes):
    """Convert oriented bounding boxes to horizontal bounding boxes.

    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]

    Returns:
        hbbs (torch.Tensor): [x_ctr,y_ctr,w,h,-pi/2]
    """
    center, w, h, theta = torch.split(obboxes, [2, 1, 1, 1], dim=-1)
    Cos, Sin = torch.cos(theta), torch.sin(theta)
    x_bias = torch.abs(w / 2 * Cos) + torch.abs(h / 2 * Sin)
    y_bias = torch.abs(w / 2 * Sin) + torch.abs(h / 2 * Cos)
    bias = torch.cat([x_bias, y_bias], dim=-1)
    hbboxes = torch.cat([center - bias, center + bias], dim=-1)
    _x = (hbboxes[..., 0] + hbboxes[..., 2]) * 0.5
    _y = (hbboxes[..., 1] + hbboxes[..., 3]) * 0.5
    _w = hbboxes[..., 2] - hbboxes[..., 0]
    _h = hbboxes[..., 3] - hbboxes[..., 1]
    _theta = theta.new_zeros(theta.size(0))
    obboxes1 = torch.stack([_x, _y, _w, _h, _theta], dim=-1)
    obboxes2 = torch.stack([_x, _y, _h, _w, _theta - np.pi / 2], dim=-1)
    obboxes = torch.where((_w >= _h)[..., None], obboxes1, obboxes2)
    return obboxes


def hbb2obb_oc(hbboxes):
    """Convert horizontal bounding boxes to oriented bounding boxes.

    Args:
        hbbs (torch.Tensor): [x_lt,y_lt,x_rb,y_rb]

    Returns:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
    """
    x = (hbboxes[..., 0] + hbboxes[..., 2]) * 0.5
    y = (hbboxes[..., 1] + hbboxes[..., 3]) * 0.5
    w = hbboxes[..., 2] - hbboxes[..., 0]
    h = hbboxes[..., 3] - hbboxes[..., 1]
    theta = x.new_zeros(*x.shape)
    rbboxes = torch.stack([x, y, h, w, theta + np.pi / 2], dim=-1)
    return rbboxes


def hbb2obb_le135(hbboxes):
    """Convert horizontal bounding boxes to oriented bounding boxes.

    Args:
        hbbs (torch.Tensor): [x_lt,y_lt,x_rb,y_rb]

    Returns:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
    """
    x = (hbboxes[..., 0] + hbboxes[..., 2]) * 0.5
    y = (hbboxes[..., 1] + hbboxes[..., 3]) * 0.5
    w = hbboxes[..., 2] - hbboxes[..., 0]
    h = hbboxes[..., 3] - hbboxes[..., 1]
    theta = x.new_zeros(*x.shape)
    obboxes1 = torch.stack([x, y, w, h, theta], dim=-1)
    obboxes2 = torch.stack([x, y, h, w, theta + np.pi / 2], dim=-1)
    obboxes = torch.where((w >= h)[..., None], obboxes1, obboxes2)
    return obboxes


def hbb2obb_le90(hbboxes):
    """Convert horizontal bounding boxes to oriented bounding boxes.

    Args:
        hbbs (torch.Tensor): [x_lt,y_lt,x_rb,y_rb]

    Returns:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
    """
    x = (hbboxes[..., 0] + hbboxes[..., 2]) * 0.5
    y = (hbboxes[..., 1] + hbboxes[..., 3]) * 0.5
    w = hbboxes[..., 2] - hbboxes[..., 0]
    h = hbboxes[..., 3] - hbboxes[..., 1]
    theta = x.new_zeros(*x.shape)
    obboxes1 = torch.stack([x, y, w, h, theta], dim=-1)
    obboxes2 = torch.stack([x, y, h, w, theta - np.pi / 2], dim=-1)
    obboxes = torch.where((w >= h)[..., None], obboxes1, obboxes2)
    return obboxes


def obb2xyxy_oc(rbboxes):
    """Convert oriented bounding boxes to horizontal bounding boxes.

    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]

    Returns:
        hbbs (torch.Tensor): [x_lt,y_lt,x_rb,y_rb]
    """
    w = rbboxes[:, 2::5]
    h = rbboxes[:, 3::5]
    a = rbboxes[:, 4::5]
    cosa = torch.cos(a)
    sina = torch.sin(a)
    hbbox_w = cosa * w + sina * h
    hbbox_h = sina * w + cosa * h
    # pi/2 >= a > 0, so cos(a)>0, sin(a)>0
    dx = rbboxes[..., 0]
    dy = rbboxes[..., 1]
    dw = hbbox_w.reshape(-1)
    dh = hbbox_h.reshape(-1)
    x1 = dx - dw / 2
    y1 = dy - dh / 2
    x2 = dx + dw / 2
    y2 = dy + dh / 2
    return torch.stack((x1, y1, x2, y2), -1)


def obb2xyxy_le135(rotatex_boxes):
    """Convert oriented bounding boxes to horizontal bounding boxes.

    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]

    Returns:
        hbbs (torch.Tensor): [x_lt,y_lt,x_rb,y_rb]
    """
    N = rotatex_boxes.shape[0]
    if N == 0:
        return rotatex_boxes.new_zeros((rotatex_boxes.size(0), 4))
    polys = obb2poly_le135(rotatex_boxes)
    xmin, _ = polys[:, ::2].min(1)
    ymin, _ = polys[:, 1::2].min(1)
    xmax, _ = polys[:, ::2].max(1)
    ymax, _ = polys[:, 1::2].max(1)
    return torch.stack([xmin, ymin, xmax, ymax], dim=1)


def obb2xyxy_le90(obboxes):
    """Convert oriented bounding boxes to horizontal bounding boxes.

    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]

    Returns:
        hbbs (torch.Tensor): [x_lt,y_lt,x_rb,y_rb]
    """
    # N = obboxes.shape[0]
    # if N == 0:
    #     return obboxes.new_zeros((obboxes.size(0), 4))
    center, w, h, theta = torch.split(obboxes, [2, 1, 1, 1], dim=-1)
    Cos, Sin = torch.cos(theta), torch.sin(theta)
    x_bias = torch.abs(w / 2 * Cos) + torch.abs(h / 2 * Sin)
    y_bias = torch.abs(w / 2 * Sin) + torch.abs(h / 2 * Cos)
    bias = torch.cat([x_bias, y_bias], dim=-1)
    return torch.cat([center - bias, center + bias], dim=-1)


def obb2poly_np_oc(rbboxes):
    """Convert oriented bounding boxes to polygons.

    Args:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle,score]

    Returns:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3,score]
    """
    x = rbboxes[:, 0]
    y = rbboxes[:, 1]
    w = rbboxes[:, 2]
    h = rbboxes[:, 3]
    a = rbboxes[:, 4]
    score = rbboxes[:, 5]
    cosa = np.cos(a)
    sina = np.sin(a)
    wx, wy = w / 2 * cosa, w / 2 * sina
    hx, hy = -h / 2 * sina, h / 2 * cosa
    p1x, p1y = x - wx - hx, y - wy - hy
    p2x, p2y = x + wx - hx, y + wy - hy
    p3x, p3y = x + wx + hx, y + wy + hy
    p4x, p4y = x - wx + hx, y - wy + hy
    polys = np.stack([p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y, score], axis=-1)
    polys = get_best_begin_point(polys)
    return polys


def obb2poly_np_le135(rrects):
    """Convert oriented bounding boxes to polygons.

    Args:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle,score]

    Returns:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3,score]
    """
    polys = []
    for rrect in rrects:
        x_ctr, y_ctr, width, height, angle, score = rrect[:6]
        tl_x, tl_y, br_x, br_y = -width / 2, -height / 2, width / 2, height / 2
        rect = np.array([[tl_x, br_x, br_x, tl_x], [tl_y, tl_y, br_y, br_y]])
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]])
        poly = R.dot(rect)
        x0, x1, x2, x3 = poly[0, :4] + x_ctr
        y0, y1, y2, y3 = poly[1, :4] + y_ctr
        poly = np.array([x0, y0, x1, y1, x2, y2, x3, y3, score],
                        dtype=np.float32)
        polys.append(poly)
    polys = np.array(polys)
    polys = get_best_begin_point(polys)
    return polys


def obb2poly_np_le90(obboxes):
    """Convert oriented bounding boxes to polygons.

    Args:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle,score]

    Returns:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3,score]
    """
    try:
        center, w, h, theta, score = np.split(obboxes, (2, 3, 4, 5), axis=-1)
    except:  # noqa: E722
        results = np.stack([0., 0., 0., 0., 0., 0., 0., 0., 0.], axis=-1)
        return results.reshape(1, -1)
    Cos, Sin = np.cos(theta), np.sin(theta)
    vector1 = np.concatenate([w / 2 * Cos, w / 2 * Sin], axis=-1)
    vector2 = np.concatenate([-h / 2 * Sin, h / 2 * Cos], axis=-1)
    point1 = center - vector1 - vector2
    point2 = center + vector1 - vector2
    point3 = center + vector1 + vector2
    point4 = center - vector1 + vector2
    polys = np.concatenate([point1, point2, point3, point4, score], axis=-1)
    polys = get_best_begin_point(polys)
    return polys


def cal_line_length(point1, point2):
    """Calculate the length of line.

    Args:
        point1 (List): [x,y]
        point2 (List): [x,y]

    Returns:
        length (float)
    """
    return math.sqrt(
        math.pow(point1[0] - point2[0], 2) +
        math.pow(point1[1] - point2[1], 2))


def get_best_begin_point_single(coordinate):
    """Get the best begin point of the single polygon.

    Args:
        coordinate (List): [x1, y1, x2, y2, x3, y3, x4, y4, score]

    Returns:
        reorder coordinate (List): [x1, y1, x2, y2, x3, y3, x4, y4, score]
    """
    x1, y1, x2, y2, x3, y3, x4, y4, score = coordinate
    xmin = min(x1, x2, x3, x4)
    ymin = min(y1, y2, y3, y4)
    xmax = max(x1, x2, x3, x4)
    ymax = max(y1, y2, y3, y4)
    combine = [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
               [[x2, y2], [x3, y3], [x4, y4], [x1, y1]],
               [[x3, y3], [x4, y4], [x1, y1], [x2, y2]],
               [[x4, y4], [x1, y1], [x2, y2], [x3, y3]]]
    dst_coordinate = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
    force = 100000000.0
    force_flag = 0
    for i in range(4):
        temp_force = cal_line_length(combine[i][0], dst_coordinate[0]) \
                     + cal_line_length(combine[i][1], dst_coordinate[1]) \
                     + cal_line_length(combine[i][2], dst_coordinate[2]) \
                     + cal_line_length(combine[i][3], dst_coordinate[3])
        if temp_force < force:
            force = temp_force
            force_flag = i
    if force_flag != 0:
        pass
    return np.hstack(
        (np.array(combine[force_flag]).reshape(8), np.array(score)))


def get_best_begin_point(coordinates):
    """Get the best begin points of polygons.

    Args:
        coordinate (ndarray): shape(n, 9).

    Returns:
        reorder coordinate (ndarray): shape(n, 9).
    """
    coordinates = list(map(get_best_begin_point_single, coordinates.tolist()))
    coordinates = np.array(coordinates)
    return coordinates


def norm_angle(angle, angle_range):
    """Limit the range of angles.

    Args:
        angle (ndarray): shape(n, ).
        angle_range (Str): angle representations.

    Returns:
        angle (ndarray): shape(n, ).
    """
    if angle_range == 'oc':
        return angle
    elif angle_range == 'le135':
        return (angle + np.pi / 4) % np.pi - np.pi / 4
    elif angle_range == 'le90':
        return (angle + np.pi / 2) % np.pi - np.pi / 2
    else:
        print('Not yet implemented.')


def dist_torch(point1, point2):
    """Calculate the distance between two points.

    Args:
        point1 (torch.Tensor): shape(n, 2).
        point2 (torch.Tensor): shape(n, 2).

    Returns:
        distance (torch.Tensor): shape(n, 1).
    """
    return torch.norm(point1 - point2, dim=-1)


def distance2rbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    assert(distance.shape[1]==5) 

    x1 = points[:, 0:1] - distance[:, 0:1]
    y1 = points[:, 1:2] - distance[:, 1:2]
    return torch.cat([x1, y1, distance[...,2:]],1)
# def distance2rbox(points, distance, max_shape=None):
#     """Decode distance prediction to bounding box.

#     Args:
#         points (Tensor): Shape (n, 2), [x, y].
#         distance (Tensor): Distance from the given point to 4
#             boundaries (left, top, right, bottom).
#         max_shape (tuple): Shape of the image.

#     Returns:
#         Tensor: Decoded bboxes.
#     """
#     assert(distance.shape[1]==5) 

#     x1 = points[:, 0:1] + distance[:, 0:1]
#     y1 = points[:, 1:2] + distance[:, 1:2]
#     return torch.cat([x1, y1, distance[...,2:]],1)

def distance2fourier(points, distance, fourier_degree=10):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    regchannel = 2*(2*fourier_degree+1)
    assert(distance.shape[1]==regchannel) 
    
    distance = distance.reshape(distance.shape[0],-1,2)
    # result = torch.zeros_like(distance)
    # distance[:,fourier_degree:fourier_degree+1]=distance[:,fourier_degree:fourier_degree+1]+points[:, None,:] 
    distance[:,fourier_degree:fourier_degree+1,0]=distance[:,fourier_degree:fourier_degree+1,0]+points[:, 0:1] 
    distance[:,fourier_degree:fourier_degree+1,1]=distance[:,fourier_degree:fourier_degree+1,1]+points[:, 1:2] 
    return distance.reshape(-1,regchannel)


def distance2points(points, distance):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    distance = distance.reshape(distance.shape[0],-1,2)
    # distance[:,fourier_degree:fourier_degree+1]=distance[:,fourier_degree:fourier_degree+1]+points[:, None,:] 
    distance[:,:,0]=distance[:,:,0]+points[:, 0:1] 
    distance[:,:,1]=distance[:,:,1]+points[:, 1:2] 
    return distance

def poly2obb_np_mask(poly):
    """Convert polygons to oriented bounding boxes.

    Args:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3]

    Returns:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle]
    """
    bboxps = np.array(poly,dtype = np.int).reshape((-1, 2))
    rbbox = cv2.minAreaRect(bboxps)
    x, y, w, h, a = rbbox[0][0], rbbox[0][1], rbbox[1][0], rbbox[1][1], rbbox[
        2]
    # if w < 2 or h < 2:
    #     return
    a = a / 180 * np.pi
    if w < h:
        w, h = h, w
        a += np.pi / 2
    while not np.pi / 2 > a >= -np.pi / 2:
        if a >= np.pi / 2:
            a -= np.pi
        else:
            a += np.pi
    assert np.pi / 2 > a >= -np.pi / 2
    return [x, y, w, h, a]


def rotated_box_to_poly(rboxes):
    """
    rrect:[x_ctr,y_ctr,w,h,angle]
    to
    poly:[x0,y0,x1,y1,x2,y2,x3,y3]
    """
    N = rboxes.shape[0]
    x_ctr, y_ctr, width, height, angle = rboxes.select(1, 0), rboxes.select(
        1, 1), rboxes.select(1, 2), rboxes.select(1, 3), rboxes.select(1, 4)
    tl_x, tl_y, br_x, br_y = -width * 0.5, -height * 0.5, width * 0.5, height * 0.5

    rects = torch.stack([tl_x, br_x, br_x, tl_x, tl_y, tl_y,
                         br_y, br_y], dim=0).reshape(2, 4, N).permute(2, 0, 1)

    sin, cos = torch.sin(angle), torch.cos(angle)
    # M.shape=[N,2,2]
    M = torch.stack([cos, -sin, sin, cos],
                    dim=0).reshape(2, 2, N).permute(2, 0, 1)
    # polys:[N,8]
    polys = M.matmul(rects).permute(2, 1, 0).reshape(-1, N).transpose(1, 0)
    polys[:, ::2] += x_ctr.unsqueeze(1)
    polys[:, 1::2] += y_ctr.unsqueeze(1)

    return polys


def rotated_box_to_poly_np(rrects):
    """
    rrect:[x_ctr,y_ctr,w,h,angle]
    to
    poly:[x0,y0,x1,y1,x2,y2,x3,y3]
    """
    polys = []
    for rrect in rrects:
        x_ctr, y_ctr, width, height, angle = rrect[:5]
        tl_x, tl_y, br_x, br_y = -width / 2, -height / 2, width / 2, height / 2
        rect = np.array([[tl_x, br_x, br_x, tl_x], [tl_y, tl_y, br_y, br_y]])
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]])
        poly = R.dot(rect)
        x0, x1, x2, x3 = poly[0, :4] + x_ctr
        y0, y1, y2, y3 = poly[1, :4] + y_ctr
        poly = np.array([x0, y0, x1, y1, x2, y2, x3, y3], dtype=np.float32)
        polys.append(poly)
    polys = np.array(polys)
    # polys = get_best_begin_point(polys)
    return polys


def rotated_box_to_bbox_np(rotatex_boxes):
    polys = rotated_box_to_poly_np(rotatex_boxes)
    xmin = polys[:, ::2].min(1, keepdims=True)
    ymin = polys[:, 1::2].min(1, keepdims=True)
    xmax = polys[:, ::2].max(1, keepdims=True)
    ymax = polys[:, 1::2].max(1, keepdims=True)
    return np.concatenate([xmin, ymin, xmax, ymax], axis=1)

def rotated_box_to_poly(rboxes):
    """
    rrect:[x_ctr,y_ctr,w,h,angle]
    to
    poly:[x0,y0,x1,y1,x2,y2,x3,y3]
    """
    N = rboxes.shape[0]
    x_ctr, y_ctr, width, height, angle = rboxes.select(1, 0), rboxes.select(
        1, 1), rboxes.select(1, 2), rboxes.select(1, 3), rboxes.select(1, 4)
    tl_x, tl_y, br_x, br_y = -width * 0.5, -height * 0.5, width * 0.5, height * 0.5

    rects = torch.stack([tl_x, br_x, br_x, tl_x, tl_y, tl_y,
                         br_y, br_y], dim=0).reshape(2, 4, N).permute(2, 0, 1)

    sin, cos = torch.sin(angle), torch.cos(angle)
    # M.shape=[N,2,2]
    M = torch.stack([cos, -sin, sin, cos],
                    dim=0).reshape(2, 2, N).permute(2, 0, 1)
    # polys:[N,8]
    polys = M.matmul(rects).permute(2, 1, 0).reshape(-1, N).transpose(1, 0)
    polys[:, ::2] += x_ctr.unsqueeze(1)
    polys[:, 1::2] += y_ctr.unsqueeze(1)

    return polys


def rotated_box_to_bbox(rotatex_boxes):
    polys = rotated_box_to_poly(rotatex_boxes)
    xmin, _ = polys[:, ::2].min(1)
    ymin, _ = polys[:, 1::2].min(1)
    xmax, _ = polys[:, ::2].max(1)
    ymax, _ = polys[:, 1::2].max(1)
    return torch.stack([xmin, ymin, xmax, ymax], dim=1)

def fourier2poly( real_maps, imag_maps,fourier_degree,num_sample):
        """Transform Fourier coefficient maps to polygon maps.

        Args:
            real_maps (tensor): A map composed of the real parts of the
                Fourier coefficients, whose shape is (-1, 2k+1)
            imag_maps (tensor):A map composed of the imag parts of the
                Fourier coefficients, whose shape is (-1, 2k+1)

        Returns
            x_maps (tensor): A map composed of the x value of the polygon
                represented by n sample points (xn, yn), whose shape is (-1, n)
            y_maps (tensor): A map composed of the y value of the polygon
                represented by n sample points (xn, yn), whose shape is (-1, n)
        """

        device = real_maps.device
        k_vect = torch.arange(
            -fourier_degree,
            fourier_degree + 1,
            dtype=torch.float,
            device=device).view(-1, 1)
        i_vect = torch.arange(
            0, num_sample, dtype=torch.float, device=device).view(1, -1)

        transform_matrix = 2 * np.pi / num_sample * torch.mm(
            k_vect, i_vect)

        x1 = torch.einsum('ak, kn-> an', real_maps,
                          torch.cos(transform_matrix))
        x2 = torch.einsum('ak, kn-> an', imag_maps,
                          torch.sin(transform_matrix))
        y1 = torch.einsum('ak, kn-> an', real_maps,
                          torch.sin(transform_matrix))
        y2 = torch.einsum('ak, kn-> an', imag_maps,
                          torch.cos(transform_matrix))

        x_maps = x1 - x2
        y_maps = y1 + y2

        return x_maps, y_maps
    
def fourier_to_bbox(mask,fourier_degree,num_sample):
    if mask.shape[0]>0:
        regchannel = 2*(2*fourier_degree+1)
        assert(mask.shape[1]==regchannel) 
        mask = mask.reshape(mask.shape[0],-1,2)
        realpart = mask[:,:,0]
        imgpart = mask[:,:,1]
        # 2、傅里叶反变换 显示反变换mask
        i_fft = fourier2poly(realpart, imgpart,fourier_degree,num_sample)
        polys = torch.cat([i_fft[0][:,:,None],i_fft[1][:,:,None]],dim=2)
    
        xmin, _ = polys[:,:,0].min(1)
        ymin, _ = polys[:, :,1].min(1)
        xmax, _ = polys[:,:,0].max(1)
        ymax, _ = polys[:, :,1].max(1)
        return torch.stack((xmin, ymin, xmax, ymax), dim=1)
    else:
        return  torch.zeros((0,4),dtype =mask.dtype ).to( mask.device)

def fourier_to_ploy(mask,fourier_degree,num_sample):
    if mask.shape[0]>0:
        regchannel = 2*(2*fourier_degree+1)
        # assert(mask.shape[1]==regchannel) 
        
        mask = mask.reshape(mask.shape[0],-1,2)
        realpart = mask[:,:,0]
        imgpart = mask[:,:,1]
        # 2、傅里叶反变换 显示反变换mask
        i_fft = fourier2poly(realpart, imgpart,fourier_degree,num_sample)
        polys = torch.cat([i_fft[0][:,:,None],i_fft[1][:,:,None]],dim=2)
    else:
        polys = torch.zeros((0,num_sample,2),dtype =mask.dtype ).to( mask.device)
    return polys


def bbox_poly2result(bboxes, polygons, labels, num_classes, num_contour_points):
    if bboxes.shape[0] == 0:
        return [[np.zeros((0, 5), dtype=np.float32) for i in range(num_classes)],
                [np.zeros((0, num_contour_points*2), dtype=np.float32) for i in range(num_classes)]]
    else:
        bboxes = bboxes.cpu().numpy()
        labels = labels.cpu().numpy()
        polygons = polygons.permute(0, 2, 1).reshape((polygons.shape[0],-1))#TODO insert
        polygons = polygons.cpu().numpy()
        return [[bboxes[labels == i, :] for i in range(num_classes)],
                [polygons[labels == i, :] for i in range(num_classes)]]
        

def encode_poly_results(mask_results, img_h, img_w):
    if isinstance(mask_results, tuple):  # mask scoring
        cls_segms, cls_mask_scores = mask_results
    else:
        cls_segms = mask_results
    num_classes = len(cls_segms)
    encoded_mask_results = [[] for _ in range(num_classes)]
    for i in range(len(cls_segms)):
        for cls_segm in cls_segms[i]:
            encoded_mask_results[i].append(
                get_rle(cls_segm, img_h, img_w)
                )  # encoded with RLE
    if isinstance(mask_results, tuple):
        return encoded_mask_results, cls_mask_scores
    else:
        return encoded_mask_results   
    
def get_rle(cls_segm, img_h, img_w):
    cls_segm = [cls_segm.copy().tolist()]
    rles = mask_util.frPyObjects(cls_segm, img_h, img_w)
    return mask_util.merge(rles)

def ploys2rboxes(i_fft):
        #  i_fft = torch.from_numpy(i_fft_np)
        if i_fft.shape[0]>0:
            ct = i_fft.mean(1)
            diff = i_fft-ct[:,None]
            M11 =(diff[:,:,0]* diff[:,:,1]).sum(1)
            M02 = (diff[:,:,1]* diff[:,:,1]).sum(1)
            M20 =(diff[:,:,0]* diff[:,:,0]).sum(1)
            thea = torch.atan2(2*M11,M20-M02)/2
            tanthe = torch.tan(thea)[:,None]
            h_dist = torch.abs(tanthe*i_fft[:,:,0] - i_fft[:,:,1]-tanthe*ct[:,0:1]+ct[:,1:2])/torch.sqrt(1+ tanthe*tanthe)
            h_dist =h_dist.max(1)[0]*2
            w_dist = torch.abs(i_fft[:,:,0] + tanthe*i_fft[:,:,1] - ct[:,0:1] - tanthe * ct[:,1:2])/torch.sqrt(1+ tanthe*tanthe)
            w_dist = w_dist.max(1)[0]*2
            rboxes = torch.cat((ct,w_dist[:,None],h_dist[:,None] ,thea[:,None]),dim=1)#.numpy()
        else:
            rboxes = i_fft[:,0:5,0]
        return rboxes