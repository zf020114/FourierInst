# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import numpy as np
from mmdet.datasets.builder import PIPELINES
from numpy.fft import fft
from numpy.linalg import norm
import torch
# import mmocr.utils.check_argument as check_argument
# from mmocr.datasets.pipelines.textdet_targets import TextSnakeTargets
from mmdet.core.bbox.transforms_rotate import  poly_to_rotated_box,obb2poly_oc,rotated_box_to_poly_np,ploys2rboxes
# from torch.utils.data import Dataset
from .base_textdet_targets import BaseTextDetTargets
import os
import mmcv, math
from mmdet.core import BitmapMasks, PolygonMasks
from mmdet.core.bbox.transforms_rotate import fourier_to_ploy
from matplotlib.patches import Polygon
from matplotlib.transforms import Affine2D

@PIPELINES.register_module()
class FCETargets(BaseTextDetTargets):
    """Generate the ground truth targets of FCENet: Fourier Contour Embedding
    for Arbitrary-Shaped Text Detection.

    [https://arxiv.org/abs/2104.10442]

    Args:
        fourier_degree (int): The maximum Fourier transform degree k.
        resample_step (float): The step size for resampling the text center
            line (TCL). It's better not to exceed half of the minimum width.
        center_region_shrink_ratio (float): The shrink ratio of text center
            region.
        level_size_divisors (tuple(int)): The downsample ratio on each level.
        level_proportion_range (tuple(tuple(int))): The range of text sizes
            assigned to each level.
    """

    def __init__(self,
                 fourier_degree=10,
                 num_reconstr_points=256,
                 debug = False,
                 circular_class=[],
                 show_dir=None
                 ):
        super().__init__()
        self.fourier_degree = fourier_degree
        self.num_sample = num_reconstr_points
        self.debug = debug
        self.circular_class = circular_class
        self.show_dir=show_dir
        
    def normalize_polygon(self, polygon):
        """Normalize one polygon so that its start point is at right most.

        Args:
            polygon (list[float]): The origin polygon.
        Returns:
            new_polygon (lost[float]): The polygon with start point at right.
        """
        temp_polygon = polygon - polygon.mean(axis=0)
        x = np.abs(temp_polygon[:, 0])
        y = temp_polygon[:, 1]
        index_x = np.argsort(x)
        index_y = np.argmin(y[index_x[:8]])
        index = index_x[index_y]
        new_polygon = np.concatenate([polygon[index:], polygon[:index]])
        return new_polygon

    def poly2fourier(self, polygon, fourier_degree):
        """Perform Fourier transformation to generate Fourier coefficients ck
        from polygon.

        Args:
            polygon (ndarray): An input polygon.
            fourier_degree (int): The maximum Fourier degree K.
        Returns:
            c (ndarray(complex)): Fourier coefficients.
        """
        points = polygon[:, 0] + polygon[:, 1] * 1j
        c_fft = fft(points) / len(points)
        c = np.hstack((c_fft[-fourier_degree:], c_fft[:fourier_degree + 1]))
        return c
    
    def clockwise(self, c, fourier_degree):
        """Make sure the polygon reconstructed from Fourier coefficients c in
        the clockwise direction.

        Args:
            polygon (list[float]): The origin polygon.
        Returns:
            new_polygon (lost[float]): The polygon in clockwise point order.
        """
        if np.abs(c[fourier_degree + 1]) > np.abs(c[fourier_degree - 1]):
            return c
        elif np.abs(c[fourier_degree + 1]) < np.abs(c[fourier_degree - 1]):
            return c[::-1]
        else:
            if np.abs(c[fourier_degree + 2]) > np.abs(c[fourier_degree - 2]):
                return c
            else:
                return c[::-1]

    def cal_fourier_signature(self, polygon, fourier_degree):
        """Calculate Fourier signature from input polygon.

        Args:
              polygon (ndarray): The input polygon.
              fourier_degree (int): The maximum Fourier degree K.
        Returns:
              fourier_signature (ndarray): An array shaped (2k+1, 2) containing
                  real part and image part of 2k+1 Fourier coefficients.
        """
        assert len(polygon.shape)==2
        resampled_polygon = cv2.resize(np.array(polygon,dtype=np.float32),(2,self.num_sample))#归一化点的数量
        resampled_polygon = self.normalize_polygon(resampled_polygon)
        fourier_coeff = self.poly2fourier(resampled_polygon, fourier_degree)
        fourier_coeff = self.clockwise(fourier_coeff, fourier_degree)

        real_part = np.real(fourier_coeff).reshape((-1, 1))
        image_part = np.imag(fourier_coeff).reshape((-1, 1))
        fourier_signature = np.hstack([real_part, image_part])
        return fourier_signature

    def fourier2poly(self, real_maps, imag_maps):
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
            -self.fourier_degree,
            self.fourier_degree + 1,
            dtype=torch.float,
            device=device).view(-1, 1)
        i_vect = torch.arange(
            0, self.num_sample, dtype=torch.float, device=device).view(1, -1)

        transform_matrix = 2 * np.pi / self.num_sample * torch.mm(
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
    
    def generate_fourier_maps(self, img_size, text_polys,gt_labels):
        """Generate Fourier coefficient maps.

        Args:
            img_size (tuple): The image size of (height, width).
            text_polys (list[list[ndarray]]): The list of text polygons.

        Returns:
            fourier_real_map (ndarray): The Fourier coefficient real part maps.
            fourier_image_map (ndarray): The Fourier coefficient image part
                maps.
        """
        assert isinstance(img_size, tuple)
        # assert check_argument.is_2dlist(text_polys)
        assert len(text_polys)==len(gt_labels)
        h, w = img_size
        k = self.fourier_degree
        fourier_coeff=[]
        text_region_mask = np.zeros((h, w), dtype=np.uint8)
        gt_rboxes, gt_bboxes=[], []
        for ind, poly in enumerate( text_polys):
            #处理一个图形有多个轮廓的问题，就是取面积最大的那个轮廓
            if  len(poly) >1:
                areas=[]
                for pol in poly:
                    text_instance = pol.reshape(-1,2)
                    polygon = np.array(text_instance,dtype=np.int).reshape((1, -1, 2))
                    rbbox = cv2.minAreaRect(polygon)
                    w, h = rbbox[1][0], rbbox[1][1]
                    areas.append( w*h)
                max_ind = np.argmax(np.array(areas))
                poly = [poly[max_ind]]
            assert len(poly) == 1
            poly = np.array(poly,dtype = np.int)
            gt_rboxes.append(poly_to_rotated_box(poly,self.circular_class))
            poly = np.array(poly).reshape((-1, 2))
            box =np.hstack([poly[:,0].min(),poly[:,1].min(),poly[:,0].max(),poly[:,1].max(),]) 
            gt_bboxes.append(box)

            text_instance = poly.reshape(-1,2)
            polygon = np.array(text_instance).reshape((1, -1, 2))
            class_index = np.int(gt_labels[ind]+1)# 这里要从1开始，如果是0无法和背景区分
            cv2.fillPoly(text_region_mask, polygon, class_index)

            fourier_coeff.append( self.cal_fourier_signature(polygon[0], k))
        return np.array(fourier_coeff), text_region_mask, np.array(gt_rboxes), np.array(gt_bboxes)

    def generate_targets(self, results):
        """Generate the ground truth targets for FCENet.

        Args:
            results (dict): The input result dictionary.

        Returns:
            results (dict): The output result dictionary.
        """

        assert isinstance(results, dict)
        polygon_masks = results['gt_masks'].masks
        gt_labels =  results['gt_labels']
        h, w, _ = results['img_shape']
        # assert len(polygon_masks) >0
        if  len(polygon_masks) >0:
            lv_text_polys = polygon_masks
            level_img_size = (h , w )
            fourier_coeff , text_region, gt_rboxes, gt_bboxes= self.generate_fourier_maps(
                level_img_size, lv_text_polys,gt_labels)
            # try:
            gt_masks = np.concatenate((gt_rboxes, fourier_coeff.reshape(fourier_coeff.shape[0],-1)),axis = 1)
            # except:
            #     print(gt_rboxes)
            results['gt_masks'] =  gt_masks
            results['gt_bboxes'] =np.array(gt_bboxes,dtype = np.float32)
            results['gt_semantic_seg']  =text_region
        else:
            results['gt_masks'] =  np.zeros((0,47))
            results['gt_bboxes'] =np.zeros((0,4))
            results['gt_semantic_seg'] =np.zeros((0,4))
            ##TODO 以下是调试代码
        if self.debug:
            #1、显示mask
            img_name = results['img_info']['file_name']
            if self.show_dir is not None:
                out_put_path = self.show_dir 
            else:
                out_put_path = os.path.join(os.path.dirname(os.path.dirname(  results['img_prefix'])),'show_four')
                output_name = os.path.join(out_put_path, img_name)
            if not os.path.exists(out_put_path):
                os.makedirs(out_put_path)
            # cv2.imwrite(output_name,text_region)
            
            fourier_coeff = torch.tensor(torch.from_numpy(
                                                            fourier_coeff),dtype = torch.float32)
            i_fft_np = fourier_to_ploy(fourier_coeff,self.fourier_degree,self.num_sample).numpy()
            # realpart = fourier_coeff_tensor[:,:,0]
            # imgpart = fourier_coeff_tensor[:,:,1]
            # # 2、傅里叶反变换 显示反变换mask
            # i_fft = self.fourier2poly(realpart, imgpart)
            # i_fft_np = torch.cat([i_fft[0][:,:,None],i_fft[1][:,:,None]],dim=2).numpy()
            mask = np.zeros((h, w), dtype=np.uint8)
            for i in range(i_fft_np.shape[0]):
                text_instance =i_fft_np[i]
                polygon = np.array(text_instance).reshape((1, -1, 2))
                cv2.fillPoly(mask, polygon.astype(np.int32), np.int(gt_labels[i]+1))
            fout_name = os.path.splitext(img_name)[0]+'_f.png'
            output_name = os.path.join(out_put_path, fout_name)
            cv2.imwrite(output_name,mask)
            
        # ##TODO 以下是调试代码
        # if self.debug:
        #     #1、显示mask
        #     img_name = results['img_info']['file_name']
        #     if self.show_dir is not None:
        #         out_put_path = self.show_dir 
        #     else:
        #         out_put_path = os.path.join(os.path.dirname(os.path.dirname(  results['img_prefix'])),'show_four')
        #         output_name = os.path.join(out_put_path, img_name)
        #     if not os.path.exists(out_put_path):
        #         os.makedirs(out_put_path)
        #     cv2.imwrite(output_name,text_region)
        #     fourier_coeff_tensor = torch.tensor(torch.from_numpy(
        #                                                     fourier_coeff),dtype = torch.float32)
        #     realpart = fourier_coeff_tensor[:,:,0]
        #     imgpart = fourier_coeff_tensor[:,:,1]
        #     # 2、傅里叶反变换 显示反变换mask
        #     i_fft = self.fourier2poly(realpart, imgpart)
        #     i_fft_np = torch.cat([i_fft[0][:,:,None],i_fft[1][:,:,None]],dim=2).numpy()
            
        #     #下面使用矩阵方法将poly转换为rbox
        #     # ct = np.array(i_fft_np).mean(1)
        #     # diff = i_fft_np-ct[:,None]
        #     # M11 =(diff[:,:,0]* diff[:,:,1]).sum(1)
        #     # M02 = (diff[:,:,1]* diff[:,:,1]).sum(1)
        #     # M20 =(diff[:,:,0]* diff[:,:,0]).sum(1)
        #     # thea = np.arctan2(2*M11,M20-M02)/2
        #     # tanthe = np.tan(thea)[:,None]
        #     # h_dist = np.abs(tanthe*i_fft_np[:,:,0] - i_fft_np[:,:,1]-tanthe*ct[:,0:1]+ct[:,1:2])/np.sqrt(1+ tanthe*tanthe)
        #     # h_dist =h_dist.max(1)*2
        #     # w_dist = np.abs(i_fft_np[:,:,0] + tanthe*i_fft_np[:,:,1] - ct[:,0:1] - tanthe * ct[:,1:2])/np.sqrt(1+ tanthe*tanthe)
        #     # w_dist = w_dist.max(1)*2
        #     # rboxes1 = np.hstack((ct,w_dist[:,None],h_dist[:,None] ,thea[:,None]))
            
        #     #下面是使用torch 进行polys到rbox变换
        #     # i_fft = torch.from_numpy(i_fft_np)
        #     # ct = i_fft.mean(1)
        #     # diff = i_fft-ct[:,None]
        #     # M11 =(diff[:,:,0]* diff[:,:,1]).sum(1)
        #     # M02 = (diff[:,:,1]* diff[:,:,1]).sum(1)
        #     # M20 =(diff[:,:,0]* diff[:,:,0]).sum(1)
        #     # thea = torch.atan2(2*M11,M20-M02)/2
        #     # tanthe = torch.tan(thea)[:,None]
        #     # h_dist = torch.abs(tanthe*i_fft[:,:,0] - i_fft[:,:,1]-tanthe*ct[:,0:1]+ct[:,1:2])/np.sqrt(1+ tanthe*tanthe)
        #     # h_dist =h_dist.max(1)[0]*2
        #     # w_dist = torch.abs(i_fft[:,:,0] + tanthe*i_fft[:,:,1] - ct[:,0:1] - tanthe * ct[:,1:2])/np.sqrt(1+ tanthe*tanthe)
        #     # w_dist = w_dist.max(1)[0]*2
        #     # rboxes = torch.cat((ct,w_dist[:,None],h_dist[:,None] ,thea[:,None]),dim=1).numpy()
        #     rboxes = ploys2rboxes(torch.from_numpy(i_fft_np)).numpy()
            
        #     mask = np.zeros((h, w), dtype=np.uint8)
        #     for i in range(i_fft_np.shape[0]):
        #         text_instance =i_fft_np[i]
        #         polygon = np.array(text_instance).reshape((1, -1, 2))
        #         cv2.fillPoly(mask, polygon.astype(np.int32), np.int(gt_labels[i]+1))

        #         # ###这是采用主成分分析的方法
        #         # Z1 = text_instance
        #         # W, V = np.linalg.eig(np.cov(Z1.T))
        #         # PC1, PC2 = V[np.argsort(abs(W))]
        #         # rotation = 180 * np.arctan2(*PC1) / np.pi
        #         # # 37.89555000213858
        #         # T = np.array([PC1[0], PC1[1]])
        #         # # PC1 切向量
        #         # transform = Affine2D().rotate_deg(-rotation)
        #         # P1 = transform.transform(Z1 - Z1.mean(axis=0))     
        #         # # P1 : 沿着x轴，旋转Z1
        #         # # print(P1)
        #         # w = P1[:, 1].max()-P1[:, 1].min()
        #         # h =  P1[:, 0].max()-P1[:, 0].min()
        #         # rbox =np.array( [Z1.mean(axis=0)[0],Z1.mean(axis=0)[1], w,h ,rotation/180*np.pi+np.pi/2]).reshape(-1,5)
                
        #         #这是采用不变矩的方法 和点到直线的距离
        #         # ct = np.array(text_instance).mean(0)
        #         # diff = np.array(text_instance)-ct
        #         # M11 = np.sum(diff[:,0]* diff[:,1])
        #         # M20 = np.sum(diff[:,0]* diff[:,0])
        #         # M02 = np.sum(diff[:,1]* diff[:,1])
        #         # thea = np.arctan2(2*M11,M20-M02)/2
        #         # tanthe = np.tan(thea)
        #         # h_dist = np.abs(tanthe*text_instance[:,0] - text_instance[:,1]-tanthe*ct[0] +ct[1])/np.sqrt(1+ tanthe*tanthe)
        #         # h = np.max(h_dist)*2
        #         # w_dist = np.abs(text_instance[:,0] + tanthe*text_instance[:,1] - ct[0] - tanthe * ct[1])/np.sqrt(1+ tanthe*tanthe)
        #         # w = np.max(w_dist)*2
        #         # rbox =np.array( [ct[0],ct[1], w,h ,thea]).reshape(-1,5)
        #         ##这是采用矩阵形式的不变矩
        #         rbox = rboxes[i:i+1]

        #         ploy = rotated_box_to_poly_np (rbox)
        #         ploy=np.array(ploy[0],dtype=np.int)
        #         polygon = np.array(text_instance).reshape((1, -1, 2))
        #         cv2.fillPoly(mask, polygon.astype(np.int32), np.int(gt_labels[i]+1))
        #         cv2.line(mask, ploy[0:2], ploy[2:4], color=(255,255,255))
        #         cv2.line(mask, ploy[2:4], ploy[4:6], color=(255,255,255))
        #         cv2.line(mask, ploy[4:6], ploy[6:8],color=(255,255,255))
        #         cv2.line(mask, ploy[6:8], ploy[0:2],color=(255,255,255))
                
        #     fout_name = os.path.splitext(img_name)[0]+'_f.png'
        #     output_name = os.path.join(out_put_path, fout_name)
        #     cv2.imwrite(output_name,mask)

        return results
    
    
    


@PIPELINES.register_module()
class RandomRotatePolyInstances:

    def __init__(self,
                fix_size=True,
                 rotate_ratio=0.5,
                 max_angle=10,
                 pad_with_fixed_color=False,
                 pad_value=(0, 0, 0)):
        """Randomly rotate images and polygon masks.

        Args:
            rotate_ratio (float): The ratio of samples to operate rotation.
            max_angle (int): The maximum rotation angle.
            pad_with_fixed_color (bool): The flag for whether to pad rotated
               image with fixed value. If set to False, the rotated image will
               be padded onto cropped image.
            pad_value (tuple(int)): The color value for padding rotated image.
        """
        self.rotate_ratio = rotate_ratio
        self.max_angle = max_angle
        self.pad_with_fixed_color = pad_with_fixed_color
        self.pad_value = pad_value
        self.fix_size=fix_size

    def rotate(self, center, points, theta, center_shift=(0, 0)):
        # rotate points.
        (center_x, center_y) = center
        center_y = -center_y
        x, y = points[::2], points[1::2]
        y = -y

        theta = theta / 180 * math.pi
        cos = math.cos(theta)
        sin = math.sin(theta)

        x = (x - center_x)
        y = (y - center_y)

        _x = center_x + x * cos - y * sin + center_shift[0]
        _y = -(center_y + x * sin + y * cos) + center_shift[1]

        points[::2], points[1::2] = _x, _y
        return points

    def cal_canvas_size(self, ori_size, degree):
        assert isinstance(ori_size, tuple)
        angle = degree * math.pi / 180.0
        h, w = ori_size[:2]

        cos = math.cos(angle)
        sin = math.sin(angle)
        canvas_h = int(w * math.fabs(sin) + h * math.fabs(cos))
        canvas_w = int(w * math.fabs(cos) + h * math.fabs(sin))

        canvas_size = (canvas_h, canvas_w)
        return canvas_size

    def sample_angle(self, max_angle):
        angle = np.random.random_sample() * 2 * max_angle - max_angle
        return angle

    def rotate_img(self, img, angle, canvas_size):
        h, w = img.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        rotation_matrix[0, 2] += int((canvas_size[1] - w) / 2)
        rotation_matrix[1, 2] += int((canvas_size[0] - h) / 2)

        if self.pad_with_fixed_color:
            target_img = cv2.warpAffine(
                img,
                rotation_matrix, (canvas_size[1], canvas_size[0]),
                flags=cv2.INTER_NEAREST,
                borderValue=self.pad_value)
        else:
            mask = np.zeros_like(img)
            (h_ind, w_ind) = (np.random.randint(0, h * 7 // 8),
                              np.random.randint(0, w * 7 // 8))
            img_cut = img[h_ind:(h_ind + h // 9), w_ind:(w_ind + w // 9)]
            img_cut = mmcv.imresize(img_cut, (canvas_size[1], canvas_size[0]))
            mask = cv2.warpAffine(
                mask,
                rotation_matrix, (canvas_size[1], canvas_size[0]),
                borderValue=[1, 1, 1])
            target_img = cv2.warpAffine(
                img,
                rotation_matrix, (canvas_size[1], canvas_size[0]),
                borderValue=[0, 0, 0])
            target_img = target_img + img_cut * mask

        return target_img

    def __call__(self, results):
        if np.random.random_sample() < self.rotate_ratio:
            img = results['img']
            ori_img_shape = img.shape
            h, w = img.shape[:2]
            angle = self.sample_angle(self.max_angle)
            canvas_size = self.cal_canvas_size((h, w), angle)
            center_shift = (int(
                (canvas_size[1] - w) / 2), int((canvas_size[0] - h) / 2))

            # rotate image
            results['rotated_poly_angle'] = angle
            img = self.rotate_img(img, angle, canvas_size)
            img_shape = img.shape
            if self.fix_size:
                img = cv2.resize(img,(h,w))
                # assert img_shape[0]==img_shape[1]
                ratio = h/img_shape[0]
                # img_shape = img.shape[:2]
                results['img_shape'] = ori_img_shape
            else:
                ratio=1.0
                results['img_shape'] = img_shape
            results['img'] = img
            
            # rotate polygons
            for key in results.get('mask_fields', []):
                if len(results[key].masks) == 0:
                    continue
                masks = results[key].masks
                rotated_masks = []
                for mask in masks:
                    rotated_mask = self.rotate((w / 2, h / 2), mask[0], angle,
                                               center_shift)*ratio
                    rotated_masks.append([rotated_mask])

                results[key] = PolygonMasks(rotated_masks, *(results['img_shape'] [:2]))

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str
    
    
    

@PIPELINES.register_module()
class FCETargets_show(BaseTextDetTargets):
    """Generate the ground truth targets of FCENet: Fourier Contour Embedding
    for Arbitrary-Shaped Text Detection.

    [https://arxiv.org/abs/2104.10442]

    Args:
        fourier_degree (int): The maximum Fourier transform degree k.
        resample_step (float): The step size for resampling the text center
            line (TCL). It's better not to exceed half of the minimum width.
        center_region_shrink_ratio (float): The shrink ratio of text center
            region.
        level_size_divisors (tuple(int)): The downsample ratio on each level.
        level_proportion_range (tuple(tuple(int))): The range of text sizes
            assigned to each level.
    """

    def __init__(self,
                 fourier_degree=10,
                 num_reconstr_points=256,
                 debug = False,
                 circular_class=[],
                 show_dir=None
                 ):
        super().__init__()
        self.fourier_degree = fourier_degree
        self.num_sample = num_reconstr_points
        self.debug = debug
        self.circular_class = circular_class
        self.show_dir=show_dir
        
    def normalize_polygon(self, polygon):
        """Normalize one polygon so that its start point is at right most.

        Args:
            polygon (list[float]): The origin polygon.
        Returns:
            new_polygon (lost[float]): The polygon with start point at right.
        """
        temp_polygon = polygon - polygon.mean(axis=0)
        x = np.abs(temp_polygon[:, 0])
        y = temp_polygon[:, 1]
        index_x = np.argsort(x)
        index_y = np.argmin(y[index_x[:8]])
        index = index_x[index_y]
        new_polygon = np.concatenate([polygon[index:], polygon[:index]])
        return new_polygon

    def poly2fourier(self, polygon, fourier_degree):
        """Perform Fourier transformation to generate Fourier coefficients ck
        from polygon.

        Args:
            polygon (ndarray): An input polygon.
            fourier_degree (int): The maximum Fourier degree K.
        Returns:
            c (ndarray(complex)): Fourier coefficients.
        """
        points = polygon[:, 0] + polygon[:, 1] * 1j
        c_fft = fft(points) / len(points)
        c = np.hstack((c_fft[-fourier_degree:], c_fft[:fourier_degree + 1]))
        return c
    
    def clockwise(self, c, fourier_degree):
        """Make sure the polygon reconstructed from Fourier coefficients c in
        the clockwise direction.

        Args:
            polygon (list[float]): The origin polygon.
        Returns:
            new_polygon (lost[float]): The polygon in clockwise point order.
        """
        if np.abs(c[fourier_degree + 1]) > np.abs(c[fourier_degree - 1]):
            return c
        elif np.abs(c[fourier_degree + 1]) < np.abs(c[fourier_degree - 1]):
            return c[::-1]
        else:
            if np.abs(c[fourier_degree + 2]) > np.abs(c[fourier_degree - 2]):
                return c
            else:
                return c[::-1]

    def cal_fourier_signature(self, polygon, fourier_degree):
        """Calculate Fourier signature from input polygon.

        Args:
              polygon (ndarray): The input polygon.
              fourier_degree (int): The maximum Fourier degree K.
        Returns:
              fourier_signature (ndarray): An array shaped (2k+1, 2) containing
                  real part and image part of 2k+1 Fourier coefficients.
        """
        assert len(polygon.shape)==2
        resampled_polygon = cv2.resize(np.array(polygon,dtype=np.float32),(2,self.num_sample))#归一化点的数量
        resampled_polygon = self.normalize_polygon(resampled_polygon)
        fourier_coeff = self.poly2fourier(resampled_polygon, fourier_degree)
        fourier_coeff = self.clockwise(fourier_coeff, fourier_degree)

        real_part = np.real(fourier_coeff).reshape((-1, 1))
        image_part = np.imag(fourier_coeff).reshape((-1, 1))
        fourier_signature = np.hstack([real_part, image_part])
        return fourier_signature

    def fourier2poly(self, real_maps, imag_maps):
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
            -self.fourier_degree,
            self.fourier_degree + 1,
            dtype=torch.float,
            device=device).view(-1, 1)
        i_vect = torch.arange(
            0, self.num_sample, dtype=torch.float, device=device).view(1, -1)

        transform_matrix = 2 * np.pi / self.num_sample * torch.mm(
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
    
    def generate_fourier_maps(self, img_size, text_polys,gt_labels):
        """Generate Fourier coefficient maps.

        Args:
            img_size (tuple): The image size of (height, width).
            text_polys (list[list[ndarray]]): The list of text polygons.

        Returns:
            fourier_real_map (ndarray): The Fourier coefficient real part maps.
            fourier_image_map (ndarray): The Fourier coefficient image part
                maps.
        """
        assert isinstance(img_size, tuple)
        # assert check_argument.is_2dlist(text_polys)
        assert len(text_polys)==len(gt_labels)
        h, w = img_size
        k = self.fourier_degree
        fourier_coeff=[]
        text_region_mask = np.zeros((h, w), dtype=np.uint8)
        gt_rboxes, gt_bboxes=[], []
        for ind, poly in enumerate( text_polys):
            #处理一个图形有多个轮廓的问题，就是取面积最大的那个轮廓
            if  len(poly) >1:
                areas=[]
                for pol in poly:
                    text_instance = pol.reshape(-1,2)
                    polygon = np.array(text_instance,dtype=np.int).reshape((1, -1, 2))
                    rbbox = cv2.minAreaRect(polygon)
                    w, h = rbbox[1][0], rbbox[1][1]
                    areas.append( w*h)
                max_ind = np.argmax(np.array(areas))
                poly = [poly[max_ind]]
            assert len(poly) == 1
            poly = np.array(poly,dtype = np.int)
            gt_rboxes.append(poly_to_rotated_box(poly,self.circular_class))
            poly = np.array(poly).reshape((-1, 2))
            box =np.hstack([poly[:,0].min(),poly[:,1].min(),poly[:,0].max(),poly[:,1].max(),]) 
            gt_bboxes.append(box)

            text_instance = poly.reshape(-1,2)
            polygon = np.array(text_instance).reshape((1, -1, 2))
            class_index = np.int(gt_labels[ind]+1)# 这里要从1开始，如果是0无法和背景区分
            cv2.fillPoly(text_region_mask, polygon, class_index)

            fourier_coeff.append( self.cal_fourier_signature(polygon[0], k))
        return np.array(fourier_coeff), text_region_mask, np.array(gt_rboxes), np.array(gt_bboxes)

    def generate_targets(self, results):
        """Generate the ground truth targets for FCENet.

        Args:
            results (dict): The input result dictionary.

        Returns:
            results (dict): The output result dictionary.
        """

        assert isinstance(results, dict)
        polygon_masks = results['gt_masks'].masks
        gt_labels =  results['gt_labels']
        h, w, _ = results['img_shape']
        assert len(polygon_masks) >0
        
        lv_text_polys = polygon_masks
        level_img_size = (h , w )
        fourier_coeff , text_region, gt_rboxes, gt_bboxes= self.generate_fourier_maps(
            level_img_size, lv_text_polys,gt_labels)
        # try:
        gt_masks = np.concatenate((gt_rboxes, fourier_coeff.reshape(fourier_coeff.shape[0],-1)),axis = 1)
        # i_fft_np = fourier_to_ploy(fourier_coeff,self.fourier_degree,self.num_sample).numpy()
        
        # except:
        #     print(gt_rboxes)
        results['gt_masks'] =  gt_masks
        results['gt_bboxes'] =np.array(gt_bboxes,dtype = np.float32)
        results['gt_semantic_seg']  =text_region
    
            ##TODO 以下是调试代码
        if self.debug:
            #1、显示mask
            img_name = results['img_info']['file_name']
            if self.show_dir is not None:
                out_put_path = self.show_dir 
            else:
                out_put_path = os.path.join(os.path.dirname(os.path.dirname(  results['img_prefix'])),'show_four')
                output_name = os.path.join(out_put_path, img_name)
            if not os.path.exists(out_put_path):
                os.makedirs(out_put_path)
            # cv2.imwrite(output_name,text_region)
            
            fourier_coeff = torch.tensor(torch.from_numpy(
                                                            fourier_coeff),dtype = torch.float32)
            i_fft_np = fourier_to_ploy(fourier_coeff,self.fourier_degree,self.num_sample).numpy()
            # realpart = fourier_coeff_tensor[:,:,0]
            # imgpart = fourier_coeff_tensor[:,:,1]
            # # 2、傅里叶反变换 显示反变换mask
            # i_fft = self.fourier2poly(realpart, imgpart)
            # i_fft_np = torch.cat([i_fft[0][:,:,None],i_fft[1][:,:,None]],dim=2).numpy()
            mask = np.zeros((h, w), dtype=np.uint8)
            for i in range(i_fft_np.shape[0]):
                text_instance =i_fft_np[i]
                polygon = np.array(text_instance).reshape((1, -1, 2))
                cv2.fillPoly(mask, polygon.astype(np.int32), np.int(gt_labels[i]+1))
            fout_name = os.path.splitext(img_name)[0]+'_f.png'
            output_name = os.path.join(out_put_path, fout_name)
            cv2.imwrite(output_name,mask)

        return results
    