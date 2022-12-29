import itertools
import logging
import os.path as osp
import tempfile
import cv2
import mmcv
import numpy as np
from mmcv.utils import print_log
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from terminaltables import AsciiTable
import operator
import cv2
from mmdet.core import eval_recalls
from .builder import DATASETS
from .custom import CustomDataset
from .coco import CocoDataset
import xml.etree.ElementTree as ET

import shutil
@DATASETS.register_module()
class HRSIDDataset(CocoDataset):
    CLASSES = (
        'ship',
      )
    PALETTE = None


    # def load_annotations(self, ann_file):
    #     """Load annotation from COCO style annotation file.

    #     Args:
    #         ann_file (str): Path of annotation file.

    #     Returns:
    #         list[dict]: Annotation info from COCO api.
    #     """

    #     self.coco = COCO(ann_file)
    #     # The order of returned `cat_ids` will not
    #     # change with the order of the CLASSES
    #     #TODO insert 增加这一条是因为在生成数据集的过程中，name没有按照ID排序，这要造成获取类别ID的时候出现混乱
    #     cat_dict = sorted(self.coco.dataset['categories'],key=operator.itemgetter('id'))
    #     self.coco.dataset['categories'] = cat_dict
            
    #     self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)

    #     self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
    #     self.img_ids = self.coco.get_img_ids()
    #     data_infos = []
    #     total_ann_ids = []
    #     for i in self.img_ids:
    #         info = self.coco.load_imgs([i])[0]
    #         info['filename'] = info['file_name']
    #         data_infos.append(info)
    #         ann_ids = self.coco.get_ann_ids(img_ids=[i])
    #         total_ann_ids.extend(ann_ids)
    #     assert len(set(total_ann_ids)) == len(
    #         total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
    #     return data_infos

