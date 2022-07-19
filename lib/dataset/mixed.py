from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os.path as osp
import numpy as np
import json_tricks as json
import pickle
import logging
import os
import copy
import cv2

from dataset.JointsDataset import JointsDataset
from utils.transforms import projectPoints
from utils.cameras_cpu import world_to_camera_frame, camera_to_world_frame, project_pose, project_pose_camera
from utils.pose_utils import process_bbox

logger = logging.getLogger(__name__)

JOINTS_DEF_PANOPTIC = {
    'neck': 0,
    'headtop': 1,  
    'mid-hip': 2,
    'l-shoulder': 3,
    'l-elbow': 4,
    'l-wrist': 5,
    'l-hip': 6,
    'l-knee': 7,
    'l-ankle': 8,
    'r-shoulder': 9,
    'r-elbow': 10,
    'r-wrist': 11,
    'r-hip': 12,
    'r-knee': 13,
    'r-ankle': 14,
}

JOINTS_DEF_MUPOTS = {
    'neck': 0,
    'headtop': 1,  
    'mid-hip': 2,
    'l-shoulder': 3,
    'l-elbow': 4,
    'l-wrist': 5,
    'l-hip': 6,
    'l-knee': 7,
    'l-ankle': 8,
    'r-shoulder': 9,
    'r-elbow': 10,
    'r-wrist': 11,
    'r-hip': 12,
    'r-knee': 13,
    'r-ankle': 14,
    'spine': 15,
    'head': 16,
}

LIMBS = [[0, 1],
         [0, 2],
         [0, 3],
         [3, 4],
         [4, 5],
         [0, 9],
         [9, 10],
         [10, 11],
         [2, 6],
         [2, 12],
         [6, 7],
         [7, 8],
         [12, 13],
         [13, 14]]

JOINTS_NAME = ('nose', 'l-eye', 'r-eye', 'l-ear', 'r-ear', 'l-shoulder', 'r-shoulder', 'l-elbow', 'r-elbow', 
                'l-wrist', 'r-wrist', 'l-hip', 'r-hip', 'l-knee', 'r-knee', 'l-ankle', 'r-ankle', 'neck', 
                'mid-hip', 'spine', 'head', 'headtop')


class Mixed(JointsDataset):
    def __init__(self, cfg, dataset_list, data_path=''):
        # assert is_train
        assert len(dataset_list) == 2
        super().__init__(cfg, 'train', True, '', dataset_list[0].transform, data_path)
        self.is_train = True
        self.pixel_std = 200.0
        self.num_joints = cfg.NETWORK.NUM_JOINTS
        self.joints_name = JOINTS_DEF_MUPOTS if self.num_joints == 17 else JOINTS_DEF_PANOPTIC
        self.match_idx = {v: JOINTS_NAME.index(k) for k, v in self.joints_name.items()}
        self.match_idx = [self.match_idx[i] for i in range(len(self.joints_name))]
        self.limbs = LIMBS
        
        self.root_id = cfg.DATASET.ROOTIDX # self.joints_name['mid-hip']
        self.lshoulder_idx = JOINTS_NAME.index('l-shoulder')
        self.rshoulder_idx = JOINTS_NAME.index('r-shoulder')
        self.lhip_idx = JOINTS_NAME.index('l-hip')
        self.rhip_idx = JOINTS_NAME.index('r-hip')

        if len(dataset_list[0].db) < len(dataset_list[1].db):
            dataset_list[0], dataset_list[1] = dataset_list[1], dataset_list[0]
        k = int(len(dataset_list[0].db) / len(dataset_list[1].db) + 0.5)
        self.db = dataset_list[0].db + k * dataset_list[1].db
        self.db_size = len(self.db)


    def __len__(self):
        return self.db_size