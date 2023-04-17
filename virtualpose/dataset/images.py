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

from .JointsDataset import JointsDataset
from virtualpose.utils.transforms import projectPoints
from virtualpose.utils.cameras_cpu import world_to_camera_frame, camera_to_world_frame, project_pose, project_pose_camera

logger = logging.getLogger(__name__)

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


JOINTS_NAME = ('headtop', 'neck',
                   'r-shoulder', 'r-elbow', 'r-wrist', 'l-shoulder', 'l-elbow', 'l-wrist',
                   'r-hip', 'r-knee', 'r-ankle', 'l-hip', 'l-knee', 'l-ankle',
                   'mid-hip', 'spine', 'head',
                   'R_Hand', 'L_Hand', 'R_Toe', 'L_Toe')


class Images(JointsDataset):
    def __init__(self, cfg, image_dir, focal_length=1700, transform=None):
        super().__init__(cfg, '', False, '', transform)
        self.num_views = 1
        self.pixel_std = 200.0
        self.joints_name = JOINTS_DEF_MUPOTS
        self.match_idx = {v: JOINTS_NAME.index(k) for k, v in JOINTS_DEF_MUPOTS.items()}
        self.match_idx = [self.match_idx[i] for i in range(len(self.joints_name))]
        self.limbs = LIMBS
        self.num_joints = len(self.joints_name)
        self.root_id = cfg.DATASET.ROOTIDX
        self.image_dir = image_dir

        self.cam = {}
        self.cam['R'] = np.identity(3)
        self.cam['T'] = np.zeros((3, 1))
        self.cam['fx'] = self.cam['fy'] = focal_length
        self.cam['cx'] = self.cam['cy'] = 0.
        self.cam['k'] = np.zeros((3, 1))
        self.cam['p'] = np.zeros((2, 1))

        self.db = self._get_db()
        self.db_size = len(self.db)

    def _get_db(self):
        img_files = sorted([osp.join(self.image_dir, img_file) for img_file in os.listdir(self.image_dir) \
            if img_file.endswith('.jpg') or img_file.endswith('.png')])

        db = []
        for img_file in img_files:
            db.append({
                'image': img_file,
                'joints_3d': np.zeros((0, len(self.match_idx), 3)),
                'joints_3d_cam': np.zeros((0, len(self.match_idx), 3)),
                'joints_3d_vis': np.zeros((0, len(self.match_idx), 3)),
                'joints_2d': np.zeros((0, len(self.match_idx), 2)),
                'joints_2d_vis': np.zeros((0, len(self.match_idx), 2)),
                'camera': self.cam,
                'bbox': np.zeros((0, 4))
            })

        return db

    def __len__(self):
        return self.db_size



