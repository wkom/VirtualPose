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

from dataset.JointsDataset import JointsDataset
from utils.transforms import projectPoints
from utils.cameras_cpu import world_to_camera_frame, camera_to_world_frame, project_pose, project_pose_camera

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


JOINTS_NAME = ('headtop', 'neck',
                   'r-shoulder', 'r-elbow', 'r-wrist', 'l-shoulder', 'l-elbow', 'l-wrist',
                   'r-hip', 'r-knee', 'r-ankle', 'l-hip', 'l-knee', 'l-ankle',
                   'mid-hip', 'spine', 'head',
                   'R_Hand', 'L_Hand', 'R_Toe', 'L_Toe')

class MuCo(JointsDataset):
    def __init__(self, cfg, image_set, is_train, dataset_root, transform=None, data_path=''):
        super().__init__(cfg, image_set, is_train, dataset_root, transform, data_path)
        self.pixel_std = 200.0
        self.num_joints = cfg.NETWORK.NUM_JOINTS
        self.joints_name = JOINTS_DEF_MUPOTS if self.num_joints == 17 else JOINTS_DEF_PANOPTIC
        self.match_idx = {v: JOINTS_NAME.index(k) for k, v in self.joints_name.items()}
        self.match_idx = [self.match_idx[i] for i in range(len(self.joints_name))]
        self.limbs = LIMBS
        
        self.root_id = cfg.DATASET.ROOTIDX

        self.db_file = 'group_train_joint{}.pkl'.format(self.num_joints)
        self.db_file = osp.join(self.dataset_root, self.db_file)
        if osp.exists(self.db_file): # lazy loading
            self.db = pickle.load(open(self.db_file, 'rb'))
        else:
            annot = json.load(open(osp.join(self.dataset_root, 'MuCo-3DHP.json'), 'r'))
            self.db = self._get_db(annot)
            pickle.dump(self.db, open(self.db_file, 'wb'))

        self.db_size = len(self.db)

    def _get_db(self, annot):
        width = 1920
        height = 1080
        db = []

        pose_annots, img_annots = annot['annotations'], annot['images']
        for img_annot in img_annots:
            img_annot.update({
                'joints_3d': [],
                'joints_2d': [],
                'joints_vis': [],
            })
        for pose_annot in pose_annots:
            img_id, joints_img, joints_3d, joints_vis, bbox = pose_annot['image_id'], np.array(pose_annot['keypoints_img']),\
                np.array(pose_annot['keypoints_cam']), np.array(pose_annot['keypoints_vis']), np.array(pose_annot['bbox'])
            x_check = np.bitwise_and(joints_img[:, 0] >= 0,
                                    joints_img[:, 0] <= width - 1)
            y_check = np.bitwise_and(joints_img[:, 1] >= 0,
                                    joints_img[:, 1] <= height - 1)
            check = np.bitwise_and(x_check, y_check)
            joints_vis[np.logical_not(check)] = 0
            joints_vis = np.ones_like(joints_vis)
            img_annots[img_id]['joints_3d'].append(joints_3d)
            img_annots[img_id]['joints_2d'].append(joints_img)
            img_annots[img_id]['joints_vis'].append(joints_vis)

        M = np.array([[1.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0],
                      [0.0, -1.0, 0.0]])

        for img_annot in img_annots:
            if len(img_annot['joints_3d']) > 0:
                our_cam = {}
                our_cam['R'] = np.array(img_annot['R']).reshape(3, 3).dot(M)
                our_cam['T'] = -np.dot(our_cam['R'].T, np.array(img_annot['T']).reshape(3, 1))
                our_cam['fx'], our_cam['fy'] = img_annot['f']
                our_cam['cx'], our_cam['cy'] = img_annot['c']
                our_cam['k'] = np.zeros((3, 1))
                our_cam['p'] = np.zeros((2, 1))
                poses_3d_cam = np.array(img_annot['joints_3d'])[:, self.match_idx]
                poses_3d = camera_to_world_frame(poses_3d_cam.reshape(-1, 3), our_cam['R'], our_cam['T']).reshape(*(poses_3d_cam.shape))
                new_center = poses_3d[:, self.root_id].mean(axis = 0) + np.random.uniform(-1, 1, size = 3) * np.array([200, 200, 30]) + np.array([0, 0, 300])
                our_cam['T'] -= new_center[:, None]
                poses_3d -= new_center[None, None]

                db.append({
                    'image': osp.join(self.dataset_root, 'images', img_annot['file_name']),
                    'joints_3d': poses_3d,
                    'joints_3d_cam': poses_3d_cam,
                    'joints_3d_vis': np.array(img_annot['joints_vis'])[:, self.match_idx, None].repeat(3, axis = -1),
                    'joints_2d': np.array(img_annot['joints_2d'])[:, self.match_idx],
                    'joints_2d_vis': np.array(img_annot['joints_vis'])[:, self.match_idx, None].repeat(2, axis = -1),
                    'camera': our_cam,
                })

        db.sort(key = lambda x: x['image'])
        return db


    def __len__(self):
        return self.db_size




