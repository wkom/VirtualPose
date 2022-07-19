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
from pycocotools.coco import COCO

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

class MSCOCO(JointsDataset):
    def __init__(self, cfg, image_set, is_train, dataset_root, transform=None, data_path=''):
        assert is_train
        super().__init__(cfg, image_set, is_train, dataset_root, transform, data_path)
        self.is_train = is_train
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

        self.db_file = 'group_{}_joint{}.pkl'.format(self.image_set, self.num_joints)
        self.db_file = osp.join(self.dataset_root, self.db_file)
        if osp.exists(self.db_file): # lazy loading
            self.db = pickle.load(open(self.db_file, 'rb'))
        else:
            self.img_dir = osp.join(self.dataset_root, 'images')
            self.annot_path = osp.join(self.dataset_root, 'annotations', 'person_keypoints_train2017.json')
            self.db = self._get_db()
            pickle.dump(self.db, open(self.db_file, 'wb'))

        self.db_size = len(self.db)

    def _get_db(self):
        db = COCO(self.annot_path)
        data = {}
        our_cam = {}
        our_cam['R'] = np.identity(3)
        our_cam['T'] = np.zeros((3, 1))
        our_cam['fx'], our_cam['fy'] = 1., 1.
        our_cam['cx'], our_cam['cy'] = 0., 0.
        our_cam['k'] = np.zeros((3, 1))
        our_cam['p'] = np.zeros((2, 1))
        M = np.array([[1.0, 0.0, 0.0],
                      [0.0, 0.0, -1.0],
                      [0.0, 1.0, 0.0]])
        data_base = []

        useful_joint = [i for i, name in enumerate(JOINTS_NAME[:-5]) if name in self.joints_name]

        for aid in db.anns.keys():
            ann = db.anns[aid]

            if (ann['image_id'] not in db.imgs) or ann['iscrowd'] or (ann['num_keypoints'] == 0) or \
                np.all(np.array(db.anns[aid]['keypoints']).reshape(-1, 3)[useful_joint, 2] == 0):
                continue
            
            imgname = osp.join('train2017', db.imgs[ann['image_id']]['file_name'])
            img_path = osp.join(self.img_dir, imgname)
            if img_path in data:
                data[img_path].append(aid)
            else:
                data[img_path] = [aid]

        for img_path, indices in data.items():
            joint_img, bboxes = [], []
            for aid in indices:
                ann = db.anns[aid]
                img = db.loadImgs(ann['image_id'])[0]
                width, height = img['width'], img['height']

                if (ann['image_id'] not in db.imgs) or ann['iscrowd'] or (ann['num_keypoints'] == 0):
                    continue
                
                bbox = process_bbox(ann['bbox'], width, height)

                joint_img.append(np.array(db.anns[aid]['keypoints']).reshape(-1,3))
                bboxes.append(np.array(bbox).reshape(-1, 2))

            joint_img, bboxes = np.stack(joint_img), np.stack(bboxes)

            # add Thorax
            thorax = (joint_img[:, self.lshoulder_idx, :] + joint_img[:, self.rshoulder_idx, :]) * 0.5
            thorax[:, 2] = joint_img[:, self.lshoulder_idx,2] * joint_img[:, self.rshoulder_idx,2]
            thorax = thorax.reshape((-1, 1, 3))
            # add Pelvis
            pelvis = (joint_img[:, self.lhip_idx, :] + joint_img[:, self.rhip_idx, :]) * 0.5
            pelvis[:, 2] = joint_img[:, self.lhip_idx,2] * joint_img[:, self.rhip_idx,2]
            pelvis = pelvis.reshape((-1, 1, 3))
            
            # add other joints(spine, head, headtop)
            others = np.zeros((len(pelvis), 3, 3))

            joint_img = np.concatenate((joint_img, thorax, pelvis, others), axis=1)[:, self.match_idx]

            joint_vis = (joint_img[..., 2:3].copy() > 0)
            joint_img[..., 2] = 1
            joint_img[:, joint_vis[0, :, 0] == 0] = joint_img[:, joint_vis[0, :, 0] > 0].mean(axis = 1, keepdims=True)

            data_base.append({
                'image': img_path,
                'joints_3d': joint_img,
                'joints_3d_cam': joint_img,
                'joints_3d_vis': joint_vis.repeat(3, axis = -1),
                'joints_2d': joint_img[..., :2],
                'joints_2d_vis': joint_vis.repeat(2, axis = -1),
                'camera': our_cam,
                'bbox': bboxes
            })

        data_base.sort(key = lambda x: x['image'])
        return data_base

    def __len__(self):
        return self.db_size