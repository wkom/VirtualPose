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

JOINTS_DEF = {
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


class Panoptic(JointsDataset):
    def __init__(self, cfg, image_set, is_train, dataset_root, transform=None, data_path=''):
        super().__init__(cfg, image_set, is_train, dataset_root, transform, data_path)
        self.pixel_std = 200.0
        self.joints_def = JOINTS_DEF
        self.limbs = LIMBS
        self.num_joints = len(JOINTS_DEF)


        self.db_file = 'group_{}_cam.pkl'.format(self.image_set)
        self.db_file = osp.join(self.dataset_root, self.db_file)
        if osp.exists(self.db_file): # lazy loading
            self.db = pickle.load(open(self.db_file, 'rb'))
        else:
            if self.image_set == 'train':
                annot = pickle.load(open(osp.join(self.dataset_root, 'train_cam.pkl'), 'rb'))
            elif self.image_set == 'validation':
                annot = pickle.load(open(osp.join(self.dataset_root, 'valid_cam.pkl'), 'rb'))
            self.db = self._get_db(annot)
            pickle.dump(self.db, open(self.db_file, 'wb'))
        if self.image_set == 'train' and len(cfg.DATASET.TRAIN_VIEW) > 0:
            self.db = [term for term in self.db \
                if int(term['image'].split('/')[-2].split('_')[1]) in cfg.DATASET.TRAIN_VIEW]
        if self.image_set == 'validation' and len(cfg.DATASET.VAL_VIEW) > 0:
            self.db = [term for term in self.db \
                if int(term['image'].split('/')[-2].split('_')[1]) in cfg.DATASET.VAL_VIEW]
        
        self.db_size = len(self.db)

    def _get_db(self, annot):
        width = 1920
        height = 1080
        db = []
        for term in annot:
            our_cam = {}
            M = np.array([[1.0, 0.0, 0.0],
                      [0.0, 0.0, -1.0],
                      [0.0, 1.0, 0.0]])
            term['intrinsics'], term['R'], term['t'], term['distCoef'] = np.array(term['intrinsics']), \
                np.array(term['R']).dot(M), np.array(term['t']), np.array(term['distCoef']) 
            our_cam['R'] = term['R']
            our_cam['T'] = -np.dot(term['R'].T, term['t']) * 10  # cm to mm
            our_cam['fx'] = float(term['intrinsics'][0, 0])
            our_cam['fy'] = float(term['intrinsics'][1, 1])
            our_cam['cx'] = float(term['intrinsics'][0, 2])
            our_cam['cy'] = float(term['intrinsics'][1, 2])
            our_cam['k'] = np.zeros((3, 1), dtype=np.float)
            our_cam['p'] = np.zeros((2, 1), dtype=np.float)

            poses_3d_cam = np.array(term['joints_3d_cam']) * 10 # cm to mm
            poses_3d = camera_to_world_frame(poses_3d_cam.reshape(-1, 3), our_cam['R'], our_cam['T']).reshape(*(poses_3d_cam.shape))
            poses_vis_3d = np.array(term['joints_3d_vis'], dtype = np.bool)
            joints_vis = poses_vis_3d[..., 0].copy()

            poses_2d = project_pose(poses_3d.reshape(-1, 3), our_cam).reshape(*(poses_3d.shape[:2]), 2)
            x_check = np.bitwise_and(poses_2d[..., 0] >= 0,
                                        poses_2d[..., 0] <= width - 1)
            y_check = np.bitwise_and(poses_2d[..., 1] >= 0,
                                        poses_2d[..., 1] <= height - 1)
            check = np.bitwise_and(x_check, y_check)
            joints_vis[np.logical_not(check)] = 0
            joints_vis = np.repeat(joints_vis[..., None], 2, axis = 2)


            db.append({
                'image': osp.join(self.dataset_root, term['image_name']),
                'joints_3d': poses_3d,
                'joints_3d_vis': poses_vis_3d,
                'joints_3d_cam': poses_3d_cam,
                'joints_2d': poses_2d,
                'joints_2d_vis': joints_vis,
                'camera': our_cam
            })
        return db

    def _get_cam(self, seq):
        cam_file = osp.join(self.dataset_root, seq, 'calibration_{:s}.json'.format(seq))
        with open(cam_file) as cfile:
            calib = json.load(cfile)

        M = np.array([[1.0, 0.0, 0.0],
                      [0.0, 0.0, -1.0],
                      [0.0, 1.0, 0.0]])
        cameras = {}
        for cam in calib['cameras']:
            if (cam['panel'], cam['node']) in self.cam_list:
                sel_cam = {}
                sel_cam['K'] = np.array(cam['K'])
                sel_cam['distCoef'] = np.array(cam['distCoef'])
                sel_cam['R'] = np.array(cam['R']).dot(M)
                sel_cam['t'] = np.array(cam['t']).reshape((3, 1))
                cameras[(cam['panel'], cam['node'])] = sel_cam
        return cameras

    def __len__(self):
        return self.db_size

    def evaluate(self, preds, backbone_outputs, grid_centers, frame_valids):
        eval_list = []
        gt_num = self.db_size
        assert len(preds) == gt_num and len(frame_valids) == gt_num, 'number mismatch'

        total_gt = 0
        for i in range(gt_num):
            index = i
            db_rec = copy.deepcopy(self.db[index])
            joints_3d = db_rec['joints_3d']
            joints_3d_vis = db_rec['joints_3d_vis']
            camera = db_rec['camera']

            if len(joints_3d) == 0 or not frame_valids[index]:
                continue

            pred, grid_center = preds[i].copy(), grid_centers[i].copy()
            pred, grid_center = pred[pred[:, 0, 3] >= 0], grid_center[pred[:, 0, 3] >= 0]
            for pose, center in zip(pred, grid_center):
                mpjpes = []
                for (gt, gt_vis) in zip(joints_3d, joints_3d_vis):
                    vis = gt_vis[:, 0] > 0
                    mpjpe = np.linalg.norm(pose[vis, :3] - gt[vis], axis = -1).mean()
                    mpjpes.append(mpjpe)

                min_gt = np.argmin(mpjpes)
                min_mpjpe = np.min(mpjpes)
                gt, vis = joints_3d[min_gt], joints_3d_vis[min_gt][:, 0] > 0
                mpjpe_aligned = np.linalg.norm((pose[vis, :3] - pose[self.root_id:self.root_id + 1, :3]) - \
                    (gt[vis] - gt[self.root_id:self.root_id+1]), axis = -1).mean()
                gt_camera, pose_camera = world_to_camera_frame(gt[:, :3], camera['R'], camera['T']), \
                    world_to_camera_frame(pose[:, :3], camera['R'], camera['T'])
                gt_pose2d, pred_pose2d = project_pose_camera(gt_camera, camera), project_pose_camera(pose_camera, camera)
                error_2d = np.linalg.norm(pred_pose2d[vis] - gt_pose2d[vis], axis = -1).mean()
                mrpe = np.abs(pose_camera[self.root_id] - gt_camera[self.root_id])
                
                score = pose[0, 4]
                eval_list.append({
                    "image_path": db_rec['image'],
                    "mpjpe": float(min_mpjpe),
                    "mpjpe_aligned": float(mpjpe_aligned),
                    "mrpe": mrpe,
                    "error2d": error_2d,
                    "score": float(score),
                    "gt_id": int(total_gt + min_gt),
                    "gt_pose": gt,
                    "pred_pose": pose[:, :3],
                    "vis": vis,
                    "root_id": self.root_id
                })

            total_gt += len(joints_3d)

        subject_list = ['haggling', 'mafia', 'ultimatum', 'pizza']
        for subject in subject_list:
            subject_eval_list = [term for term in eval_list if subject in term['image_path']]
            print(subject, ':\n', self._eval_list_to_mpjpe(subject_eval_list))

        return self._eval_list_to_mpjpe(eval_list), eval_list

    @staticmethod
    def _eval_list_to_mpjpe(eval_list, threshold=500):
        eval_list.sort(key=lambda k: k["score"], reverse=True)
        gt_det = []

        mpjpes, mpjpes_aligned, mrpes, errors_2d = [], [], [], []
        for i, item in enumerate(eval_list):
            if item["mpjpe"] < threshold and item["gt_id"] not in gt_det:
                mpjpes.append(item["mpjpe"])
                mpjpes_aligned.append(item["mpjpe_aligned"])
                mrpes.append(item["mrpe"])
                gt_det.append(item["gt_id"])
                errors_2d.append(item["error2d"])
        mrpes = np.array(mrpes)

        metric = {
            'mpjpe': np.mean(mpjpes) if len(mpjpes) > 0 else np.inf,
            'mpjpe_aligned': np.mean(mpjpes_aligned) if len(mpjpes_aligned) > 0 else np.inf,
            'mrpe': {
                'x': mrpes[:, 0].mean() if len(mpjpes) > 0 else np.inf,
                'y': mrpes[:, 1].mean() if len(mpjpes) > 0 else np.inf,
                'z': mrpes[:, 2].mean() if len(mpjpes) > 0 else np.inf,
                'root': np.linalg.norm(mrpes, axis = -1).mean() if len(mpjpes) > 0 else np.inf,
            },
            'error2d': np.mean(errors_2d) if len(mpjpes) > 0 else np.inf
        }
        return metric



