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
import scipy.io as sio
from pathlib import Path

from .JointsDataset import JointsDataset
from virtualpose.utils.transforms import projectPoints
from virtualpose.utils.cameras_cpu import world_to_camera_frame, camera_to_world_frame, project_pose, project_pose_camera

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

class MuPoTS(JointsDataset):
    def __init__(self, cfg, image_set, is_train, dataset_root, transform=None, data_path=''):
        super().__init__(cfg, image_set, is_train, dataset_root, transform, data_path)
        self.matched_threshold = cfg.ROOT_ESTIMATION.MATCHED_THRESHOLD
        self.pixel_std = 200.0
        self.num_joints = cfg.NETWORK.NUM_JOINTS
        self.joints_name = JOINTS_DEF_MUPOTS if self.num_joints == 17 else JOINTS_DEF_PANOPTIC
        self.match_idx = {v: JOINTS_NAME.index(k) for k, v in self.joints_name.items()}
        self.match_idx = [self.match_idx[i] for i in range(len(self.joints_name))]
        self.match_idx_back = [0 for _ in range(self.num_joints)]
        for i, idx in enumerate(self.match_idx):
            self.match_idx_back[idx] = i
        self.limbs = LIMBS
        self.root_id = cfg.DATASET.ROOTIDX
        self.ground_center = cfg.DATASET.GROUND_CENTER

        self.db_file = 'group_{}_joint{}{}.pkl'.format(self.image_set, self.num_joints, \
            'ground' if self.ground_center else '')
        self.db_file = osp.join(self.dataset_root, self.db_file)
        if osp.exists(self.db_file) and False: # lazy loading
            self.db = pickle.load(open(self.db_file, 'rb'))
        else:
            annot = json.load(open(osp.join(self.dataset_root, 'MuPoTS-3D.json'), 'r'))
            cameras = pickle.load(open(osp.join(self.dataset_root, 'cameras.pkl'), 'rb'))
            self.db = self._get_db(annot, cameras)
            pickle.dump(self.db, open(self.db_file, 'wb'))

        self.db_size = len(self.db)


    def _get_db(self, annot, cameras):
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
            if not pose_annot['is_valid'] :
                continue
            img_id, joints_img, joints_3d, joints_vis = pose_annot['image_id'], \
                np.array(pose_annot['keypoints_img']), np.array(pose_annot['keypoints_cam']), np.array(pose_annot['keypoints_vis'])
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
                      [0.0, 0.0, -1.0],
                      [0.0, 1.0, 0.0]])

        for img_annot in img_annots:
            if len(img_annot['joints_3d']) > 0:
                sub = int(img_annot['file_name'].split('/')[0][2:]) - 1
                our_cam = {}
                our_cam['R'] = cameras[sub]['R']
                if (np.linalg.norm(np.cross(our_cam['R'][:, 0], our_cam['R'][:, 1]) - our_cam['R'][:, 2])) > 1e-1:
                    our_cam['R'][:, 1] = -our_cam['R'][:, 1]
                our_cam['T'] = -np.dot(our_cam['R'].T, np.array(cameras[sub]['t']).reshape(3, 1))
                our_cam['fx'], our_cam['fy'], our_cam['cx'], our_cam['cy'] = img_annot['intrinsic']
                our_cam['k'] = np.zeros((3, 1))
                our_cam['p'] = np.zeros((2, 1))
                poses_3d_cam = np.array(img_annot['joints_3d'])[:, self.match_idx]
                poses_3d = camera_to_world_frame(poses_3d_cam.reshape(-1, 3), our_cam['R'], our_cam['T']).reshape(*(poses_3d_cam.shape))
                
                new_center = poses_3d[:, self.root_id].mean(axis = 0)
                if self.ground_center:
                    new_center[2] = -50
                    our_cam['T'] -= new_center[:, None]
                    poses_3d -= new_center[None, None]

                db.append({
                    'image': osp.join(self.dataset_root, 'images', img_annot['file_name']),
                    'joints_3d': poses_3d,
                    'joints_3d_vis': np.array(img_annot['joints_vis'])[:, self.match_idx, None].repeat(3, axis = -1),
                    'joints_2d': np.array(img_annot['joints_2d'])[:, self.match_idx],
                    'joints_2d_vis': np.array(img_annot['joints_vis'])[:, self.match_idx, None].repeat(2, axis = -1),
                    'camera': our_cam
                })

        db.sort(key = lambda x: x['image'])
        return db

    def __len__(self):
        return self.db_size

    def evaluate(self, preds, backbone_outputs, grid_centers, frame_valids):
        eval_list = []
        eval_list_TS = [[] for i in range(20)] 
        gt_num = self.db_size
        assert len(preds) == gt_num and len(frame_valids) == gt_num, 'number mismatch'

        pred_2d_save = {}
        pred_3d_save = {}

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

            for gt_num, (gt, gt_vis) in enumerate(zip(joints_3d, joints_3d_vis)):
                mpjpes = []
                vis = gt_vis[:, 0] > 0
                if len(pred) == 0:
                    continue
                for pose, center in zip(pred, grid_center):
                    vis = gt_vis[:, 0] > 0
                    mpjpe = np.linalg.norm(pose[vis, :3] - gt[vis], axis = -1).mean()
                    mpjpes.append(mpjpe)

                min_pred = np.argmin(mpjpes)
                min_mpjpe = np.min(mpjpes)
                pose, center = pred[min_pred], grid_center[min_pred]
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
                    "gt_id": int(total_gt + gt_num),
                    "gt_pose": gt,
                    "pred_pose": pose[:, :3],
                    "vis": vis,
                    "root_id": self.root_id
                })

                TS_num = int(db_rec['image'].split('/TS')[1].split('/')[0]) - 1
                eval_list_TS[TS_num].append(copy.deepcopy(eval_list[-1]))

            total_gt += len(joints_3d)
            img_name = db_rec['image'].split('/')
            img_name = img_name[-2] + '_' + img_name[-1].split('.')[0] # e.g., TS1_img_0001
            pred_3d_save[img_name] = np.zeros((len(pred), 17, 3))
            pred_2d_save[img_name] = np.zeros((len(pred), 17, 2))
            
            pred = pred[:, self.match_idx_back, :3]
            pred_3d_save[img_name][:, :self.num_joints] = world_to_camera_frame(pred.reshape(-1, 3), camera['R'], camera['T']).reshape(*(pred.shape))
            pred_2d_save[img_name][:, :self.num_joints] = project_pose(pred[..., :3].reshape(-1, 3), camera).reshape(*(pred.shape[:2]), 2)
        
        try:
            Path('mupots_results').mkdir(exist_ok=True)
            output_path = 'mupots_results/preds_2d_kpt_mupots.mat'
            sio.savemat(output_path, pred_2d_save)
            print("Testing result is saved at " + output_path)
            output_path = 'mupots_results/preds_3d_kpt_mupots.mat'
            sio.savemat(output_path, pred_3d_save)
            print("Testing result is saved at " + output_path)
            output_path = 'mupots_results/eval_list.pkl'
            pickle.dump(eval_list, open(output_path, 'wb'))
        except:
            None

        return self._eval_list_to_mpjpe(eval_list, self.matched_threshold), eval_list


    @staticmethod
    def _eval_list_to_mpjpe(eval_list, threshold):
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




