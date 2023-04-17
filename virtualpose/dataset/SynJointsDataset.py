import copy
import logging

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import time

from virtualpose.utils.transforms import get_affine_transform
from virtualpose.utils.transforms import affine_transform, get_scale
from virtualpose.utils.cameras_cpu import project_pose, world_to_camera_frame, angles_to_camera
from .JointsDataset import JointsDataset

logger = logging.getLogger(__name__)



class SynJointsDataset(JointsDataset):
    def __init__(self, cfg, image_set, is_train, dataset_root, transform=None, dataset_path='./'):
        super().__init__(cfg, image_set, is_train, dataset_root, transform, dataset_path)

    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.db[idx])

        image_file = db_rec['image']

        if self.data_format == 'zip':
            from utils import zipreader
            data_numpy = zipreader.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        else:
            data_numpy = cv2.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        # data_numpy = np.zeros((500, 500, 3), dtype=np.uint8)

        if data_numpy is None:
            # logger.error('=> fail to read {}'.format(image_file))
            # raise ValueError('Fail to read {}'.format(image_file))
            return None, None, None, None, None, None

        if self.color_rgb:
            data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)
            mean_color = data_numpy.mean(axis = (0, 1), keepdims = True)
            if np.abs(data_numpy - mean_color).sum(axis = -1).max() < 20:
                frame_valid = False
            else:
                frame_valid = True

        joints = db_rec['joints_2d']
        joints_vis = db_rec['joints_2d_vis']
        joints_3d = db_rec['joints_3d']
        assert len(joints) == len(joints_3d), idx
        joints_3d_vis = db_rec['joints_3d_vis']

        nposes = len(joints)
        assert nposes <= self.maximum_person, 'too many persons'

        height, width, _ = data_numpy.shape
        c = np.array([width / 2.0, height / 2.0])
        s = get_scale((width, height), self.image_size)
        if self.is_train and len(self.scale_factor_range) == 2:
            s /= self.scale_factor_range[0] + np.random.random() * \
                (self.scale_factor_range[1] - self.scale_factor_range[0])
        r = 0

        trans = get_affine_transform(c, s, r, self.image_size)
        input = cv2.warpAffine(
            data_numpy,
            trans, (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)

        if self.transform:
            input = self.transform(input)

        for n in range(nposes):
            for i in range(len(joints[0])):
                joints[n][i, 0:2] = affine_transform(
                        joints[n][i, 0:2], trans)
                if joints_vis[n][i, 0] > 0.0:
                    if (np.min(joints[n][i, :2]) < 0 or
                            joints[n][i, 0] >= self.image_size[0] or
                            joints[n][i, 1] >= self.image_size[1]):
                        joints_vis[n][i, :] = 0


        if self.is_train and not self.fix_camera: 
            db_rec['camera'] = self.generate_synthetic_camera()

        if self.is_train and self.random_place: 
            joints_3d = self.random_place_people()

        joints = project_pose(joints_3d.reshape(-1, 3), db_rec['camera']).reshape(*(joints_3d.shape[:2]), 2)
        for n in range(nposes):
            for i in range(len(joints[0])):
                joints[n][i, 0:2] = affine_transform(
                        joints[n][i, 0:2], trans)
                if joints_vis[n][i, 0] > 0.0:
                    if (np.min(joints[n][i, :2]) < 0 or
                            joints[n][i, 0] >= self.image_size[0] or
                            joints[n][i, 1] >= self.image_size[1]):
                        joints_vis[n][i, :] = 0
            
        #################for debug#############
        if self.is_train:
            input = np.zeros((*(input.shape[1:]), 3))
            LIMBS15 = [[0, 1], [0, 2], [0, 3], [3, 4], [4, 5], [0, 9], [9, 10], \
                [10, 11], [2, 6], [2, 12], [6, 7], [7, 8], [12, 13], [13, 14]]
            for n in range(nposes):
                for i in range(len(joints[0])):
                    input = cv2.circle(input, (int(joints[n, i, 0]), int(joints[n, i, 1])), 4, (0, 255, 0), -1)
                for [x, y] in LIMBS15:
                    input = cv2.line(input, (int(joints[n, x, 0]), int(joints[n, x, 1])), (int(joints[n, y, 0]), int(joints[n, y, 1])), (0, 255, 0), 3)
            input = torch.tensor(input).permute(2, 0, 1).contiguous()
        #######################################
        input_heatmap = self.generate_input_heatmap(joints, joints_3d, self.heatmap_size)
        input_heatmap = torch.tensor(input_heatmap)

        target_heatmap, target_weight = self.generate_target_heatmap(
            joints, joints_vis)
        target_heatmap = torch.from_numpy(target_heatmap)
        target_weight = torch.from_numpy(target_weight)

        # make joints and joints_vis having same shape
        joints_u = np.zeros((self.maximum_person, self.num_joints, 2))
        joints_vis_u = np.zeros((self.maximum_person, self.num_joints, 2))
        for i in range(nposes):
            joints_u[i] = joints[i]
            joints_vis_u[i] = joints_vis[i]

        joints_3d_u = np.zeros((self.maximum_person, self.num_joints, 3))
        joints_3d_vis_u = np.zeros((self.maximum_person, self.num_joints, 3))
        for i in range(nposes):
            joints_3d_u[i] = joints_3d[i][:, 0:3]
            joints_3d_vis_u[i] = np.logical_and(joints_3d_vis[i][:, 0:3], joints_vis_u[i][:, :1])

        target_3d = self.generate_3d_target(joints_3d)
        target_3d = torch.from_numpy(target_3d)

        if isinstance(self.root_id, int):
            roots_3d = joints_3d_u[:, self.root_id]
            roots_2d = joints_u[:, self.root_id]
        elif isinstance(self.root_id, list):
            roots_3d = np.mean([joints_3d_u[:, j] for j in self.root_id], axis=0)
            roots_2d = np.mean([joints_u[:, j] for j in self.root_id], axis=0)
        
        # for detection
        wh = np.zeros((self.maximum_person, 4), dtype=np.float32)
        bboxes = np.zeros((self.maximum_person, 4), dtype=np.float32)
        depth_camera = np.zeros((self.maximum_person, 1), dtype=np.float32)
        depth_norm = np.zeros((self.maximum_person, 1), dtype=np.float32)
        pitch = np.zeros((self.maximum_person, 2), dtype=np.float32)
        ind = np.zeros((self.maximum_person, ), dtype=np.int64)
        ind_3d = np.zeros((self.maximum_person, ), dtype=np.int64)
        bias_3d = np.zeros((self.maximum_person, 3), dtype=np.float32)
        reg_mask = np.zeros((self.maximum_person, ), dtype=np.float32)
        depth_norm_factor = np.sqrt(db_rec['camera']['fx'] * db_rec['camera']['fy'] / (s[1] * 200 * s[0] * 200)\
            * (self.image_size[0] * self.image_size[1]))

        feat_stride = self.image_size[0] / self.heatmap_size[0]

        boxmap = torch.zeros(4, self.heatmap_size[1], self.heatmap_size[0]).float()
        
        for i in range(nposes):
            extention = [(joints_u[i, :, j].max() - joints_u[i, :, j].min()) * self.bbox_extention[j] for j in range(2)]
            bbox = [np.clip(joints_u[i, :, 0].min() - extention[0], 0, self.image_size[0]), \
                    np.clip(joints_u[i, :, 1].min() - extention[1], 0, self.image_size[1]),\
                    np.clip(joints_u[i, :, 0].max() + extention[0], 0, self.image_size[0]),\
                    np.clip(joints_u[i, :, 1].max() + extention[1], 0, self.image_size[1])]

            bboxes[i] = bbox / feat_stride
            _, depth = project_pose(roots_3d[i:i + 1], db_rec['camera'], need_depth=True)
            depth_camera[i] = depth
            depth_norm[i] = depth / depth_norm_factor

            wh[i] = np.array([roots_2d[i, 0] - bbox[0], roots_2d[i, 1] - bbox[1], \
                    bbox[2] - roots_2d[i, 0], bbox[3] - roots_2d[i, 1]]) / feat_stride


            if roots_2d[i, 0] < 0 or roots_2d[i, 1] < 0 or roots_2d[i, 0] >= self.image_size[0] \
                or roots_2d[i, 1] >= self.image_size[1]:
                joints_3d_vis, joints_vis_u = np.zeros_like(joints_3d_vis), np.zeros_like(joints_vis_u)
            else:
                reg_mask[i] = 1
                ind[i] = int(roots_2d[i, 1] / feat_stride) * self.heatmap_size[0] + int(roots_2d[i, 0] / feat_stride)
                ind3d = (roots_3d[i] - self.space_center + self.space_size / 2) / self.cube_length
                ind_3d[i] = round(ind3d[0]) * self.initial_cube_size[1] * self.initial_cube_size[2] +\
                    round(ind3d[1]) * self.initial_cube_size[2] + round(ind3d[2]) 


            r = 15
            boxmap[:, max(0, int(roots_2d[i, 1] / feat_stride) - r): min(self.heatmap_size[1] - 1, int(roots_2d[i, 1] / feat_stride) + r),\
                max(0, int(roots_2d[i, 0] / feat_stride) - r): min(self.heatmap_size[0] - 1, int(roots_2d[i, 0] / feat_stride) + r)] = torch.tensor(wh[i])[:, None, None]

        
        input_AGR = torch.cat((input_heatmap, boxmap), dim = 0)
        meta = {
            'image': image_file,
            'num_person': nposes,
            'joints_3d': joints_3d_u,
            'joints_3d_vis': joints_3d_vis_u,
            'roots_3d': roots_3d,
            'roots_2d': roots_2d,
            'joints': joints_u,
            'joints_vis': joints_vis_u,
            'center': c,
            'scale': s,
            'rotation': r,
            'camera': db_rec['camera'],
            'depth': depth_camera,
            'depth_norm': depth_norm,
            'depth_norm_factor': depth_norm_factor,
            'wh': wh,
            'ind': ind,
            'ind_3d': ind_3d,
            'reg_mask': reg_mask,
            'bbox': bboxes,
            'frame_valid': frame_valid,
        }

        return input, target_heatmap, target_weight, target_3d, meta, input_AGR


    def generate_input_heatmap(self, joints, joints_cam, heatmap_size = None):
        '''
        :param joints:  [[num_joints, 3]]
        :param joints_cam:  [[num_joints, 3]]
        :return: input_heatmap
        '''
        nposes = len(joints)
        num_joints = self.cfg.NETWORK.NUM_JOINTS
        if heatmap_size is None: heatmap_size = self.heatmap_size

        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'

        if self.target_type == 'gaussian':
            target = np.zeros(
                (num_joints, heatmap_size[1], heatmap_size[0]),
                dtype=np.float32)
            feat_stride = self.image_size / heatmap_size

            for n in range(nposes):
                human_scale = 2 * self.compute_human_scale(joints[n][:, 0:2] / feat_stride, np.ones((num_joints, 1)))
                if human_scale == 0:
                    continue

                if self.heatmap_sigma_scale:
                    cur_sigma = self.heatmap_sigma * np.sqrt((human_scale / (96.0 * 96.0)))
                else:
                    cur_sigma = self.heatmap_sigma * np.sqrt((self.heatmap_size[0] * self.heatmap_size[1] / (96.0 * 96.0)))
                tmp_size = cur_sigma * 3
                for joint_id in range(num_joints):
                    feat_stride = self.image_size / heatmap_size
                    if self.heatmap_noise and self.is_train:
                        noise = np.random.randn(2) * cur_sigma / 4
                    else:
                        noise = (0, 0)
                    mu_x = int(joints[n][joint_id][0] / feat_stride[0] + noise[0])
                    mu_y = int(joints[n][joint_id][1] / feat_stride[1] + noise[1])
                    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                    if ul[0] >= heatmap_size[0] or \
                            ul[1] >= heatmap_size[1] \
                            or br[0] < 0 or br[1] < 0:
                        continue

                    size = 2 * tmp_size + 1
                    x = np.arange(0, size, 1, np.float32)
                    y = x[:, np.newaxis]
                    x0 = y0 = size // 2
                    max_value = joints[n][joint_id][2] if len(joints[n][joint_id]) == 3 else 1.0

                    if self.heatmap_noise and self.is_train:
                        scale = 0.9 + np.random.randn(1) * 0.03 if np.random.random() < 0.8 else 1.0
                        if joint_id in [4, 10]:   # elbow
                            scale = scale * 0.7 if np.random.random() < 0.35 else scale
                        elif joint_id in [5, 11]: # wrist
                            scale = scale * 0.5 if np.random.random() < 0.35 else scale
                        elif joint_id in [15, 16]: # spine and head
                            scale = scale * 0.7 if np.random.random() < 0.35 else scale
                        else:
                            scale = scale * 0.7 if np.random.random() < 0.25 else scale
                        g = np.exp(
                            -((x - x0) ** 2 + (y - y0) ** 2) / (2 * cur_sigma ** 2)) * scale
                    else:
                        g = np.exp(
                            -((x - x0)**2 + (y - y0)**2) / (2 * cur_sigma**2)) * max_value

                    # Usable gaussian range
                    g_x = max(0,
                              -ul[0]), min(br[0], heatmap_size[0]) - ul[0]
                    g_y = max(0,
                              -ul[1]), min(br[1], heatmap_size[1]) - ul[1]
                    # Image range
                    img_x = max(0, ul[0]), min(br[0], heatmap_size[0])
                    img_y = max(0, ul[1]), min(br[1], heatmap_size[1])

                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]],
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
                target = np.clip(target, 0, 1)

        return target

    def random_place_people(self, max_iter = 100, max_people = 4, dist_thre = 600):
        joints_3d, roots = [], []
        pose_no = np.random.choice(len(self.pose_list), max_people, replace = False) 
        thetas = np.random.rand(max_people) * np.pi * 2
        for i in range(max_iter):
            center = np.random.uniform(-2000, 2000, size = 2)
            if len(roots) == 0 or np.linalg.norm(center[None] - np.array(roots), axis = -1).min() > dist_thre:
                roots.append(center)
                pose, theta = self.pose_list[pose_no[len(joints_3d)]], thetas[len(joints_3d)]
                center = np.array([center[0], center[1], pose[self.root_id, -1]])
                pose_rel = pose - pose[self.root_id][None]
                R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
                pose_rel = pose_rel @ R
                joints_3d.append(pose_rel + center[None])
                if len(joints_3d) >= max_people:
                    break
        
        joints_3d = np.stack(joints_3d, axis = 0)
        return joints_3d






