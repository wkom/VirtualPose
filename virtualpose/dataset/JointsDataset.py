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

logger = logging.getLogger(__name__)


class JointsDataset(Dataset):

    def __init__(self, cfg, image_set, is_train, dataset_root, transform=None, dataset_path='./'):
        self.cfg = cfg
        self.num_joints = 0
        self.pixel_std = 200
        self.maximum_person = cfg.ROOT_ESTIMATION.MAX_PEOPLE_NUM

        self.is_train = is_train

        dataset_root = os.path.join(dataset_path, dataset_root)
        self.dataset_root = os.path.abspath(dataset_root)
        self.root_id = cfg.DATASET.ROOTIDX
        self.image_set = image_set
        self.dataset_name = cfg.DATASET.TEST_DATASET

        self.data_format = cfg.DATASET.DATA_FORMAT
        self.data_augmentation = cfg.DATASET.DATA_AUGMENTATION
        self.heatmap_noise = cfg.DATASET.HEATMAP.NOISE
        self.heatmap_sigma_scale = cfg.DATASET.HEATMAP.SIGMA_SCALE
        self.heatmap_sigma = cfg.DATASET.HEATMAP.SIGMA
        self.fix_camera = cfg.DATASET.HEATMAP.FIX_CAMERA
        self.random_place = cfg.DATASET.RANDOM_PLACE

        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.scale_factor_range = cfg.DATASET.SCALE_FACTOR_RANGE
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.color_rgb = cfg.DATASET.COLOR_RGB

        self.target_type = cfg.NETWORK.TARGET_TYPE
        self.image_size = np.array(cfg.NETWORK.IMAGE_SIZE)
        self.heatmap_size = np.array(cfg.NETWORK.HEATMAP_SIZE)
        self.use_different_joints_weight = cfg.LOSS.USE_DIFFERENT_JOINTS_WEIGHT
        self.joints_weight = 1

        self.transform = transform
        self.db = []

        self.space_size = np.array(cfg.ROOT_ESTIMATION.SPACE_SIZE)
        self.space_center = np.array(cfg.ROOT_ESTIMATION.SPACE_CENTER)
        self.initial_cube_size = np.array(cfg.ROOT_ESTIMATION.INITIAL_CUBE_SIZE)
        self.cube_length = self.space_size / (self.initial_cube_size - 1)
        self.bbox_extention = cfg.DATASET.BBOX_EXTENTION


    def _get_db(self):
        raise NotImplementedError

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        raise NotImplementedError

    def __len__(self,):
        return len(self.db)

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
            if data_numpy is None:
                assert 0, image_file

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

        input_AGR = torch.zeros(self.cfg.NETWORK.NUM_JOINTS, self.heatmap_size[1], self.heatmap_size[0])

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
        

        for i in range(nposes):
            if 'bbox' in db_rec:
                bbox = db_rec['bbox'][i]
                for j in range(2):
                    bbox[j] = affine_transform(bbox[j], trans)
                bbox = bbox.reshape(4)
            else:
                visible_joints = joints_vis_u[i, :, 0] > 0
                if visible_joints.sum() == 0:
                    visible_joints[0] = 1
                    
                extention = [(joints_u[i, visible_joints, j].max() - joints_u[i, visible_joints, j].min()) * self.bbox_extention[j] for j in range(2)]
                bbox = [np.clip(joints_u[i, visible_joints, 0].min() - extention[0], 0, self.image_size[0]), \
                        np.clip(joints_u[i, visible_joints, 1].min() - extention[1], 0, self.image_size[1]),\
                        np.clip(joints_u[i, visible_joints, 0].max() + extention[0], 0, self.image_size[0]),\
                        np.clip(joints_u[i, visible_joints, 1].max() + extention[1], 0, self.image_size[1])]

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
            'trans': trans,
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

    def compute_human_scale(self, pose, joints_vis):
        idx = joints_vis[:, 0] == 1
        if np.sum(idx) == 0:
            return 0
        minx, maxx = np.min(pose[idx, 0]), np.max(pose[idx, 0])
        miny, maxy = np.min(pose[idx, 1]), np.max(pose[idx, 1])
        return np.clip(np.maximum(maxy - miny, maxx - minx)**2,  1.0 / 4 * 96**2, 4 * 96**2)

    def generate_target_heatmap(self, joints, joints_vis):
        '''
        :param joints:  [[num_joints, 3]]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        nposes = len(joints)
        num_joints = self.num_joints
        target_weight = np.zeros((num_joints, 1), dtype=np.float32)

        try:
            for i in range(num_joints):
                for n in range(nposes):
                    if joints_vis[n][i, 0] == 1:
                        target_weight[i, 0] = 1
        except:
            assert 0, (joints.shape, joints_vis.shape)

        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'

        if self.target_type == 'gaussian':
            target = np.zeros(
                (num_joints, self.heatmap_size[1], self.heatmap_size[0]),
                dtype=np.float32)
            feat_stride = self.image_size / self.heatmap_size

            for n in range(nposes):
                # human_scale = 2 * self.compute_human_scale(joints[n] / feat_stride, np.ones_like(joints_vis[n]))
                human_scale = 2 * self.compute_human_scale(joints[n] / feat_stride, joints_vis[n])
                if human_scale == 0:
                    continue
                
                if self.heatmap_sigma_scale:
                    cur_sigma = self.heatmap_sigma * np.sqrt((human_scale / (96.0 * 96.0)))
                else:
                    cur_sigma = self.heatmap_sigma * np.sqrt((self.heatmap_size[0] * self.heatmap_size[1] / (96.0 * 96.0)))
                tmp_size = cur_sigma * 3
                for joint_id in range(num_joints):
                    feat_stride = self.image_size / self.heatmap_size
                    mu_x = int(joints[n][joint_id][0] / feat_stride[0])
                    mu_y = int(joints[n][joint_id][1] / feat_stride[1])
                    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                    if (joints_vis[n][joint_id, 0] == 0) or \
                            ul[0] >= self.heatmap_size[0] or \
                            ul[1] >= self.heatmap_size[1] \
                            or br[0] < 0 or br[1] < 0:
                        continue

                    size = 2 * tmp_size + 1
                    x = np.arange(0, size, 1, np.float32)
                    y = x[:, np.newaxis]
                    x0 = y0 = size // 2
                    g = np.exp(
                        -((x - x0)**2 + (y - y0)**2) / (2 * cur_sigma**2))

                    # Usable gaussian range
                    g_x = max(0,
                              -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                    g_y = max(0,
                              -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                    # Image range
                    img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                    img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]],
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
                target = np.clip(target, 0, 1)

        if self.use_different_joints_weight:
            target_weight = np.multiply(target_weight, self.joints_weight)

        return target, target_weight

    def generate_3d_target(self, joints_3d):
        num_people = len(joints_3d)

        space_size = self.space_size
        space_center = self.space_center
        cube_size = self.initial_cube_size
        grid1Dx = np.linspace(-space_size[0] / 2, space_size[0] / 2, cube_size[0]) + space_center[0]
        grid1Dy = np.linspace(-space_size[1] / 2, space_size[1] / 2, cube_size[1]) + space_center[1]
        grid1Dz = np.linspace(-space_size[2] / 2, space_size[2] / 2, cube_size[2]) + space_center[2]

        target = np.zeros((cube_size[0], cube_size[1], cube_size[2]), dtype=np.float32)
        cur_sigma = 200.0

        for n in range(num_people):
            joint_id = self.root_id  # mid-hip
            if isinstance(joint_id, int):
                mu_x = joints_3d[n][joint_id][0]
                mu_y = joints_3d[n][joint_id][1]
                mu_z = joints_3d[n][joint_id][2]
            elif isinstance(joint_id, list):
                mu_x = (joints_3d[n][joint_id[0]][0] + joints_3d[n][joint_id[1]][0]) / 2.0
                mu_y = (joints_3d[n][joint_id[0]][1] + joints_3d[n][joint_id[1]][1]) / 2.0
                mu_z = (joints_3d[n][joint_id[0]][2] + joints_3d[n][joint_id[1]][2]) / 2.0
            i_x = [np.searchsorted(grid1Dx,  mu_x - 3 * cur_sigma),
                       np.searchsorted(grid1Dx,  mu_x + 3 * cur_sigma, 'right')]
            i_y = [np.searchsorted(grid1Dy,  mu_y - 3 * cur_sigma),
                       np.searchsorted(grid1Dy,  mu_y + 3 * cur_sigma, 'right')]
            i_z = [np.searchsorted(grid1Dz,  mu_z - 3 * cur_sigma),
                       np.searchsorted(grid1Dz,  mu_z + 3 * cur_sigma, 'right')]
            if i_x[0] >= i_x[1] or i_y[0] >= i_y[1] or i_z[0] >= i_z[1]:
                continue

            gridx, gridy, gridz = np.meshgrid(grid1Dx[i_x[0]:i_x[1]], grid1Dy[i_y[0]:i_y[1]], grid1Dz[i_z[0]:i_z[1]], indexing='ij')
            g = np.exp(-((gridx - mu_x) ** 2 + (gridy - mu_y) ** 2 + (gridz - mu_z) ** 2) / (2 * cur_sigma ** 2))
            target[i_x[0]:i_x[1], i_y[0]:i_y[1], i_z[0]:i_z[1]] = np.maximum(target[i_x[0]:i_x[1], i_y[0]:i_y[1], i_z[0]:i_z[1]], g)

        target = np.clip(target, 0, 1)
        return target






