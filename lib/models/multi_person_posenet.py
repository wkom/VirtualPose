from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models import pose_resnet, posedet_resnet
from models.root_estimation_net import RootEstimationNet
from models.pose_estimation_net import PoseEstimationNet
from core.loss import PerJointMSELoss, PerJointL1Loss, RegL1Loss, RegLoss
from utils.decode import det_decode


class MultiPersonPoseNet(nn.Module):
    def __init__(self, backbone, cfg):
        super(MultiPersonPoseNet, self).__init__()
        self.root_id = cfg.DATASET.ROOTIDX
        self.num_cand = cfg.ROOT_ESTIMATION.MAX_PEOPLE_NUM
        self.num_joints = cfg.NETWORK.NUM_JOINTS

        self.backbone = backbone
        self.REN = RootEstimationNet(cfg)
        self.PEN = PoseEstimationNet(cfg)

        self.synthesize_AGR = cfg.DATASET.HEATMAP.SYNTHESIZE
        self.cfg = cfg

    def forward(self, views=None, meta=None, targets_2d=None, weights_2d=None, targets_3d=None, input_AGR=None):
        if not self.synthesize_AGR:
            AGR = self.backbone(views)
            bboxes, depths, scores, inds, xys = det_decode(AGR['pose'][:, self.root_id], \
            AGR['bbox'], AGR['depth'], K = self.num_cand)
            output_decoded = {'bboxes': bboxes, 'depths': depths * meta['depth_norm_factor'][:, None, None]}
        else:
            AGR = {'pose': input_AGR[:, :-4], 'bbox': input_AGR[:, -4:]}

        heatmaps = AGR['pose']
        batch_size, device = heatmaps.shape[0], heatmaps.device

        # calculate 2D heatmap and detection loss
        criterion = PerJointMSELoss().cuda()
        crit_reg = RegL1Loss() if self.cfg.LOSS.REG_LOSS == 'l1' else \
            RegLoss() if self.cfg.LOSS.REG_LOSS == 'sl1' else None
        
        loss_2d = loss_bbox = torch.zeros(1, device=device)
        if not self.synthesize_AGR:
            if targets_2d is not None:
                loss_2d = criterion(heatmaps, targets_2d, True, weights_2d)
            loss_bbox = crit_reg(AGR['bbox'], meta['reg_mask'], meta['ind'], meta['wh'])
        

        # Root Estimation
        loss_depth = loss_3d = torch.zeros(1, device=device)
        pred = torch.zeros(batch_size, self.num_cand, self.num_joints, 5, device=device)
        if 'REN' in self.cfg.TRAIN.SCHEME or 'PEN' in self.cfg.TRAIN.SCHEME:
            root_depth_map, root_cubes, grid_centers, output_decoded = self.REN(AGR, meta)
            pred[:, :, :, 3:] = grid_centers[:, :, 3:].reshape(batch_size, -1, 1, 2)  # matched gt

            # calculate root depth loss
            loss_depth = crit_reg(root_depth_map * meta['depth_norm_factor'][:, None, None, None], \
                meta['reg_mask'], meta['ind'], meta['depth'])
            # calculate 3D heatmap loss
            if targets_3d is not None and root_cubes is not None:
                loss_3d = criterion(root_cubes, targets_3d)
                del root_cubes
        else:
            grid_centers = torch.zeros(batch_size, self.num_cand, 5, device=device)

        # Pose Estimation
        # loss_cord = criterion(torch.zeros(1, device=device), torch.zeros(1, device=device))
        loss_cord = torch.zeros(1, device=device)
        if 'PEN' in self.cfg.TRAIN.SCHEME:
            criterion_cord = PerJointL1Loss().cuda()
            count = 0
            for n in range(self.num_cand):
                index = (pred[:, n, 0, 3] >= 0)
                if torch.sum(index) > 0:
                    single_pose = self.PEN(heatmaps, meta, grid_centers[:, n])
                    pred[:, n, :, 0:3] = single_pose.detach()

                    # calculate 3D pose loss
                    if self.training and 'joints_3d' in meta and 'joints_3d_vis' in meta:
                        gt_3d = meta['joints_3d'].float()
                        for i in range(batch_size):
                            if pred[i, n, 0, 3] >= 0:
                                targets = gt_3d[i:i + 1, pred[i, n, 0, 3].long()]
                                weights_3d = meta['joints_3d_vis'][i:i + 1, pred[i, n, 0, 3].long(), :, 0:1].float()
                                count += 1
                                loss_cord = (loss_cord * (count - 1) +
                                            criterion_cord(single_pose[i:i + 1], targets, True, weights_3d)) / count
                    del single_pose

        losses = {
            'loss_2d': loss_2d,
            'loss_bbox': loss_bbox,
            'loss_depth': loss_depth,
            'loss_3d': loss_3d,
            'loss_cord': loss_cord,
        }
        return pred, heatmaps, output_decoded, grid_centers, losses


def get_multi_person_pose_net(cfg, is_train=True):
    if cfg.BACKBONE_MODEL:
        backbone = eval(cfg.BACKBONE_MODEL + '.get_pose_net')(cfg, is_train=is_train)
    else:
        backbone = None
    model = MultiPersonPoseNet(backbone, cfg)
    return model
