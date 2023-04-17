from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F
from virtualpose.utils.det_utils import _tranpose_and_gather_feat


class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += self.criterion(heatmap_pred.mul(target_weight[:, idx]),
                                       heatmap_gt.mul(target_weight[:, idx]))
            else:
                loss += self.criterion(heatmap_pred, heatmap_gt)

        return loss


class PerJointMSELoss(nn.Module):
    def __init__(self):
        super(PerJointMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, output, target, use_target_weight = False, target_weight=None):
        if use_target_weight:
            batch_size = output.size(0)
            num_joints = output.size(1)
            
            heatmap_pred = output.reshape((batch_size, num_joints, -1))
            heatmap_gt = target.reshape((batch_size, num_joints, -1))
            loss = self.criterion(heatmap_pred.mul(target_weight), heatmap_gt.mul(target_weight))
        else:
            loss = self.criterion(output, target)

        return loss


class PerJointL1Loss(nn.Module):
    def __init__(self):
        super(PerJointL1Loss, self).__init__()
        self.criterion = nn.L1Loss(reduction='mean')

    def forward(self, output, target, use_target_weight=False, target_weight=None):
        if use_target_weight:
            batch_size = output.size(0)
            num_joints = output.size(1)

            pred = output.reshape((batch_size, num_joints, -1))
            gt = target.reshape((batch_size, num_joints, -1))
            loss = self.criterion(pred.mul(target_weight), gt.mul(target_weight))
        else:
            loss = self.criterion(output, target)

        return loss

def _reg_loss(regr, gt_regr, mask):
    ''' L1 regression loss
        Arguments:
        regr (batch x max_objects x dim)
        gt_regr (batch x max_objects x dim)
        mask (batch x max_objects)
    '''
    num = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr).float()

    regr = regr * mask
    gt_regr = gt_regr * mask
        
    regr_loss = F.smooth_l1_loss(regr, gt_regr, size_average=False)
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss


class RegLoss(nn.Module):
    '''Regression loss for an output tensor
        Arguments:
        output (batch x dim x h x w)
        mask (batch x max_objects)
        ind (batch x max_objects)
        target (batch x max_objects x dim)
    '''
    def __init__(self):
        super(RegLoss, self).__init__()
    
    def forward(self, output, mask, ind, target):
        pred = _tranpose_and_gather_feat(output, ind)
        loss = _reg_loss(pred, target, mask)
        return loss

class RegL1Loss(nn.Module):
    def __init__(self):
        super(RegL1Loss, self).__init__()
    
    def forward(self, output, mask, ind, target):
        pred = _tranpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss
