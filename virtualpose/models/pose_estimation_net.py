import torch
import torch.nn as nn
import torch.nn.functional as F

from .v2v_net import V2VNet
from .project_layer import ProjectLayer


class SoftArgmaxLayer(nn.Module):
    def __init__(self, cfg):
        super(SoftArgmaxLayer, self).__init__()
        self.beta = cfg.NETWORK.BETA

    def forward(self, x, grids):
        batch_size = x.size(0)
        channel = x.size(1)
        x = x.reshape(batch_size, channel, -1, 1)
        # x = F.softmax(x, dim=2)
        x = F.softmax(self.beta * x, dim=2)
        grids = grids.unsqueeze(1)
        x = torch.mul(x, grids)
        x = torch.sum(x, dim=2)
        return x


class PoseEstimationNet(nn.Module):
    def __init__(self, cfg):
        super(PoseEstimationNet, self).__init__()
        self.grid_size = cfg.PICT_STRUCT.GRID_SIZE
        self.cube_size = cfg.PICT_STRUCT.CUBE_SIZE

        self.project_layer = ProjectLayer(cfg)
        self.v2v_net = V2VNet(cfg.NETWORK.NUM_JOINTS, cfg.NETWORK.NUM_JOINTS)
        self.soft_argmax_layer = SoftArgmaxLayer(cfg)

    def forward(self, heatmaps, meta, grid_centers):
        batch_size = heatmaps.shape[0]
        num_joints = heatmaps.shape[1]
        device = heatmaps.device
        pred = torch.zeros(batch_size, num_joints, 3, device=device)
        cubes, grids = self.project_layer(heatmaps, meta,
                                          self.grid_size, grid_centers, self.cube_size)

        index = grid_centers[:, 3] >= 0
        valid_cubes = self.v2v_net(cubes[index])
        # print(pred.shape, grids.shape, index.shape, grid_centers.shape)
        # assert 0
        pred[index] = self.soft_argmax_layer(valid_cubes, grids[index])

        return pred
