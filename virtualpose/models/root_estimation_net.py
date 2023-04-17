import torch
import torch.nn as nn

from .v2v_net import V2VNet
from .project_layer import ProjectLayer
from .posedet_resnet import get_pose_net as get_pose_net_single
from virtualpose.core.filter import nms
from virtualpose.utils.decode import det_decode


class FilterLayer(nn.Module):
    def __init__(self, cfg):
        super(FilterLayer, self).__init__()
        self.grid_size = torch.tensor(cfg.ROOT_ESTIMATION.SPACE_SIZE)
        self.cube_size = torch.tensor(cfg.ROOT_ESTIMATION.INITIAL_CUBE_SIZE)
        self.grid_center = torch.tensor(cfg.ROOT_ESTIMATION.SPACE_CENTER)
        self.num_cand = cfg.ROOT_ESTIMATION.MAX_PEOPLE_NUM
        self.root_id = cfg.DATASET.ROOTIDX
        self.num_joints = cfg.NETWORK.NUM_JOINTS
        self.threshold = cfg.ROOT_ESTIMATION.THRESHOLD
        self.matched_threshold = cfg.ROOT_ESTIMATION.MATCHED_THRESHOLD

    def filter_refine(self, topk_index, gt_3d, num_person):
        batch_size = topk_index.shape[0]
        cand_num = topk_index.shape[1]
        cand2gt = torch.zeros(batch_size, cand_num)

        for i in range(batch_size):
            cand = topk_index[i].reshape(cand_num, 1, -1)
            gt = gt_3d[i, :num_person[i]].reshape(1, num_person[i], -1)

            dist = torch.sqrt(torch.sum((cand - gt)**2, dim=-1))
            min_dist, min_gt = torch.min(dist, dim=-1)

            cand2gt[i] = min_gt
            cand2gt[i][min_dist > self.matched_threshold] = -1.0

        return cand2gt

    def get_real_loc(self, index):
        device = index.device
        cube_size = self.cube_size.to(device=device, dtype=torch.float)
        grid_size = self.grid_size.to(device=device)
        grid_center = self.grid_center.to(device=device)
        loc = index.float() / (cube_size - 1) * grid_size + grid_center - grid_size / 2.0
        return loc

    def forward(self, root_cubes, meta):
        batch_size = root_cubes.shape[0]

        topk_values, topk_unravel_index = nms(root_cubes.detach(), self.num_cand)
        topk_unravel_index = self.get_real_loc(topk_unravel_index)

        grid_centers = torch.zeros(batch_size, self.num_cand, 5, device=root_cubes.device)
        
        grid_centers[:, :, 0:3] = topk_unravel_index
        grid_centers[:, :, 4] = topk_values

        # match gt to filter those invalid proposals for training/validate REN
        if 'roots_3d' in meta and 'num_person' in meta and torch.any(meta['num_person'] > 0):
            gt_3d = meta['roots_3d'].float()
            num_person = meta['num_person']
            cand2gt = self.filter_refine(grid_centers[:, :, 0:3], gt_3d, num_person)
            grid_centers[:, :, 3] = cand2gt
        else:
            grid_centers[:, :, 3] = (topk_values > self.threshold).float() - 1.0  # if ground-truths are not available.

        return grid_centers


class RootEstimationNet(nn.Module):
    def __init__(self, cfg):
        super(RootEstimationNet, self).__init__()
        self.flip_test = cfg.TEST.FLIP 
        self.flip_pairs = [[3, 9], [4, 10], [5, 11], [6, 12], [7, 13], [8, 14]]
        self.depth_joints = [0, 1, 2, 3, 6, 7, 9, 12, 13]
        self.root_id = cfg.DATASET.ROOTIDX
        self.num_cand = cfg.ROOT_ESTIMATION.MAX_PEOPLE_NUM

        self.grid_size = cfg.ROOT_ESTIMATION.SPACE_SIZE
        self.cube_size = cfg.ROOT_ESTIMATION.INITIAL_CUBE_SIZE
        self.grid_center = cfg.ROOT_ESTIMATION.SPACE_CENTER
        kwargs = {
            'is_train': True,
            'keep_scale': True,
            'input_channels': cfg.DEPTH_RESNET.INPUT_CHANNELS,
            'num_layers': cfg.DEPTH_RESNET.NUM_LAYERS,
        }
        self.depth_net = get_pose_net_single(cfg, **kwargs)
        self.project_layer = ProjectLayer(cfg)
        self.v2v_net = V2VNet(cfg.NETWORK.NUM_JOINTS, 1)
        self.filter_layer = FilterLayer(cfg)


    def forward(self, AGR, meta):
        heatmaps = AGR['pose']
        depth_map = self.depth_net(heatmaps[:, self.depth_joints])['depth']
        bboxes, depths, scores, inds, xys = det_decode(heatmaps[:, self.root_id], \
            AGR['bbox'], depth_map, K = self.num_cand)
        
        initial_cubes, grids = self.project_layer(heatmaps, meta, self.grid_size,
                                    [self.grid_center], self.cube_size, bboxes, depths)
        root_cubes = self.v2v_net(initial_cubes)[:, 0]
        grid_centers = self.filter_layer(root_cubes, meta)

        output_decoded = {'bboxes': bboxes, 'depths': depths * meta['depth_norm_factor'][:, None, None], 'roots_2d': xys}
        return depth_map, root_cubes, grid_centers, output_decoded