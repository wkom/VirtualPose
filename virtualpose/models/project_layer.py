import torch
import torch.nn as nn
import torch.nn.functional as F

import virtualpose.utils.cameras as cameras
from virtualpose.utils.transforms import get_affine_transform as get_transform
from virtualpose.utils.transforms import affine_transform_pts_cuda as do_transform


class ProjectLayer(nn.Module):
    def __init__(self, cfg):
        super(ProjectLayer, self).__init__()

        self.img_size = cfg.NETWORK.IMAGE_SIZE
        self.heatmap_size = cfg.NETWORK.HEATMAP_SIZE
        self.grid_size = cfg.ROOT_ESTIMATION.SPACE_SIZE
        self.cube_size = cfg.ROOT_ESTIMATION.INITIAL_CUBE_SIZE
        self.grid_center = cfg.ROOT_ESTIMATION.SPACE_CENTER
        self.sigma = 200.0

    def compute_grid(self, boxSize, boxCenter, nBins, device=None):
        if isinstance(boxSize, int) or isinstance(boxSize, float):
            boxSize = [boxSize, boxSize, boxSize]
        if isinstance(nBins, int):
            nBins = [nBins, nBins, nBins]

        grid1Dx = torch.linspace(-boxSize[0] / 2, boxSize[0] / 2, nBins[0], device=device)
        grid1Dy = torch.linspace(-boxSize[1] / 2, boxSize[1] / 2, nBins[1], device=device)
        grid1Dz = torch.linspace(-boxSize[2] / 2, boxSize[2] / 2, nBins[2], device=device)
        gridx, gridy, gridz = torch.meshgrid(
            grid1Dx + boxCenter[0],
            grid1Dy + boxCenter[1],
            grid1Dz + boxCenter[2],
        )
        gridx = gridx.contiguous().view(-1, 1)
        gridy = gridy.contiguous().view(-1, 1)
        gridz = gridz.contiguous().view(-1, 1)
        grid = torch.cat([gridx, gridy, gridz], dim=1)
        return grid

    def get_voxel(self, heatmaps, meta, grid_size, grid_center, cube_size, bboxes, depths):
        device = heatmaps.device
        batch_size = heatmaps.shape[0]
        num_joints = heatmaps.shape[1]
        nbins = cube_size[0] * cube_size[1] * cube_size[2]
        cubes = torch.zeros(batch_size, num_joints, 1, nbins, device=device)
        # h, w = heatmaps[0].shape[2], heatmaps[0].shape[3]
        w, h = self.heatmap_size
        grids = torch.zeros(batch_size, nbins, 3, device=device)
        bounding = torch.zeros(batch_size, 1, 1, nbins, device=device, dtype = torch.bool)
        for i in range(batch_size):
            if len(grid_center[0]) == 3 or grid_center[i][3] >= 0:
                # This part of the code can be optimized because the projection operation is time-consuming.
                # If the camera locations always keep the same, the grids and sample_grids are repeated across frames
                # and can be computed only one time.
                if len(grid_center) == 1:
                    grid = self.compute_grid(grid_size, grid_center[0], cube_size, device=device)
                else:
                    grid = self.compute_grid(grid_size, grid_center[i], cube_size, device=device)
                grids[i:i + 1] = grid

                center = meta['center'][i]
                scale = meta['scale'][i]

                width, height = center * 2
                trans = torch.as_tensor(
                    get_transform(center, scale, 0, self.img_size),
                    dtype=torch.float,
                    device=device)
                cam = {}
                for k, v in meta['camera'].items():
                    cam[k] = v[i]
                xy, z = cameras.project_pose(grid, cam, need_depth=True)
                bounding[i, 0, 0, :] = (xy[:, 0] >= 0) & (xy[:, 1] >= 0) & (xy[:, 0] < width) & (
                                xy[:, 1] < height)
                xy = torch.clamp(xy, -1.0, max(width, height))
                xy = do_transform(xy, trans)
                xy = xy * torch.tensor(
                    [w, h], dtype=torch.float, device=device) / torch.tensor(
                    self.img_size, dtype=torch.float, device=device)

                sample_grid = xy / torch.tensor(
                        [w - 1, h - 1], dtype=torch.float,
                        device=device) * 2.0 - 1.0
                sample_grid = torch.clamp(sample_grid.view(1, 1, nbins, 2), -1.1, 1.1)

                if depths is None or bboxes is None:
                    # # # if pytorch version < 1.3.0, align_corners=True should be omitted.
                    cubes[i:i + 1, :, :, :] = F.grid_sample(heatmaps[i:i + 1, :, :, :], sample_grid, align_corners=True)
                else:
                    for j in range(bboxes[i].shape[0]):
                        if depths[i][j] < 0:
                            break
            
                        tmp_bounding = (xy[:, 0] >= bboxes[i, j, 0]) & (xy[:, 1] >= bboxes[i, j, 1]) & (
                            xy[:, 0] <= bboxes[i, j, 2]) & (xy[:, 1] <= bboxes[i, j, 3]) & bounding[i, 0, 0, :]
                        depth_mask = torch.exp(-(z - depths[i][j] * meta['depth_norm_factor'][i]) ** 2 / (2 * self.sigma ** 2))
                        sample_grid = xy / torch.tensor(
                            [w - 1, h - 1], dtype=torch.float,
                            device=device) * 2.0 - 1.0
                        sample_grid = torch.clamp(sample_grid.view(1, 1, nbins, 2), -1.1, 1.1)
                        tmp_feature = tmp_bounding * depth_mask * F.grid_sample(heatmaps[i:i + 1, :, :, :], sample_grid, align_corners=True)
                        mask = tmp_feature.detach() > cubes[i:i + 1, :, :, :].detach()
                        cubes[i:i + 1, :, :, :] = mask * tmp_feature + ~mask * cubes[i:i + 1, :, :, :]

        cubes = torch.mul(cubes, bounding)
        cubes[cubes != cubes] = 0.0
        cubes = cubes.clamp(0.0, 1.0)

        cubes = cubes.view(batch_size, num_joints, cube_size[0], cube_size[1], cube_size[2])  ##
        return cubes, grids

    def forward(self, heatmaps, meta, grid_size, grid_center, cube_size, bboxes=None, depths=None):
        cubes, grids = self.get_voxel(heatmaps, meta, grid_size, grid_center, cube_size, bboxes, depths)
        return cubes, grids