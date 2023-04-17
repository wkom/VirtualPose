import math
import numpy as np
import torch
import torchvision
import cv2
import os
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from .cameras import project_pose
from .transforms import get_affine_transform as get_transform
from .transforms import affine_transform_pts_cuda as do_transform


def save_batch_image_with_joints_multi(batch_image,
                                 batch_joints,
                                 batch_joints_vis,
                                 num_person,
                                 file_name,
                                 nrow=8,
                                 padding=2):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_person, num_joints, 3],
    batch_joints_vis: [batch_size, num_person, num_joints, 1],
    num_person: [batch_size]
    }
    '''
    batch_image = batch_image.flip(1)
    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()

    nmaps = batch_image.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            for n in range(num_person[k]):
                joints = batch_joints[k, n]
                joints_vis = batch_joints_vis[k, n]

                for joint, joint_vis in zip(joints, joints_vis):
                    joint[0] = x * width + padding + joint[0]
                    joint[1] = y * height + padding + joint[1]
                    if joint_vis[0]:
                        cv2.circle(ndarr, (int(joint[0]), int(joint[1])), 2,
                                   [0, 255, 255], 2)
            k = k + 1
    cv2.imwrite(file_name, ndarr)


def save_batch_heatmaps_multi(batch_image, batch_heatmaps, file_name, normalize=True):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_heatmaps: ['batch_size, num_joints, height, width]
    file_name: saved file name
    '''
    if normalize:
        batch_image = batch_image.clone()
        min = float(batch_image.min())
        max = float(batch_image.max())

        batch_image.add_(-min).div_(max - min + 1e-5)
    batch_image = batch_image.flip(1)

    batch_size = batch_heatmaps.size(0)
    num_joints = batch_heatmaps.size(1)
    heatmap_height = batch_heatmaps.size(2)
    heatmap_width = batch_heatmaps.size(3)

    grid_image = np.zeros(
        (batch_size * heatmap_height, (num_joints + 1) * heatmap_width, 3),
        dtype=np.uint8)

    for i in range(batch_size):
        image = batch_image[i].mul(255)\
                              .clamp(0, 255)\
                              .byte()\
                              .permute(1, 2, 0)\
                              .cpu().numpy()
        heatmaps = batch_heatmaps[i].mul(255)\
                                    .clamp(0, 255)\
                                    .byte()\
                                    .cpu().numpy()

        resized_image = cv2.resize(image,
                                   (int(heatmap_width), int(heatmap_height)))

        height_begin = heatmap_height * i
        height_end = heatmap_height * (i + 1)
        for j in range(num_joints):
            heatmap = heatmaps[j, :, :]
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            masked_image = colored_heatmap * 0.7 + resized_image * 0.3

            width_begin = heatmap_width * (j + 1)
            width_end = heatmap_width * (j + 2)
            grid_image[height_begin:height_end, width_begin:width_end, :] = \
                masked_image
            # grid_image[height_begin:height_end, width_begin:width_end, :] = \
            #     colored_heatmap*0.7 + resized_image*0.3

        grid_image[height_begin:height_end, 0:heatmap_width, :] = resized_image

    cv2.imwrite(file_name, grid_image)


def save_debug_images_multi(config, input, meta, target, output, pred, prefix):
    if not config.DEBUG.DEBUG:
        return

    basename = os.path.basename(prefix)
    dirname = os.path.dirname(prefix)
    dirname1 = os.path.join(dirname, 'image_with_joints')
    dirname2 = os.path.join(dirname, 'batch_heatmaps')

    for dir in [dirname1, dirname2]:
        if not os.path.exists(dir):
            os.makedirs(dir)

    prefix1 = os.path.join(dirname1, basename)
    prefix2 = os.path.join(dirname2, basename)

    if config.DEBUG.SAVE_BATCH_IMAGES_GT:
        save_batch_image_with_joints_multi(input, meta['joints'], meta['joints_vis'], meta['num_person'], '{}_gt.jpg'.format(prefix1))
    if config.DEBUG.SAVE_BATCH_IMAGES_PRED:
        pose2d = torch.zeros_like(meta['joints']).cuda()
        if isinstance(pred, torch.Tensor):
            pred_torch = pred.clone()
        else:
            pred_torch = torch.tensor(pred.copy()).cuda()
        joints_vis = torch.ones_like(meta['joints_vis'])
        num_person = torch.zeros_like(meta['num_person'])
        for i in range(pose2d.shape[0]):
            camera = {k: v[i] for k, v in meta['camera'].items()}
            center = meta['center'][i]
            scale = meta['scale'][i]

            width, height = center * 2
            trans = torch.as_tensor(
                get_transform(center, scale, 0, config.NETWORK.IMAGE_SIZE),
                dtype=torch.float,
                device=pose2d.device)
            xy = project_pose(pred_torch[i, ..., :3].view(-1, 3), camera)
            pose2d[i] = do_transform(xy, trans).view(pose2d.shape[1:])
            
            for np in range(pred_torch.shape[1]):
                if pred_torch[i, np, 0, 4] >= 0:
                    num_person[i] += 1 
        save_batch_image_with_joints_multi(input, pose2d, joints_vis, num_person, '{}_pred.jpg'.format(prefix1))
    if config.DEBUG.SAVE_HEATMAPS_GT:
        save_batch_heatmaps_multi(input, target, '{}_hm_gt.jpg'.format(prefix2))
    if config.DEBUG.SAVE_HEATMAPS_PRED:
        save_batch_heatmaps_multi(input, output, '{}_hm_pred.jpg'.format(prefix2))

# panoptic
LIMBS15 = [[0, 1], [0, 2], [0, 3], [3, 4], [4, 5], [0, 9], [9, 10],
         [10, 11], [2, 6], [2, 12], [6, 7], [7, 8], [12, 13], [13, 14]]

# # h36m
# LIMBS17 = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8],
#          [8, 9], [9, 10], [8, 14], [14, 15], [15, 16], [8, 11], [11, 12], [12, 13]]
# coco17
LIMBS17 = [[0, 1], [0, 2], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [11, 13], [13, 15],
        [6, 12], [12, 14], [14, 16], [5, 6], [11, 12]]

# shelf / campus
LIMBS14 = [[0, 1], [1, 2], [3, 4], [4, 5], [2, 3], [6, 7], [7, 8], [9, 10],
          [10, 11], [2, 8], [3, 9], [8, 12], [9, 12], [12, 13]]


def save_debug_3d_images(config, meta, preds, prefix):
    if not config.DEBUG.DEBUG:
        return

    basename = os.path.basename(prefix)
    dirname = os.path.dirname(prefix)
    dirname1 = os.path.join(dirname, '3d_joints')

    if not os.path.exists(dirname1):
        os.makedirs(dirname1)

    prefix = os.path.join(dirname1, basename)
    file_name = prefix + "_3d.png"

    # preds = preds.cpu().numpy()
    batch_size = meta['num_person'].shape[0]
    xplot = min(4, batch_size)
    yplot = int(math.ceil(float(batch_size) / xplot))

    width = 4.0 * xplot
    height = 4.0 * yplot
    fig = plt.figure(0, figsize=(width, height))
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05,
                        top=0.95, wspace=0.05, hspace=0.15)
    for i in range(batch_size):
        num_person = meta['num_person'][i]
        joints_3d = meta['joints_3d'][i]
        joints_3d_vis = meta['joints_3d_vis'][i]
        ax = plt.subplot(yplot, xplot, i + 1, projection='3d')
        for n in range(num_person):
            joint = joints_3d[n, :15]
            joint_vis = joints_3d_vis[n]
            for k in eval("LIMBS{}".format(len(joint))):
                if joint_vis[k[0], 0] and joint_vis[k[1], 0]:
                    x = [float(joint[k[0], 0]), float(joint[k[1], 0])]
                    y = [float(joint[k[0], 1]), float(joint[k[1], 1])]
                    z = [float(joint[k[0], 2]), float(joint[k[1], 2])]
                    ax.plot(x, y, z, c='r', lw=1.5, marker='o', markerfacecolor='w', markersize=2,
                            markeredgewidth=1)
                else:
                    x = [float(joint[k[0], 0]), float(joint[k[1], 0])]
                    y = [float(joint[k[0], 1]), float(joint[k[1], 1])]
                    z = [float(joint[k[0], 2]), float(joint[k[1], 2])]
                    ax.plot(x, y, z, c='r', ls='--', lw=1.5, marker='o', markerfacecolor='w', markersize=2,
                            markeredgewidth=1)

        colors = ['b', 'g', 'c', 'y', 'm', 'orange', 'pink', 'royalblue', 'lightgreen', 'gold']
        if preds is not None:
            pred = preds[i]
            for n in range(len(pred)):
                joint = pred[n, :15]
                if joint[0, 3] >= 0:
                    for k in eval("LIMBS{}".format(len(joint))):
                        x = [float(joint[k[0], 0]), float(joint[k[1], 0])]
                        y = [float(joint[k[0], 1]), float(joint[k[1], 1])]
                        z = [float(joint[k[0], 2]), float(joint[k[1], 2])]
                        ax.plot(x, y, z, c=colors[int(n % 10)], lw=1.5, marker='o', markerfacecolor='w', markersize=2,
                                markeredgewidth=1)
    plt.savefig(file_name)
    plt.close(0)


def save_debug_3d_cubes(config, meta, root, prefix):
    if not config.DEBUG.DEBUG:
        return

    basename = os.path.basename(prefix)
    dirname = os.path.dirname(prefix)
    dirname1 = os.path.join(dirname, 'root_cubes')

    if not os.path.exists(dirname1):
        os.makedirs(dirname1)

    prefix = os.path.join(dirname1, basename)
    file_name = prefix + "_root.png"

    batch_size = root.shape[0]
    root_id = config.DATASET.ROOTIDX

    xplot = min(4, batch_size)
    yplot = int(math.ceil(float(batch_size) / xplot))

    width = 6.0 * xplot
    height = 4.0 * yplot
    fig = plt.figure(0, figsize=(width, height))
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05,
                        top=0.95, wspace=0.05, hspace=0.15)
    for i in range(batch_size):
        roots_gt = meta['roots_3d'][i]
        num_person = meta['num_person'][i]
        roots_pred = root[i]
        ax = plt.subplot(yplot, xplot, i + 1, projection='3d')

        x = roots_gt[:num_person, 0].cpu()
        y = roots_gt[:num_person, 1].cpu()
        z = roots_gt[:num_person, 2].cpu()
        ax.scatter(x, y, z, c='r')

        index = roots_pred[:, 3] >= 0
        x = roots_pred[index, 0].cpu()
        y = roots_pred[index, 1].cpu()
        z = roots_pred[index, 2].cpu()
        ax.scatter(x, y, z, c='b')

        space_size = config.ROOT_ESTIMATION.SPACE_SIZE
        space_center = config.ROOT_ESTIMATION.SPACE_CENTER
        ax.set_xlim(space_center[0] - space_size[0] / 2, space_center[0] + space_size[0] / 2)
        ax.set_ylim(space_center[1] - space_size[1] / 2, space_center[1] + space_size[1] / 2)
        ax.set_zlim(space_center[2] - space_size[2] / 2, space_center[2] + space_size[2] / 2)

    plt.savefig(file_name)
    plt.close(0)


def save_debug_detection(meta, image_batch, output, heatmap_batch, target_batch, prefix, normalize=True):
    basename = os.path.basename(prefix)
    dirname = os.path.dirname(prefix)
    dirname1 = os.path.join(dirname, 'detection')

    if not os.path.exists(dirname1):
        os.makedirs(dirname1)

    prefix = os.path.join(dirname1, basename)

    vis_img_batch, vis_gt_batch = [], []
    img_paths = meta['image']
    gt_bbox, gt_depth = meta['bbox'], meta['depth']
    bbox_batch, depth_batch = output['bboxes'], output['depths']
    if normalize:
        image_batch = image_batch.clone()
        min = float(image_batch.min())
        max = float(image_batch.max())
        image_batch.add_(-min).div_(max - min + 1e-5)
    image_batch = image_batch.flip(1).mul(255).clamp(0, 255).byte()
    for i in range(len(img_paths)):
        img = image_batch[i].permute(1, 2, 0).cpu().numpy()
        img_origin = img.copy()
        scale = (img.shape[0] / heatmap_batch.shape[-2], img.shape[1] / heatmap_batch.shape[-1])
        img = cv2.resize(img, None, fx=1 / scale[1], fy=1 / scale[0])
        img_det = img_origin.copy()

        for bbox, depth in zip(bbox_batch[i], depth_batch[i]):
            img_det = cv2.putText(img_det, '%.2fmm'%depth, (int(bbox[0] * scale[1]), int(bbox[1] * scale[0] - 5)),\
                cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)
            img_det = cv2.rectangle(img_det, (int(bbox[0] * scale[1]), int(bbox[1] * scale[0])), (int(bbox[2] * scale[1]), int(bbox[3] * scale[0])), \
                (255, 0, 0), 1)
            
        img_heatmaps = []
        for heatmap in heatmap_batch[i]:
            heatmap_color = cv2.applyColorMap((heatmap.detach().cpu().numpy() * 255).astype(np.uint8), cv2.COLORMAP_JET)
            img_heatmaps.append(img * 0.6 + heatmap_color * 0.4)
        vis_heatmap = np.concatenate(img_heatmaps, axis = 1)
        vis_det = img_det

        img_det = img_origin.copy()
        for bbox, depth in zip(gt_bbox[i], gt_depth[i]):
            img_det = cv2.putText(img_det, '%.2fmm'%depth, (int(bbox[0] * scale[1]), int(bbox[1] * scale[0] - 5)),\
                cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)
            img_det = cv2.rectangle(img_det, (int(bbox[0] * scale[1]), int(bbox[1] * scale[0])), (int(bbox[2] * scale[1]), int(bbox[3] * scale[0])),\
                (255, 0, 0), 1)
        img_heatmaps = []
        for heatmap in target_batch[i]:
            heatmap_color = cv2.applyColorMap((heatmap.detach().cpu().numpy() * 255).astype(np.uint8), cv2.COLORMAP_JET)
            img_heatmaps.append(img * 0.6 + heatmap_color * 0.4)
        vis_heatmap_gt = np.concatenate(img_heatmaps, axis = 1)
        vis_det_gt = img_det

        vis_img_batch.append(np.concatenate((vis_heatmap, vis_heatmap_gt), axis = 0))
        vis_gt_batch.append(np.concatenate((vis_det, vis_det_gt), axis = 0))


    cv2.imwrite(prefix + '_hm.png', np.concatenate(vis_img_batch, axis = 0))
    cv2.imwrite(prefix + '_det.png', np.concatenate(vis_gt_batch, axis = 1))
