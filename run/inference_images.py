from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import argparse
import os
from os import path as osp
import pprint
import logging
import json
import pickle
from tqdm import tqdm
import numpy as np
import cv2
from pathlib import Path

from virtualpose.core.config import config
from virtualpose.core.config import update_config
from virtualpose.core.function import train_3d, validate_3d
from virtualpose.utils.utils import create_logger
from virtualpose.utils.utils import save_checkpoint, load_checkpoint, load_model_state
from virtualpose.utils.utils import load_backbone, load_backbone_validate
from virtualpose.utils.transforms import inverse_affine_transform_pts_cuda
from virtualpose import dataset, models


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument(
        '--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument(
        '--image_dir', help='image directory', default= './', type=str)
    parser.add_argument(
        '--data_path', help='dataset path', default= './', type=str)
    parser.add_argument(
        '--cur_path', help='current path', default='./', type=str)
    parser.add_argument(
        '--focal_length', help='focal length', default=1700, type=float)
    parser.add_argument(
        '--gpus', help='num of gpus', default=2, type=int)

    args, rest = parser.parse_known_args()
    update_config(args.cfg)

    return args

def output2original_scale(cfg, meta, output, vis=True):
    img_paths, trans_batch = meta['image'], meta['trans']
    bbox_batch, depth_batch, roots_2d = output['bboxes'], output['depths'], output['roots_2d']
    outputs_original_scale = {}

    scale = torch.tensor((cfg.NETWORK.IMAGE_SIZE[0] / cfg.NETWORK.HEATMAP_SIZE[0], \
                        cfg.NETWORK.IMAGE_SIZE[1] / cfg.NETWORK.HEATMAP_SIZE[1]), \
                        device=bbox_batch.device, dtype=torch.float32)

    for i, img_path in enumerate(img_paths):
        if vis:
            img = cv2.imread(img_path)
        trans = trans_batch[i].to(bbox_batch[i].device).float()
        outputs_original_scale[img_path] = []

        for bbox, depth, root_2d in zip(bbox_batch[i], depth_batch[i], roots_2d[i]):
            if torch.all(bbox == 0):
                break
            bbox = (bbox.view(-1, 2) * scale[None, [1, 0]]).view(-1)
            root_2d *= scale[[1, 0]]
            bbox_origin = inverse_affine_transform_pts_cuda(bbox.view(-1, 2), trans).reshape(-1)
            roots_2d_origin = inverse_affine_transform_pts_cuda(root_2d.view(-1, 2), trans).reshape(-1)

            outputs_original_scale[img_path].append({
                'bbox': bbox_origin.cpu().numpy(), # (4,)
                'root_2d': roots_2d_origin.cpu().numpy(), # (2,)
                'depth': depth.cpu().numpy(), # (1, )
            })

            if vis:
                img = cv2.putText(img, '%.2fmm'%depth, (int(bbox_origin[0]), int(bbox_origin[1] - 5)),\
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)
                img = cv2.rectangle(img, (int(bbox_origin[0]), int(bbox_origin[1])), (int(bbox_origin[2]), int(bbox_origin[3])), \
                    (255, 0, 0), 1)
                img = cv2.circle(img, (int(roots_2d_origin[0]), int(roots_2d_origin[1])), 5, (0, 0, 255), -1)
        
        if vis:
            vis_path = 'vis_result'
            Path(vis_path).mkdir(exist_ok=True)
            cv2.imwrite(f'{vis_path}/origin_det_{i}.jpg', img)
    
    return outputs_original_scale


def inference(config, model, loader):
    model.eval()

    output_original_scale = {}
    with torch.no_grad():
        for i, (inputs, targets_2d, weights_2d, targets_3d, meta, input_AGR) in enumerate(tqdm(loader, ncols=100)):
            pred, heatmaps, output, grid_centers, loss_dict = model(views=inputs, meta=meta, targets_2d=targets_2d,
                                                            weights_2d=weights_2d, targets_3d=targets_3d, input_AGR=input_AGR)
            
            output_original_scale.update(output2original_scale(config, meta, output))

    return output_original_scale

def main():
    args = parse_args()
    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'validate')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    gpus = [i for i in range(args.gpus)]
    print('=> Using GPUs', gpus)
    print('=> Loading data ..')
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    test_dataset = dataset.images(
        config, args.image_dir, focal_length=args.focal_length,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True)

    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    print('=> Constructing models ..')
    model = eval('models.' + config.MODEL + '.get_multi_person_pose_net')(
        config, is_train=True)

    with torch.no_grad():
        model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    
    test_model_file = os.path.join(final_output_dir, config.TEST.MODEL_FILE)
    if config.TEST.MODEL_FILE and os.path.isfile(test_model_file):
        logger.info('=> load models state {}'.format(test_model_file))
        model.module.load_state_dict(torch.load(test_model_file))
    elif config.NETWORK.PRETRAINED:
        pretrained_model_file = osp.join(args.data_path, config.NETWORK.PRETRAINED)
        logger.info('=> load models state {}'.format(pretrained_model_file))
        state_dict = torch.load(pretrained_model_file)
        new_state_dict = {k:v for k, v in state_dict.items() if 'backbone.pose_branch.' not in k}
        model.module.load_state_dict(new_state_dict, strict = False)
    else:
        raise ValueError('Check the model file for testing!')

    if config.NETWORK.PRETRAINED_BACKBONE:
        model = load_backbone_validate(model, osp.join(args.cur_path, config.NETWORK.PRETRAINED_BACKBONE))
    

    output_original_scale = inference(config, model, test_loader)


if __name__ == '__main__':
    main()
