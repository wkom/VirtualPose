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

import _init_paths
from core.config import config
from core.config import update_config
from core.function import train_3d, validate_3d
from utils.utils import create_logger
from utils.utils import save_checkpoint, load_checkpoint, load_model_state
from utils.utils import load_backbone, load_backbone_validate
import dataset
import models


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument(
        '--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument(
        '--data_path', help='dataset path', default= './', type=str)
    parser.add_argument(
        '--cur_path', help='current path', default='./', type=str)
    parser.add_argument(
        '--gpus', help='num of gpus', default=2, type=int)

    args, rest = parser.parse_known_args()
    update_config(args.cfg)

    return args


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

    test_dataset = eval('dataset.' + config.DATASET.TEST_DATASET)(
        config, config.DATASET.TEST_SUBSET, False, config.DATASET.TEST_ROOT,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]), args.data_path)

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
    
    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    eval_list = validate_3d(config, model, test_loader, final_output_dir, writer_dict)
    pickle.dump(eval_list, open(osp.join(final_output_dir, 'results.pkl'), 'wb'))


if __name__ == '__main__':
    main()
