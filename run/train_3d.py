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
from copy import copy
import numpy as np

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
        '--data_path', help='dataset path', default= './', type=str)
    parser.add_argument(
        '--cur_path', help='current path', default='./', type=str)
    parser.add_argument(
        '--gpus', help='num of gpus', default=2, type=int)

    args, rest = parser.parse_known_args()
    update_config(args.cfg)

    return args

def worker_init_fn_seed(worker_id):
    seed = worker_id
    np.random.seed(seed)

def get_optimizer(model):
    lr = config.TRAIN.LR
    if model.module.backbone is not None:
        for params in model.module.backbone.parameters():
            params.requires_grad = 'backbone' in config.TRAIN.SCHEME  
    for params in model.module.REN.parameters():
        params.requires_grad = 'REN' in config.TRAIN.SCHEME
    for params in model.module.PEN.parameters():
        params.requires_grad = 'PEN' in config.TRAIN.SCHEME
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.module.parameters()), lr=lr)

    return model, optimizer


def main():
    args = parse_args()
    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'train', args.cur_path)

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    gpus = [i for i in range(args.gpus)]
    print('=> Using GPUs', gpus)
    print('=> Loading data ..')
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    if isinstance(config.DATASET.TRAIN_DATASET, list):
        assert len(config.DATASET.TRAIN_DATASET) == len(config.DATASET.TRAIN_ROOT)
        dataset_rootlist = copy(config.DATASET.TRAIN_ROOT)
        dataset_list = []
        for name, path in zip(config.DATASET.TRAIN_DATASET, dataset_rootlist):
            dataset_list.append(
                eval('dataset.' + name)(
                    config, config.DATASET.TRAIN_SUBSET, True, path,
                    transforms.Compose([
                        transforms.ToTensor(),
                        normalize,
                    ], ), args.data_path)
            )
        train_dataset = eval('dataset.' + 'mixed')(
            config, dataset_list, args.data_path)
    else:
        train_dataset = eval('dataset.' + config.DATASET.TRAIN_DATASET)(
            config, config.DATASET.TRAIN_SUBSET, True, config.DATASET.TRAIN_ROOT,
            transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ], ), args.data_path)


    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE * len(gpus),
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        worker_init_fn= worker_init_fn_seed,
        pin_memory=True)
    

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
        model = torch.nn.DataParallel(model, device_ids=gpus)
        model = model.cuda()

    model, optimizer = get_optimizer(model)

    start_epoch = config.TRAIN.BEGIN_EPOCH
    end_epoch = config.TRAIN.END_EPOCH
    save_epoch = config.TRAIN.SAVE_EPOCH

    if config.NETWORK.PRETRAINED:
        pretrained_model_file = osp.join(args.data_path, config.NETWORK.PRETRAINED)
        logger.info('=> load models state {}'.format(pretrained_model_file))
        model.module.load_state_dict(torch.load(pretrained_model_file), strict = False)
    if config.NETWORK.PRETRAINED_BACKBONE:
        model = load_backbone(model, osp.join(args.cur_path, config.NETWORK.PRETRAINED_BACKBONE))
    if config.TRAIN.RESUME:
        start_epoch, model, optimizer = load_checkpoint(model, optimizer, final_output_dir)

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    print('=> Training...')
    for epoch in range(start_epoch, end_epoch):
        print('Epoch: {}'.format(epoch))

        # lr_scheduler.step()
        train_3d(config, model, optimizer, train_loader, epoch, final_output_dir, writer_dict)
        if config.TEST.NEED:
            eval_list = validate_3d(config, model, test_loader, final_output_dir, writer_dict, epoch)
            pickle.dump(eval_list, open(osp.join(final_output_dir, 'results.pkl'), 'wb'))
        else:
            best_model = False

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, final_output_dir)
        
        if save_epoch > 0 and (epoch + 1) % save_epoch == 0:
            save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, final_output_dir, epoch = epoch)

    final_model_state_file = os.path.join(final_output_dir,
                                          'final_state.pth.tar')
    logger.info('saving final model state to {}'.format(
        final_model_state_file))
    torch.save(model.module.state_dict(), final_model_state_file)

    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
