from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from os import path as osp
import logging
import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import shutil

from core.config import get_model_name


def create_logger(cfg, cfg_path, phase='train', cur_path='./'):
    cur_path = Path(cur_path)
    root_output_dir = (cur_path / cfg.OUTPUT_DIR).resolve()  ##
    tensorboard_log_dir = (cur_path / cfg.LOG_DIR).resolve()
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.DATASET.TRAIN_DATASET
    if isinstance(dataset, list):
        name = 'mix'
        for term in dataset:
            name += '_' + term
        dataset = name

    model, _ = get_model_name(cfg)
    cfg_name = os.path.basename(cfg_path).split('.')[0]

    if cfg.EXPERIMENT_NAME != '':
        final_output_dir = root_output_dir / dataset / model / cfg.EXPERIMENT_NAME / cfg_name
    else:
        final_output_dir = root_output_dir / dataset / model / cfg_name

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    tz = datetime.timezone(datetime.timedelta(hours=8))
    time_str = str(datetime.datetime.now(tz))[:-16].replace(' ', '-').replace(':', '-')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    if cfg.EXPERIMENT_NAME != '':
        tensorboard_log_dir = tensorboard_log_dir / dataset / model / cfg.EXPERIMENT_NAME / (cfg_name + time_str)
    else:
        tensorboard_log_dir = tensorboard_log_dir / dataset / model / (cfg_name + time_str)
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(cfg_path, tensorboard_log_dir)

    tensorboard_log_dir = str(tensorboard_log_dir)


    return logger, str(final_output_dir), tensorboard_log_dir

def get_optimizer(cfg, model):
    optimizer = None
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            nesterov=cfg.TRAIN.NESTEROV
        )
    elif cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.TRAIN.LR
        )

    return optimizer


def load_model_state(model, output_dir, epoch):
    file = os.path.join(output_dir, 'checkpoint_3d_epoch'+str(epoch)+'.pth.tar')
    if os.path.isfile(file):
        model.module.load_state_dict(torch.load(file, map_location=torch.device('cpu')))
        print('=> load models state {} (epoch {})'
              .format(file, epoch))
        return model
    else:
        print('=> no checkpoint found at {}'.format(file))
        return model


def load_checkpoint(model, optimizer, output_dir, filename='checkpoint.pth.tar'):
    file = os.path.join(output_dir, filename)
    if os.path.isfile(file):
        checkpoint = torch.load(file, map_location=torch.device('cpu'))
        start_epoch = checkpoint['epoch']
        model.module.load_state_dict(checkpoint['state_dict'], strict = False)
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
        except:
            print('Fail to load optimizer from', file)
        print('=> load checkpoint {} (epoch {})'
              .format(file, start_epoch))

        return start_epoch, model, optimizer

    else:
        print('=> no checkpoint found at {}'.format(file))
        return 0, model, optimizer


def save_checkpoint(states, output_dir,
                    filename='checkpoint.pth.tar', epoch=None):
    if epoch is None:
        torch.save(states, os.path.join(output_dir, filename))
        torch.save(states['state_dict'], os.path.join(output_dir, 'model_last.pth.tar'))
    else:
        torch.save(states['state_dict'], os.path.join(output_dir, 'model_%03d.pth.tar'%epoch))


def load_backbone(model, pretrained_file):
    pretrained_state_dict = torch.load(pretrained_file, map_location=torch.device('cpu'))
    if 'state_dict' in pretrained_state_dict:
        pretrained_state_dict = pretrained_state_dict['state_dict']
    model_state_dict = model.module.backbone.state_dict()

    prefix = "module."
    new_pretrained_state_dict = {}
    for k, v in pretrained_state_dict.items():
        if k.replace(prefix, "") in model_state_dict and v.shape == model_state_dict[k.replace(prefix, "")].shape:
            new_pretrained_state_dict[k.replace(prefix, "")] = v
        elif k.replace(prefix, "") == "final_layer.weight": 
            print("Reiniting final layer filters:", k)

            o = torch.zeros_like(model_state_dict[k.replace(prefix, "")][:, :, :, :])
            nn.init.xavier_uniform_(o)
            n_filters = min(o.shape[0], v.shape[0])
            o[:n_filters, :, :, :] = v[:n_filters, :, :, :]

            new_pretrained_state_dict[k.replace(prefix, "")] = o
        elif k.replace(prefix, "") == "final_layer.bias":
            print("Reiniting final layer biases:", k)
            o = torch.zeros_like(model_state_dict[k.replace(prefix, "")][:])
            nn.init.zeros_(o)
            n_filters = min(o.shape[0], v.shape[0])
            o[:n_filters] = v[:n_filters]

            new_pretrained_state_dict[k.replace(prefix, "")] = o
        elif k.replace('backbone.', '') in model_state_dict and v.shape == model_state_dict[k.replace('backbone.', '')].shape:
            new_pretrained_state_dict[k.replace('backbone.', '')] = v
        elif k.replace('keypoint_head.', '') in model_state_dict and v.shape == model_state_dict[k.replace('keypoint_head.', '')].shape:
            new_pretrained_state_dict[k.replace('keypoint_head.', '')] = v
        elif k.startswith('1.') and v.shape == model_state_dict[k[2:]].shape:
                new_pretrained_state_dict[k[2:]] = v
    logging.info("load backbone statedict from {}".format(pretrained_file))

    try:
        model.module.backbone.load_state_dict(new_pretrained_state_dict)
    except:
        print('load backbone statedict in unstrict mode')
        model.module.backbone.load_state_dict(new_pretrained_state_dict, strict = False)

    return model

def load_backbone_validate(model, pretrained_file):
    print("=> load backbone statedict from {}".format(pretrained_file))
    state_dict = torch.load(open(pretrained_file, 'rb'), map_location='cpu')
    new_state_dict = {k: v for k, v in state_dict.items() if 'backbone.' in k}
    model.module.load_state_dict(new_state_dict, strict = False)
    return model
