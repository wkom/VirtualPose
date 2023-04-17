from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import yaml

import numpy as np
from easydict import EasyDict as edict

config = edict()

config.OUTPUT_DIR = 'output'
config.LOG_DIR = 'log'
config.DATA_DIR = ''
config.BACKBONE_MODEL = 'pose_resnet'
config.MODEL = 'multi_person_posenet'
config.WORKERS = 12
config.WRITER_FREQ = 1
config.PRINT_FREQ = 100
config.PRINT_LOG = True
config.EXPERIMENT_NAME = ''

# higherhrnet definition
config.MODEL_EXTRA = edict()
config.MODEL_EXTRA.PRETRAINED_LAYERS = ['*']
config.MODEL_EXTRA.FINAL_CONV_KERNEL = 1
config.MODEL_EXTRA.STEM_INPLANES = 64

config.MODEL_EXTRA.STAGE2 = edict()
config.MODEL_EXTRA.STAGE2.NUM_MODULES = 1
config.MODEL_EXTRA.STAGE2.NUM_BRANCHES= 2
config.MODEL_EXTRA.STAGE2.BLOCK = 'BASIC'
config.MODEL_EXTRA.STAGE2.NUM_BLOCKS = [4, 4]
config.MODEL_EXTRA.STAGE2.NUM_CHANNELS = [48, 96]
config.MODEL_EXTRA.STAGE2.FUSE_METHOD = 'SUM'

config.MODEL_EXTRA.STAGE3 = edict()
config.MODEL_EXTRA.STAGE3.NUM_MODULES = 4
config.MODEL_EXTRA.STAGE3.NUM_BRANCHES = 3
config.MODEL_EXTRA.STAGE3.BLOCK = 'BASIC'
config.MODEL_EXTRA.STAGE3.NUM_BLOCKS = [4, 4, 4]
config.MODEL_EXTRA.STAGE3.NUM_CHANNELS = [48, 96, 192]
config.MODEL_EXTRA.STAGE3.FUSE_METHOD = 'SUM'

config.MODEL_EXTRA.STAGE4 = edict()
config.MODEL_EXTRA.STAGE4.NUM_MODULES = 3
config.MODEL_EXTRA.STAGE4.NUM_BRANCHES = 4
config.MODEL_EXTRA.STAGE4.BLOCK = 'BASIC'
config.MODEL_EXTRA.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
config.MODEL_EXTRA.STAGE4.NUM_CHANNELS = [48, 96, 192, 384]
config.MODEL_EXTRA.STAGE4.FUSE_METHOD = 'SUM'

config.MODEL_EXTRA.DECONV = edict()
config.MODEL_EXTRA.DECONV.NUM_DECONVS = 1
config.MODEL_EXTRA.DECONV.NUM_CHANNELS = 32
config.MODEL_EXTRA.DECONV.KERNEL_SIZE = 4
config.MODEL_EXTRA.DECONV.NUM_BASIC_BLOCKS = 4
config.MODEL_EXTRA.DECONV.CAT_OUTPUT = True

# Cudnn related params
config.CUDNN = edict()
config.CUDNN.BENCHMARK = True
config.CUDNN.DETERMINISTIC = False
config.CUDNN.ENABLED = True

# common params for NETWORK
config.NETWORK = edict()
config.NETWORK.PRETRAINED = 'models/pytorch/imagenet/resnet50-19c8e357.pth'
config.NETWORK.PRETRAINED_BACKBONE = ''
config.NETWORK.NUM_JOINTS = 20
config.NETWORK.INPUT_SIZE = 512
config.NETWORK.HEATMAP_SIZE = np.array([80, 80])
config.NETWORK.IMAGE_SIZE = np.array([320, 320])
config.NETWORK.TARGET_TYPE = 'gaussian'
config.NETWORK.BETA = 100.0

# pose_resnet related params
config.POSE_RESNET = edict()
config.POSE_RESNET.INPUT_CHANNELS = 3
config.POSE_RESNET.NUM_LAYERS = 50
config.POSE_RESNET.DECONV_WITH_BIAS = False
config.POSE_RESNET.NUM_DECONV_LAYERS = 3
config.POSE_RESNET.NUM_DECONV_FILTERS = [256, 256, 256]
config.POSE_RESNET.NUM_DECONV_KERNELS = [4, 4, 4]
config.POSE_RESNET.FINAL_CONV_KERNEL = 1

config.DEPTH_RESNET = edict()
config.DEPTH_RESNET.INPUT_CHANNELS = 19
config.DEPTH_RESNET.NUM_LAYERS = 18


# proposal network related params
config.ROOT_ESTIMATION = edict()
config.ROOT_ESTIMATION.SPACE_SIZE = np.array([4000.0, 5200.0, 2400.0])
config.ROOT_ESTIMATION.SPACE_CENTER = np.array([300.0, 300.0, 300.0])
config.ROOT_ESTIMATION.INITIAL_CUBE_SIZE = np.array([24, 32, 16])
config.ROOT_ESTIMATION.MAX_PEOPLE_NUM = 10
config.ROOT_ESTIMATION.THRESHOLD = 0.1
config.ROOT_ESTIMATION.MATCHED_THRESHOLD = 500

config.LOSS = edict()
config.LOSS.USE_TARGET_WEIGHT = True
config.LOSS.USE_DIFFERENT_JOINTS_WEIGHT = False
config.LOSS.REG_LOSS = 'l1'
config.LOSS.HEATMAP_WEIGHT = 10000
config.LOSS.BBOX_WEIGHT = 0.01
config.LOSS.DEPTH_WEIGHT = 0.1
config.LOSS.PROP_WEIGHT = 1000
config.LOSS.CORD_WEIGHT = 1

# DATASET related params
config.DATASET = edict()
config.DATASET.TRAIN_ROOT = ''
config.DATASET.TEST_ROOT = ''
config.DATASET.TRAIN_DATASET = ''
config.DATASET.TEST_DATASET = ''
config.DATASET.TRAIN_SUBSET = 'train'
config.DATASET.TEST_SUBSET = 'validation'
config.DATASET.ROOTIDX = 2
config.DATASET.DATA_FORMAT = 'jpg'
config.DATASET.BBOX = 2000
config.DATASET.CROP = True
config.DATASET.COLOR_RGB = False
config.DATASET.DATA_AUGMENTATION = True
config.DATASET.BBOX_EXTENTION = [0.25, 0.12]
config.DATASET.TRAIN_VIEW = []
config.DATASET.VAL_VIEW = []

config.DATASET.HEATMAP = edict()
config.DATASET.HEATMAP.SYNTHESIZE = False
config.DATASET.HEATMAP.FIX_CAMERA = False
config.DATASET.HEATMAP.SIGMA_SCALE = True
config.DATASET.HEATMAP.NOISE = False
config.DATASET.HEATMAP.SIGMA = 3

# training data augmentation
config.DATASET.SCALE_FACTOR = 0
config.DATASET.ROT_FACTOR = 0
config.DATASET.SCALE_FACTOR_RANGE = []

config.DATASET.GROUND_CENTER = False
config.DATASET.RANDOM_PLACE = False

# train
config.TRAIN = edict()
config.TRAIN.LR_FACTOR = 0.1
config.TRAIN.LR_STEP = [90, 110]
config.TRAIN.LR = 0.001

config.TRAIN.OPTIMIZER = 'adam'
config.TRAIN.MOMENTUM = 0.9
config.TRAIN.WD = 0.0001
config.TRAIN.NESTEROV = False

config.TRAIN.BEGIN_EPOCH = 0
config.TRAIN.END_EPOCH = 140
config.TRAIN.SAVE_EPOCH = 0

config.TRAIN.RESUME = False

config.TRAIN.BATCH_SIZE = 8
config.TRAIN.MAX_SAMPLE = None
config.TRAIN.SHUFFLE = True
config.TRAIN.SCHEME = ['REN','PEN']
config.TRAIN.FLIP = True

# testing
config.TEST = edict()
config.TEST.NEED = True
config.TEST.BATCH_SIZE = 8
config.TEST.STATE = 'last'
config.TEST.BBOX_FILE = ''
config.TEST.MODEL_FILE = ''
config.TEST.FLIP = True

# debug
config.DEBUG = edict()
config.DEBUG.DEBUG = True
config.DEBUG.SAVE_BATCH_IMAGES_GT = True
config.DEBUG.SAVE_BATCH_IMAGES_PRED = True
config.DEBUG.SAVE_HEATMAPS_GT = True
config.DEBUG.SAVE_HEATMAPS_PRED = True

# pictorial structure
config.PICT_STRUCT = edict()
config.PICT_STRUCT.GRID_SIZE = [2000.0, 2000.0, 2000.0]
config.PICT_STRUCT.CUBE_SIZE = [64, 64, 64]



def _update_dict(cfg, k, v):
    if k == 'DATASET':
        if 'MEAN' in v and v['MEAN']:
            v['MEAN'] = np.array(
                [eval(x) if isinstance(x, str) else x for x in v['MEAN']])
        if 'STD' in v and v['STD']:
            v['STD'] = np.array(
                [eval(x) if isinstance(x, str) else x for x in v['STD']])
    if k == 'NETWORK':
        if 'HEATMAP_SIZE' in v:
            if isinstance(v['HEATMAP_SIZE'], int):
                v['HEATMAP_SIZE'] = np.array(
                    [v['HEATMAP_SIZE'], v['HEATMAP_SIZE']])
            else:
                v['HEATMAP_SIZE'] = np.array(v['HEATMAP_SIZE'])
        if 'IMAGE_SIZE' in v:
            if isinstance(v['IMAGE_SIZE'], int):
                v['IMAGE_SIZE'] = np.array([v['IMAGE_SIZE'], v['IMAGE_SIZE']])
            else:
                v['IMAGE_SIZE'] = np.array(v['IMAGE_SIZE'])
    for vk, vv in v.items():
        if vk in cfg[k]:
            if isinstance(vv, dict) and vk != 'PRETRAINED_BACKBONE':
                _update_dict(cfg[k], vk, vv)
            else:
                cfg[k][vk] = vv
        else:
            raise ValueError("{}.{} not exist in config.py".format(k, vk))


def update_config(config_file):
    exp_config = None
    with open(config_file) as f:
        exp_config = edict(yaml.load(f, Loader=yaml.FullLoader))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    _update_dict(config, k, v)
                else:
                    if k == 'SCALES':
                        config[k][0] = (tuple(v))
                    else:
                        config[k] = v
            else:
                raise ValueError("{} not exist in config.py".format(k))


def gen_config(config_file):
    cfg = dict(config)
    for k, v in cfg.items():
        if isinstance(v, edict):
            cfg[k] = dict(v)

    with open(config_file, 'w') as f:
        yaml.dump(dict(cfg), f, default_flow_style=False)


def update_dir(model_dir, log_dir, data_dir):
    if model_dir:
        config.OUTPUT_DIR = model_dir

    if log_dir:
        config.LOG_DIR = log_dir

    if data_dir:
        config.DATA_DIR = data_dir

    config.DATASET.ROOT = os.path.join(config.DATA_DIR, config.DATASET.ROOT)

    config.TEST.BBOX_FILE = os.path.join(config.DATA_DIR, config.TEST.BBOX_FILE)

    config.NETWORK.PRETRAINED = os.path.join(config.DATA_DIR,
                                             config.NETWORK.PRETRAINED)


def get_model_name(cfg):
    name = '{model}_{num_layers}'.format(
        model=cfg.MODEL, num_layers=cfg.POSE_RESNET.NUM_LAYERS)
    deconv_suffix = ''.join(
        'd{}'.format(num_filters)
        for num_filters in cfg.POSE_RESNET.NUM_DECONV_FILTERS)
    full_name = '{height}x{width}_{name}_{deconv_suffix}'.format(
        height=cfg.NETWORK.IMAGE_SIZE[1],
        width=cfg.NETWORK.IMAGE_SIZE[0],
        name=name,
        deconv_suffix=deconv_suffix)

    return name, full_name


if __name__ == '__main__':
    import sys
    gen_config(sys.argv[1])
