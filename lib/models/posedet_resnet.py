import torch
import torch.nn as nn
from .pose_resnet import PoseResNet, resnet_spec
import numpy as np


class PoseDetResNet(PoseResNet):
    def __init__(self, block, layers, heads, head_conv, cfg, **kwargs):
        super().__init__(block, layers, cfg, **kwargs)

        self.num_joints = cfg.NETWORK.NUM_JOINTS
        self.flip_training = cfg.TRAIN.FLIP
        self.flip_test = cfg.TEST.FLIP
        self.flip_pairs = [[3, 9], [4, 10], [5, 11], [6, 12], [7, 13], [8, 14]]
        self.flip_idx = [i for i in range(self.num_joints)]
        for flip_pair in self.flip_pairs:
            self.flip_idx[flip_pair[0]], self.flip_idx[flip_pair[1]] = flip_pair[1], flip_pair[0]
        
        self.heads = heads
        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
                fc = nn.Sequential(
                    nn.Conv2d(cfg.POSE_RESNET.NUM_DECONV_FILTERS[-1], head_conv,
                    kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, classes, 
                    kernel_size=1, stride=1, 
                    padding=0, bias=True))
            else:
                fc = nn.Conv2d(cfg.POSE_RESNET.NUM_DECONV_FILTERS[-1], classes, 
                    kernel_size=1, stride=1, 
                    padding=0, bias=True)
            self.__setattr__(head, fc)

    def forward_once(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.deconv_layers(x)

        ret = {}
        ret['pose'] = self.final_layer(x)
        for head in self.heads:
            ret[head] = self.__getattr__(head)(x)

        return ret
    
    def forward(self, x):
        # flip training
        if self.flip_training and self.training:
            is_flip = np.random.randint(2)
            if is_flip:
                x = torch.flip(x, dims = [-1])

        AGR = self.forward_once(x)

        # flip training
        if self.flip_training and self.training and is_flip:
            for key in AGR.keys():
                AGR[key] = torch.flip(AGR[key], dims = [-1])
            AGR['bbox'] = AGR['bbox'][:, [2, 1, 0, 3]].contiguous()
            AGR['pose'] = AGR['pose'][:, self.flip_idx]

        # flip test
        if self.flip_test and not self.training:
            x_flip = torch.flip(x, dims = [-1])
            AGR_flip = self.forward_once(x_flip)
            heatmap_flip = AGR_flip['pose']
            flip_idx = [i for i in range(heatmap_flip.shape[1])]
            for flip_pair in self.flip_pairs:
                flip_idx[flip_pair[0]], flip_idx[flip_pair[1]] = flip_pair[1], flip_pair[0]
            heatmap_flip = torch.flip(heatmap_flip[:, flip_idx], dims = [-1])
            AGR['pose'] = (AGR['pose'] + heatmap_flip) / 2

        return AGR


def get_pose_net(cfg, is_train, **kwargs):
    num_layers = kwargs['num_layers'] if 'num_layers' in kwargs \
        else cfg.POSE_RESNET.NUM_LAYERS

    block_class, layers = resnet_spec[num_layers]
    head_conv = 256
    heads = {'bbox': 4, 'depth': 1}
    model = PoseDetResNet(block_class, layers, heads, head_conv, cfg, **kwargs)

    if is_train:
        model.init_weights()

    return model