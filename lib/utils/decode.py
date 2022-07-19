from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from .det_utils import _gather_feat, _tranpose_and_gather_feat
import numpy as np

def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def _topk_channel(scores, K=40):
      batch, height, width = scores.size()
      
      topk_scores, topk_inds = torch.topk(scores.view(batch, -1), K)

      topk_inds = topk_inds % (height * width)
      topk_ys   = (topk_inds / width).int().float()
      topk_xs   = (topk_inds % width).int().float()

      return topk_scores, topk_inds, topk_ys, topk_xs

def _topk(scores, K=40):
    batch, height, width = scores.size()
      
    topk_scores, topk_inds = torch.topk(scores.view(batch, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys   = (topk_inds / width).int().float()
    topk_xs   = (topk_inds % width).int().float()
      
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def det_decode(heat, wh, depth, K=100):
    batch, height, width = heat.size()

    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    heat = _nms(heat)

    scores, inds, clses, ys, xs = _topk(heat, K=K)
    xs = xs.view(batch, K, 1) + 0.5
    ys = ys.view(batch, K, 1) + 0.5
        
    wh = _tranpose_and_gather_feat(wh, inds).clamp(min = 0.)
    wh = wh.view(batch, K, 4)
    depth = _tranpose_and_gather_feat(depth, inds).clamp(min = 0.)
    depth = depth.view(batch, K, 1)

    scores = scores.view(batch, K, 1)

    
    bboxes = torch.cat([xs - wh[..., 0:1],
                        ys - wh[..., 1:2],
                        xs + wh[..., 2:3],
                        ys + wh[..., 3:4]], dim=2)
    
    depth[scores < 0.1] = -1
    bboxes[scores[..., 0] < 0.1] = 0
    # print(bboxes)
    # assert 0
    sorted_indices = torch.argsort(depth, dim = 1, descending=True) # (b, k, 1)
    bboxes = bboxes.gather(1, sorted_indices.expand_as(bboxes))
    scores = scores.gather(1, sorted_indices.expand_as(scores))
    depths = depth.gather(1, sorted_indices.expand_as(depth))
    inds = inds.gather(1, sorted_indices.squeeze(-1))
    xs = xs.gather(1, sorted_indices.expand_as(xs))
    ys = ys.gather(1, sorted_indices.expand_as(ys))

    return bboxes, depths, scores, inds, torch.cat((xs, ys), dim = -1)
