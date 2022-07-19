from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import os
import copy

import torch
import numpy as np
import cv2

from utils.vis import save_debug_images_multi
from utils.vis import save_debug_3d_images
from utils.vis import save_debug_3d_cubes, save_debug_detection
from tqdm import tqdm
from itertools import islice

logger = logging.getLogger(__name__)


def train_3d(config, model, optimizer, loader, epoch, output_dir, writer_dict, device=torch.device('cuda'), dtype=torch.float):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_2d = AverageMeter()
    losses_bbox = AverageMeter()
    losses_depth = AverageMeter()
    losses_pitch = AverageMeter()
    losses_3d = AverageMeter()
    losses_cord = AverageMeter()
    losses_bias = AverageMeter()

    model.train()

    if model.module.backbone is not None:
        if 'backbone' not in config.TRAIN.SCHEME:
            model.module.backbone.eval()  
    if 'REN' not in config.TRAIN.SCHEME:
        model.module.REN.eval()
    if 'PEN' not in config.TRAIN.SCHEME:
        model.module.PEN.eval()

    end = time.time()
    optimizer.zero_grad()

    total_iter = config.TRAIN.MAX_SAMPLE // config.TRAIN.BATCH_SIZE if config.TRAIN.MAX_SAMPLE else len(loader)

    for i, (inputs, targets_2d, weights_2d, targets_3d, meta, input_AGR) in enumerate(tqdm(loader, ncols=100, total=total_iter)):
        data_time.update(time.time() - end)

        pred, heatmaps, output, grid_centers, loss_dict = model(views=inputs, meta=meta,
                                                                targets_2d=targets_2d,
                                                                weights_2d=weights_2d,
                                                                targets_3d=targets_3d,
                                                                input_AGR=input_AGR)

        loss_2d = loss_dict['loss_2d'].mean()
        loss_bbox = loss_dict['loss_bbox'].mean()
        loss_depth = loss_dict['loss_depth'].mean()
        loss_3d = loss_dict['loss_3d'].mean()
        loss_cord = loss_dict['loss_cord'].mean()

        losses_2d.update(loss_2d.item())
        losses_bbox.update(loss_bbox.item())
        losses_depth.update(loss_depth.item())
        losses_3d.update(loss_3d.item())
        losses_cord.update(loss_cord.item())
        loss = config.LOSS.HEATMAP_WEIGHT * loss_2d + config.LOSS.BBOX_WEIGHT * loss_bbox + config.LOSS.DEPTH_WEIGHT * loss_depth +\
             + config.LOSS.PROP_WEIGHT * loss_3d + config.LOSS.CORD_WEIGHT * loss_cord
        losses.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        batch_time.update(time.time() - end)
        end = time.time()

        if config.PRINT_FREQ > 0 and i % config.PRINT_FREQ == 0:
            gpu_memory_usage = torch.cuda.memory_allocated(0)
            if config.PRINT_LOG:
                msg = 'Epoch: [{0}][{1}/{2}]\t' \
                'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                'Speed: {speed:.1f} samples/s\t' \
                'Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                'Loss: {loss.val:.6f} ({loss.avg:.6f})\t' \
                'Loss_2d: {loss_2d.val:.7f} ({loss_2d.avg:.7f})\t' \
                'Loss_depth: {loss_depth.val:.7f} ({loss_depth.avg:.7f})\t' \
                'loss_bbox: {loss_bbox.val:.7f} ({loss_bbox.avg:.7f})\t' \
                'Loss_3d: {loss_3d.val:.7f} ({loss_3d.avg:.7f})\t' \
                'Loss_cord: {loss_cord.val:.6f} ({loss_cord.avg:.6f})\t' \
                'Memory {memory:.1f}'.format(
                    epoch, i, len(loader), batch_time=batch_time,
                    speed=len(inputs) * inputs[0].size(0) / batch_time.val,
                    data_time=data_time, loss=losses, loss_2d=losses_2d, loss_bbox = losses_bbox,
                    loss_depth = losses_depth, loss_3d=losses_3d, loss_cord=losses_cord, memory=gpu_memory_usage)
                logger.info(msg)

            prefix = '{}_{:08}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images_multi(config, inputs, meta, targets_2d, heatmaps, pred, prefix)
            prefix2 = '{}_{:08}'.format(os.path.join(output_dir, 'train'), i)

            save_debug_3d_cubes(config, meta, grid_centers, prefix2)
            save_debug_3d_images(config, meta, pred, prefix2)
            save_debug_detection(meta, inputs, output, heatmaps, targets_2d, prefix2)

        if i % config.WRITER_FREQ == 0:
            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train/loss_2d', loss_2d.item(), global_steps)
            writer.add_scalar('train/loss_depth', loss_depth.item(), global_steps)
            writer.add_scalar('train/loss_bbox', loss_bbox.item(), global_steps)
            writer.add_scalar('train/loss_3d', loss_3d.item(), global_steps)
            writer.add_scalar('train/loss_cord', loss_cord.item(), global_steps)
            writer.add_scalar('train/loss', loss.item(), global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

        if i >= total_iter - 1: break


def validate_3d(config, model, loader, output_dir, writer_dict = None, epoch = None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_depth = AverageMeter()
    losses_pitch = AverageMeter()
    losses_depth_mean = AverageMeter()
    losses_depth_max = AverageMeter()
    model.eval()

    preds, backbone_outputs, proposal_grid_centers, frame_valids = [], [], [], []
    with torch.no_grad():
        end = time.time()
        for i, (inputs, targets_2d, weights_2d, targets_3d, meta, input_AGR) in enumerate(tqdm(loader, ncols=100)):
            data_time.update(time.time() - end)
            pred, heatmaps, output, grid_centers, loss_dict = model(views=inputs, meta=meta, targets_2d=targets_2d,
                                                            weights_2d=weights_2d, targets_3d=targets_3d, input_AGR=input_AGR)
            pred = pred.detach().cpu().numpy()
            for b in range(pred.shape[0]):
                preds.append(pred[b])
                backbone_outputs.append({key: output[key][b].detach().cpu().numpy() for key in output.keys()})
                proposal_grid_centers.append(grid_centers[b].cpu().numpy())
                frame_valids.append(meta['frame_valid'][b])

            batch_time.update(time.time() - end)
            
            end = time.time()
            if config.PRINT_FREQ > 0 and (i % config.PRINT_FREQ == 0 or i == len(loader) - 1):
                gpu_memory_usage = torch.cuda.memory_allocated(0)
                if config.PRINT_LOG:
                    msg = 'Test: [{0}/{1}]\t' \
                        'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                        'Speed: {speed:.1f} samples/s\t' \
                        'Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                        'Memory {memory:.1f}'.format(
                            i, len(loader), batch_time=batch_time,
                            speed=len(inputs) * inputs[0].size(0) / batch_time.val,
                            data_time=data_time, memory=gpu_memory_usage)
                    logger.info(msg)

                prefix = '{}_{:08}'.format(os.path.join(output_dir, 'val'), i)
                save_debug_images_multi(config, inputs, meta, targets_2d, heatmaps, pred, prefix)
                prefix2 = '{}_{:08}'.format(os.path.join(output_dir, 'val'), i)

                save_debug_3d_cubes(config, meta, grid_centers, prefix2)
                save_debug_3d_images(config, meta, pred, prefix2)
                save_debug_detection(meta, inputs, output, heatmaps, targets_2d, prefix2)

    mpjpe_metric, eval_list = loader.dataset.evaluate(preds, backbone_outputs, proposal_grid_centers, frame_valids)

    if writer_dict is not None:
        writer = writer_dict['writer']
        writer.add_scalar('val/mpjpe_500mm', mpjpe_metric['mpjpe'], epoch)
        writer.add_scalar('val/mpjpe_aligned_500mm', mpjpe_metric['mpjpe_aligned'], epoch)
        writer.add_scalar('val/mrpe_500mm', mpjpe_metric['mrpe']['root'], epoch)
        writer.add_scalar('val/mrpe_z_500mm', mpjpe_metric['mrpe']['z'], epoch)

    return eval_list


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count
