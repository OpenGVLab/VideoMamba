import math
import time
import sys
from typing import Iterable
import torch
import torch.nn as nn
import utils
from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

def train_one_epoch(
        model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
        device: torch.device, epoch: int, loss_scaler, amp_autocast, max_norm: float = 0,
        log_writer=None, lr_scheduler=None, start_steps=None,
        lr_schedule_values=None, wd_schedule_values=None, 
        teacher_model=None, clip_input_resolution=224,
        clip_loss_type='l2', clip_loss_ratio=0.5,
        mask_type='tube', mask_ratio=0.,
        bf16=False,
    ):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    if clip_loss_type == 'mse':
        loss_func_clip = nn.MSELoss()
    elif clip_loss_type == 'smooth_l1':
        loss_func_clip = nn.SmoothL1Loss()

    data_type = torch.bfloat16 if bf16 else torch.float16

    for step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        videos, bool_masked_pos = batch
        videos = videos.to(device, non_blocking=True)
        if mask_type in ['attention']:
            bool_masked_pos = None
        else:
            bool_masked_pos = bool_masked_pos.flatten(1)
            bool_masked_pos = torch.cat((torch.zeros(bool_masked_pos.shape[0], 1), bool_masked_pos), dim=1)
            bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).to(torch.bool)

        with torch.no_grad():
            # calculate the predicted CLIP features
            B, C, T, H, W = videos.shape
            if H != clip_input_resolution:
                clip_videos = torch.nn.functional.interpolate(
                    videos.view(B, C*T, H, W), 
                    size=(clip_input_resolution, clip_input_resolution), 
                    mode='bicubic', align_corners=False
                )
                clip_videos = clip_videos.view(B, C, T, clip_input_resolution, clip_input_resolution)
            else:
                clip_videos = videos
                
            with torch.cuda.amp.autocast(dtype=data_type):
                if bool_masked_pos is None:
                    norm_clip, attn = teacher_model(clip_videos)
                else:
                    norm_clip = teacher_model(clip_videos)

            if mask_type == 'attention':
                BT, N = attn.shape
                N_vis = N - int(N * mask_ratio)
                importance = torch.multinomial(attn, N)
                bool_masked_pos = torch.ones((BT, N))
                pos1 = torch.arange(BT).view(-1, 1).repeat(1, N_vis)
                pos2 = importance[:, :N_vis]
                bool_masked_pos[pos1, pos2] = 0
                bool_masked_pos = bool_masked_pos.view(B, -1)
                bool_masked_pos = torch.cat((torch.zeros(B, 1), bool_masked_pos), dim=1)
                bool_masked_pos = bool_masked_pos.to(torch.bool)
                    
            C_CLIP = norm_clip.shape[-1]
            if len(norm_clip.shape) == 4:
                K = norm_clip.shape[0]
                clip_bool_masked_pos = bool_masked_pos.unsqueeze(0).repeat(K, 1, 1)
                targets_clip_vis = norm_clip[~clip_bool_masked_pos].reshape(K, B, -1, C_CLIP)
            else:
                clip_bool_masked_pos = bool_masked_pos
                targets_clip_vis = norm_clip[~clip_bool_masked_pos].reshape(B, -1, C_CLIP)
            targets_clip = targets_clip_vis

        with amp_autocast:
            outputs_clip = model(videos, bool_masked_pos)
            loss_pixel = torch.zeros(1).type_as(outputs_clip).to(outputs_clip.device)
            # align CLIP
            if clip_loss_type == 'l2':
                loss_clip = (2 - 2 * (outputs_clip * targets_clip).sum(dim=-1)).mean()
            elif clip_loss_type in ['mse', 'smooth_l1']:
                loss_clip = loss_func_clip(input=outputs_clip, target=targets_clip)
            else:
                raise NotImplementedError

        loss = loss_pixel + clip_loss_ratio * loss_clip
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_pixel=loss_pixel.item())
        metric_logger.update(loss_clip=loss_clip.item())
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_pixel=loss_pixel.item(), head="loss_pixel")
            log_writer.update(loss_clip=loss_clip, head="loss_clip")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    timestep = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{timestep}] Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
