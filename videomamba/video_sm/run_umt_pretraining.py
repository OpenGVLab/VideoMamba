import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
from functools import partial

from pathlib import Path
from timm.models import create_model
from optim_factory import create_optimizer
from datasets import build_pretraining_dataset
from engines.engine_for_pretraining_umt import train_one_epoch
from utils import NativeScalerWithGradNormCount as NativeScaler
from utils import multiple_pretrain_samples_collate
import utils
from models import *


def get_args():
    parser = argparse.ArgumentParser('VideoMAE pre-training script', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=800, type=int)
    parser.add_argument('--save_ckpt_freq', default=50, type=int)

    # Model parameters
    parser.add_argument('--model', default='pretrain_videomae_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--decoder_depth', default=4, type=int,
                        help='depth of decoder')
    parser.add_argument('--mask_type', default='tube', choices=['random', 'tube', 'attention'],
                        type=str, help='masked strategy of video tokens/patches')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='ratio of the visual tokens/patches need be masked')
    parser.add_argument('--input_size', default=224, type=int,
                        help='videos input size for backbone')
    parser.add_argument('--drop_path', type=float, default=0.0, metavar='PCT',
                        help='Drop path rate (default: 0.0)')
    parser.add_argument('--normlize_target', default=True, type=bool,
                        help='normalized the target patch pixels')
    parser.add_argument('--tubelet_size', default=2, type=int,
                        help='temporal tube size for the patch embedding')
    parser.add_argument('--use_learnable_pos_emb', action='store_true')
    parser.set_defaults(use_learnable_pos_emb=False)

    # CLIP decpder parameters
    parser.add_argument('--clip_teacher', default='clip_b16', type=str,
                        help='Name of CLIP teacher')
    parser.add_argument('--clip_input_resolution', default=224, type=int,
                        help='input resolution of CLIP decoder')
    parser.add_argument('--clip_loss_ratio', default=1., type=float,
                        help='ratio for CLIP loss, pixel_loss + RATIO * clip_loss')
    parser.add_argument('--clip_loss_type', default='l2', type=str,
                        help='type of CLIP loss')
    parser.add_argument('--clip_decoder_type', default='SA_Decoder', type=str,
                        help='type of CLIP decoder')
    parser.add_argument('--clip_decoder_embed_dim', default=512, type=int,
                        help='embedding dimension of CLIP decoder')
    parser.add_argument('--clip_output_dim', default=768, type=int,
                        help='output dimension of CLIP decoder')
    parser.add_argument('--clip_norm_type', default='l2', type=str,
                        help='type of feature normalization')
    parser.add_argument('--clip_return_attn', default=False, type=bool,
                        help='whether return CLIP attention')
    parser.add_argument('--clip_return_layer', default=1, type=int,
                        help='number of CLIP return layers')
    parser.add_argument('--clip_return_interval', default=1, type=float,
                        help='interval of CLIP teacher return layers')
    parser.add_argument('--clip_student_return_interval', default=1, type=float,
                        help='interval of CLIP student return layers')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD. 
        (Set the same value with args.weight_decay to keep weight decay no change)""")

    parser.add_argument('--lr', type=float, default=1.5e-4, metavar='LR',
                        help='learning rate (default: 1.5e-4)')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--use_checkpoint', action='store_true')
    parser.set_defaults(use_checkpoint=False)
    parser.add_argument('--checkpoint_num', type=int, default=0)

    # Augmentation parameters
    parser.add_argument('--num_sample', type=int, default=1, help='Repeated_aug (default: 1)')
    parser.add_argument('--color_jitter', type=float, default=0.0, metavar='PCT',
                        help='Color jitter factor (default: 0.0)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    parser.add_argument('--flip', default=False,
                        help='whether flip the video in pretraining')

    # Dataset parameters
    parser.add_argument('--prefix', default='', type=str, help='prefix for data')
    parser.add_argument('--split', default=' ', type=str, help='split for metadata')
    parser.add_argument('--data_path', default='you_data_path', type=str,
                        help='dataset path')
    parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')
    parser.add_argument('--use_decord', default=True,
                        help='whether use decord to load video, otherwise load image')
    parser.add_argument('--num_segments', type=int, default=1)
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--sampling_rate', type=int, default=4)
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--test_best', action='store_true',
                        help='Whether test the best model')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    return parser.parse_args()


def get_model(args):
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        use_learnable_pos_emb=args.use_learnable_pos_emb,
        pretrained=False,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        use_checkpoint=args.use_checkpoint,
        checkpoint_num=args.checkpoint_num,
        clip_decoder_embed_dim=args.clip_decoder_embed_dim,
        clip_output_dim=args.clip_output_dim,
        clip_norm_type=args.clip_norm_type,
        num_frames=args.num_frames,
        tubelet_size=args.tubelet_size,
        clip_return_layer=args.clip_return_layer,
        clip_student_return_interval=args.clip_student_return_interval
    )
    return model


def main(args):
    utils.init_distributed_mode(args)

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    model = get_model(args)
    patch_size = model.encoder.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    print("Tubelet size = %s" % str(args.tubelet_size))
    args.window_size = (args.num_frames // args.tubelet_size, args.input_size // patch_size[0], args.input_size // patch_size[1])
    args.patch_size = patch_size

    # teacher model
    print(f'Teacher model: {args.clip_teacher}')
    print(f'Loss ratio: {args.clip_loss_ratio}')
    print(f'Loss type: {args.clip_loss_type}')
    
    teacher_model = eval(args.clip_teacher)(
        clip_norm_type=args.clip_norm_type,
        input_resolution=args.clip_input_resolution,
        return_attn=args.clip_return_attn,
        clip_return_layer=args.clip_return_layer,
        clip_return_interval=args.clip_return_interval
    )

    # get dataset
    dataset_train = build_pretraining_dataset(args)

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    sampler_rank = global_rank
    num_training_steps_per_epoch = len(dataset_train) // args.batch_size // num_tasks

    sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=num_tasks, rank=sampler_rank, shuffle=True)
    print("Sampler_train = %s" % str(sampler_train))

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    if args.num_sample > 1:
        collate_func = partial(multiple_pretrain_samples_collate, fold=False)
    else:
        collate_func = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        collate_fn=collate_func,
        worker_init_fn=utils.seed_worker,
        persistent_workers=True
    )

    model.to(device)
    teacher_model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params: {} M'.format(n_parameters / 1e6))

    total_batch_size = args.batch_size * utils.get_world_size()

    args.lr = args.lr * total_batch_size * args.num_sample / 256
    args.min_lr = args.min_lr * total_batch_size * args.num_sample / 256
    args.warmup_lr = args.warmup_lr * total_batch_size * args.num_sample / 256
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Repeated sample = %d" % args.num_sample)
    print("Number of training steps = %d" % num_training_steps_per_epoch)
    print("Number of training examples per epoch = %d" % (total_batch_size * num_training_steps_per_epoch))

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module

    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()

    print("Use step level LR & WD scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    utils.auto_load_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
    torch.cuda.empty_cache()
    print(f"Start training for {args.epochs} epochs")
    print(f"Mask ratio: {args.mask_ratio}")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            teacher_model=teacher_model, 
            clip_input_resolution=args.clip_input_resolution,
            clip_loss_type=args.clip_loss_type, 
            clip_loss_ratio=args.clip_loss_ratio,
            mask_type=args.mask_type,
            mask_ratio=args.mask_ratio,
        )
        if args.output_dir:
            # if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
            #     utils.save_model(
            #         args=args, model=model, model_without_ddp=model_without_ddp, 
            #         optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch
            #     )
            utils.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, 
                optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch,
                model_name='latest'
            )

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, 'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    opts = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts)
