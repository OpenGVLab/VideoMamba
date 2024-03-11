import copy
import logging
import os
import os.path as osp
from os.path import join

import torch
from torch.utils.data import ConcatDataset, DataLoader

from models.backbones.bert.tokenization_bert import BertTokenizer
from utils.optimizer import create_optimizer
from utils.scheduler import create_scheduler

logger = logging.getLogger(__name__)


def get_media_types(datasources):
    """get the media types for for all the dataloaders.

    Args:
        datasources (List): List of dataloaders or datasets.

    Returns: List. The media_types.

    """
    if isinstance(datasources[0], DataLoader):
        datasets = [dataloader.dataset for dataloader in datasources]
    else:
        datasets = datasources
    media_types = [
        dataset.datasets[0].media_type
        if isinstance(dataset, ConcatDataset)
        else dataset.media_type
        for dataset in datasets
    ]

    return media_types


def setup_model(
    config, model_cls, has_decoder=False, pretrain=False, find_unused_parameters=False
):
    logger.info("Creating model")
    config = copy.deepcopy(config)

    if "bert" in config.model.text_encoder.name:
        tokenizer = BertTokenizer.from_pretrained(config.model.text_encoder.pretrained, local_files_only=True)
    else:
        raise ValueError(f"Not supported text encoder.")

    model = model_cls(config=config, tokenizer=tokenizer, is_pretrain=pretrain)

    model = model.to(torch.device(config.device))
    if config.fp16:
        if config.get('bf16', True):
            logger.info("Change to bfloat16 for model")
            model = model.to(torch.bfloat16)
        else:
            logger.info("Change to float16 for model")
            model = model.half()

    model_without_ddp = model
    if config.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[config.gpu],
            find_unused_parameters=find_unused_parameters,  # `False` for image-only task
        )

    optimizer = create_optimizer(config.optimizer, model)
    scheduler = create_scheduler(config.scheduler, optimizer)
    scaler = torch.cuda.amp.GradScaler(enabled=config.fp16) # This is never used actually if we fixed bf16

    start_epoch = 0
    global_step = 0

    # auto resume the latest checkpoint
    if config.get("auto_resume", False):
        logger.info("Auto resuming")
        model_latest = join(config.output_dir, "ckpt_latest.pth")
        model_best = join(config.output_dir, "ckpt_best.pth")
        large_num = -1
        for p in os.listdir(config.output_dir):
            if 'ckpt' in p:
                num = p.split('_')[1].split('.')[0]
                if str.isnumeric(num):
                    if int(num) > large_num:
                        large_num = int(num)
        if large_num != -1:
            model_latest = join(config.output_dir, f"ckpt_{large_num:02d}.pth")
        if osp.isfile(model_latest):
            config.pretrained_path = model_latest
            config.resume = True
        elif osp.isfile(model_best):
            config.pretrained_path = model_best
            config.resume = True
        else:
            logger.info(f"Not found checkpoint in {config.output_dir}")

    if osp.isfile(config.pretrained_path):
        logger.info(f"Loading checkpoint from {config.pretrained_path}")
        checkpoint = torch.load(config.pretrained_path, map_location="cpu")
        if 'model' in checkpoint.keys():
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        if config.resume:
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            scaler.load_state_dict(checkpoint["scaler"])
            start_epoch = checkpoint["epoch"] + 1
            global_step = checkpoint["global_step"]
        elif not pretrain:  # downstream init from pretrained ckpt

            # interpolate positional embeddings.
            if "vit" in config.model.vision_encoder.name or "videomamba" in config.model.vision_encoder.name:
                pass
            else:
                raise ValueError(
                    f" vision encoder: {config.model.vision_encoder.name} not implelented"
                )
            if not config.evaluate or config.get("zero_shot", False):  # finetuning from a pretarined weights.
                for key in list(state_dict.keys()):
                    if "bert" in key:
                        encoder_key = key.replace("bert.", "")
                        state_dict[encoder_key] = state_dict[key]
                        if not has_decoder:
                            del state_dict[key]

                    # init text decoder as multimodal encoder (last 6 layers of model.text_encoder)
                    # only for generation tasks like VQA
                    if has_decoder and "text_encoder" in key:
                        if "layer" in key:
                            encoder_keys = key.split(".")
                            layer_num = int(encoder_keys[4])
                            if layer_num < config.model.text_encoder.fusion_layer:
                                del state_dict[key]
                                continue
                            else:
                                decoder_layer_num = layer_num - config.model.text_encoder.fusion_layer
                                encoder_keys[4] = str(decoder_layer_num)
                                encoder_key = ".".join(encoder_keys)
                        else:
                            encoder_key = key
                        decoder_key = encoder_key.replace("text_encoder", "text_decoder")
                        state_dict[decoder_key] = state_dict[key]
                        del state_dict[key]

        msg = model_without_ddp.load_state_dict(state_dict, strict=False)
        logger.info(msg)
        logger.info(f"Loaded checkpoint from {config.pretrained_path}")
    else:
        logger.warning("No pretrained checkpoint provided, training from scratch")

    return (
        model,
        model_without_ddp,
        optimizer,
        scheduler,
        scaler,
        tokenizer,
        start_epoch,
        global_step,
    )
