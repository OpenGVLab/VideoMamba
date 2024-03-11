import copy
import datetime
import logging
import os
import time
import json
from os.path import join

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import wandb
import torch.nn.functional as F
from einops import rearrange

from dataset import MetaLoader, create_dataset, create_loader
from models.utils import tile
from models import *
from tasks.pretrain import setup_dataloaders
from tasks.retrieval_utils import evaluation_wrapper
from tasks.shared_utils import setup_model
from utils.basic_utils import MetricLogger, SmoothedValue, setup_seed, flat_list_of_lists, save_json
from utils.config import Config
from utils.config_utils import setup_main
from utils.distributed import get_rank, is_main_process
from utils.logger import log_dict_to_wandb, setup_wandb

logger = logging.getLogger(__name__)


def get_sim_for_each_question(model, pooled_image_feat, pooled_text_feat):
    """TODO: Docstring for get_sim_for_each_question.

    Args:
        model (TODO): TODO
        pooled_image_feat (torch.Tensor): Shape: [b,t, c]
        pooled_text_feat (torch.Tensor): Shape: [b, n, c]. n is the number of answer candidates.

    Returns: TODO

    """
    image_proj = model.vision_proj
    text_proj = model.text_proj

    image_feat = F.normalize(image_proj(pooled_image_feat), dim=-1)
    text_feat = F.normalize(text_proj(pooled_text_feat), dim=-1)
    sim = torch.matmul(image_feat, rearrange(text_feat, "b n c -> b c n"))  # [b, t, n]
    sim = sim.mean(1) / model.temp  # [b,n]
    sim = F.softmax(sim, dim=1)  # [b, n]
    return sim


def main_with_ensemble(config, test_loader, model_without_ddp, tokenizer, dtype=data_type):
    logger.info(f"train_file: {config.train_file}")

    setup_seed(config.seed + get_rank())
    device = torch.device(config.device)
    cudnn.benchmark = True

    config.scheduler.num_training_steps = 10
    config.scheduler.num_warmup_steps = 10
    model = model_without_ddp
    model.eval()

    logger.info("Start " + "evaluation" if config.evaluate else "training")
    metric_logger = MetricLogger(delimiter="  ")
    iterator = metric_logger.log_every(test_loader, 5, "Evaluation: ")
    num_options_per_q = 5
    all_gt_answers = []
    all_pred_answers = []
    predictions = []
    with torch.cuda.amp.autocast(enabled=config.fp16, data_type=data_type), torch.no_grad():
        for image, text, ans, ann in iterator:
            image = image.to(device, non_blocking=True)  # bsz
            all_gt_answers.append(ans)
            text = flat_list_of_lists(list(zip(*text)))  # List(str), len=bsz*5
            text_input = tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=config.max_txt_l,
                return_tensors="pt",
            ).to(
                device
            )  # bsz*5, ?

            # encode text
            # [b*5, l, c], [b*5, c]
            text_feat, pooled_text_feat = model.encode_text(text_input)
            # encode image
            if config.evaluation.eval_frame_ensemble == "concat":  # default
                image_feats, pooled_image_feat = model.encode_vision(image, test=True)
                if len(image_feats.shape) == 4:
                    image_feats = rearrange(image_feats, "b t l c -> b (t l) c")
                # (bsz, #frm*L, d), (bsz, #frm, d)
                image_feats = image_feats.unsqueeze(1)  # (bsz, 1, #frm*L, d)
                pooled_image_feat = pooled_image_feat.unsqueeze(1)  # (bsz, 1, #frm, d)
            else:
                assert config.video_input.num_frames == 1, "only support single-frame"
                assert config.evaluation.eval_frame_ensemble in ["mean", "max", "lse"]
                image_feats, pooled_image_feat = model.encode_vision(
                    image
                )  # (bsz, #frm, L, d), (bsz, #frm, d)
            # generate score for each clip, and aggregate all clip scores for a video
            n_clip_per_video = image_feats.shape[1]
            clip_scores = []
            for clip_idx in range(n_clip_per_video):
                image_feat = image_feats[:, clip_idx]
                pooled_image_feat = pooled_image_feat[:, clip_idx]
                image_feat = tile(image_feat, 0, num_options_per_q)
                image_mask = torch.ones(image_feat.size()[:-1], dtype=torch.long).to(
                    device, non_blocking=True
                )

                # contrastive score
                pooled_text_feat = rearrange(
                    pooled_text_feat, "(b n) c -> b n c", n=num_options_per_q
                )
                sim = get_sim_for_each_question(
                    model, pooled_image_feat, pooled_text_feat
                )  # [b, n]
                sim = sim.flatten()  # [b*n,]

                # cross-modal encode
                output = model.get_text_encoder()(
                    encoder_embeds=text_feat,
                    attention_mask=text_input.attention_mask,
                    encoder_hidden_states=image_feat,
                    encoder_attention_mask=image_mask,
                    return_dict=True,
                    mode="fusion",
                )
                itm_embeds = output.last_hidden_state[:, 0]  # [CLS]

                score = F.softmax(model.itm_head(itm_embeds), dim=1)[:, 1]  # [bs*5]
                score = score * 0.7 + sim * 0.3

                clip_scores.append(score)

            if len(clip_scores) == 1:
                score = clip_scores[0]
            else:
                assert config.evaluation.eval_frame_ensemble in ["mean", "max", "lse"]
                clip_scores = torch.stack(clip_scores)  # (#clips, k)
                if config.evaluation.eval_frame_ensemble == "mean":
                    score = clip_scores.mean(0)
                elif config.evaluation.eval_frame_ensemble == "max":
                    score = clip_scores.max(0)[0]
                elif config.evaluation.eval_frame_ensemble == "lse":  # LogSumExp
                    score = torch.logsumexp(clip_scores, dim=0)
                else:
                    raise ValueError(
                        "config.evaluation.eval_frame_ensemble must in [mean, max, lse] when #clip > 1."
                    )

            pred_ans = score.view(-1, num_options_per_q).max(1)[1].cpu()
            all_pred_answers.append(pred_ans)

            # assemble predictions
            ensemble_scores = score.view(-1, num_options_per_q).cpu()  # (bsz, 5)
            if n_clip_per_video > 1:
                clip_scores = clip_scores.view(
                    n_clip_per_video, -1, num_options_per_q
                ).cpu()  # (#clips, bsz, 5)
            for q_idx in range(len(ensemble_scores)):  # bsz
                _pred = dict(
                    video=ann["video"][q_idx],
                    options=[e[q_idx] for e in ann["caption"]],
                    answer=ann["answer"][q_idx].item(),
                    pred_ans_ensemble=pred_ans[q_idx].item(),
                    pred_scores_ensemble=ensemble_scores[q_idx].numpy(),  # (5, )
                )
                # clip scores
                if n_clip_per_video > 1:
                    _pred["pred_scores_frame"] = clip_scores[:, q_idx].numpy()  # (#clips, 5)
                    _pred["pred_ans_frame"] = (
                        clip_scores[:, q_idx].max(1)[1].numpy()
                    )  # (#clips, )
                predictions.append(_pred)

    all_gt_answers = torch.cat(all_gt_answers, 0)
    all_pred_answers = torch.cat(all_pred_answers, 0)
    acc = all_gt_answers == all_pred_answers
    acc = float(torch.sum(acc) / len(acc))
    eval_res = {"acc": round(100 * acc, 2)}
    logger.info(f"\n{eval_res}")
    save_json(eval_res, join(config.output_dir, "eval_res.json"))
    torch.save(predictions, join(config.output_dir, "prediction_scores.pth"))
    return eval_res


def train(
    model,
    train_loaders,
    optimizer,
    tokenizer,
    epoch,
    global_step,
    device,
    scheduler,
    scaler,
    config,
    data_type
):
    model.train()

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window=1, fmt="{value:.6f}"))
    metric_logger.add_meter("temperature", SmoothedValue(window=1, fmt="{value:.4f}"))
    loss_names = ["loss_" + k for k, v in config.criterion.loss_weight.items() if v != 0]

    media_types = [loader.dataset.media_type for loader in train_loaders]
    for name in loss_names:
        for m in media_types:
            metric_logger.add_meter(f"{m}-{name}", SmoothedValue(window=1, fmt="{value:.4f}"))

    header = f"Train Epoch: [{epoch}]"
    log_freq = config.log_freq

    if config.distributed:
        for d in train_loaders:
            d.sampler.set_epoch(epoch)
    train_loader = MetaLoader(name2loader=dict(list(zip(media_types, train_loaders))))

    model_without_ddp = model.module if config.distributed else model
    iterator = metric_logger.log_every(train_loader, log_freq, header)
    for i, (media_type, (image, text, idx)) in enumerate(iterator):
        image = image.to(device, non_blocking=True)
        idx = idx.to(device, non_blocking=True)
        text_input = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=config.max_txt_l,
            return_tensors="pt",
        ).to(device)

        with torch.cuda.amp.autocast(enabled=config.fp16, data_type=data_type):
            loss_dict = model(image, text_input, idx=idx)
            loss = sum(loss_dict.values())

        if not config.fp16 or config.get('bf16', True):
            optimizer.zero_grad()
            loss.backward()
            if config.optimizer.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimizer.max_grad_norm)
            optimizer.step()
            scheduler.step()
        else:
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            if config.optimizer.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimizer.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        # logging
        for name in loss_names:
            value = loss_dict[name]
            value = value if isinstance(value, float) else value.item()
            metric_logger.update(**{f"{media_type}-{name}": value})
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(temperature=model_without_ddp.temp.item())

        if is_main_process() and config.wandb.enable and global_step % log_freq == 0:
            logs = metric_logger.get_global_avg_dict()
            log_dict_to_wandb(logs, step=global_step, prefix="train/")

        global_step += 1

        if config.debug and (i + 1) % 5 == 0:
            break

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged train stats: {metric_logger.global_avg()}")
    return global_step


def main(config):
    if is_main_process() and config.wandb.enable:
        run = setup_wandb(config)

    logger.info(f"config: \n{config}")
    logger.info(f"train_file: {config.train_file}")

    setup_seed(config.seed + get_rank())
    device = torch.device(config.device)
    cudnn.benchmark = True

    train_loaders, test_name2loaders, train_media_types = setup_dataloaders(config, mode="ret")
    num_steps_per_epoch = sum(len(d) for d in train_loaders)
    config.scheduler.num_training_steps = num_steps_per_epoch * config.scheduler.epochs
    config.scheduler.num_warmup_steps = num_steps_per_epoch * config.scheduler.warmup_epochs

    model_cls = eval(config.model.get('model_cls', 'UMT'))
    (
        model,
        model_without_ddp,
        optimizer,
        scheduler,
        scaler,
        tokenizer,
        start_epoch,
        global_step,
    ) = setup_model(
        config,
        model_cls=model_cls,
        has_decoder=False,
        pretrain=False,
        # find_unused_parameters=True,
        find_unused_parameters=False,
    )
    if is_main_process() and config.wandb.enable:
        wandb.watch(model)

    
    # create test dataloader
    test_dataset = create_dataset("mc_test", config)
    test_loader = create_loader(
        [test_dataset],
        [None],
        batch_size=[config.inputs.batch_size_test.video],
        num_workers=[config.num_workers],
        is_trains=[False],
        collate_fns=[None],
    )[0]
    res = main_with_ensemble(config, test_loader, model_without_ddp, tokenizer)
    if is_main_process():
        if config.wandb.enable:
            log_dict_to_wandb(res, step=0, prefix=config.test_types)

    best = 0
    best_epoch = 0

    if config.get('bf16', True):
        data_type = torch.bfloat16
    else:
        data_type = torch.float16

    logger.info("Start " + "evaluation" if config.evaluate else "training")
    start_time = time.time()
    for epoch in range(start_epoch, config.scheduler.epochs):
        if not config.evaluate:
            global_step = train(
                model,
                train_loaders,
                optimizer,
                tokenizer,
                epoch,
                global_step,
                device,
                scheduler,
                scaler,
                config,
                data_type
            )
        
        if is_main_process():
            save_obj = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "config": config,
                "epoch": epoch,
                "global_step": global_step,
            }
            if config.get("save_latest", False):
                torch.save(save_obj, join(config.output_dir, "ckpt_latest.pth"))
            else:
                torch.save(save_obj, join(config.output_dir, f"ckpt_{epoch:02d}.pth"))

        with torch.cuda.amp.autocast(enabled=config.fp16, dtype=data_type):
            res = main_with_ensemble(config, test_loader, model_without_ddp, tokenizer, dtype=data_type)
            eval_res = res

        if is_main_process():
            if config.wandb.enable:
                log_dict_to_wandb(eval_res, step=global_step, prefix=config.test_types)

            acc = eval_res["acc"]   
            logger.info(f"Epoch {epoch}")
            logger.info(f"\n{eval_res}")

            save_json(eval_res, join(config.output_dir, "eval_res_latest.json"))

            if not config.evaluate and acc > best:
                eval_file = "eval_res_best.json"
                save_json(eval_res, join(config.output_dir, eval_file))
                torch.save(save_obj, join(config.output_dir, "ckpt_best.pth"))
                best = acc
                best_epoch = epoch
            if config.evaluate:
                save_json(eval_res, join(config.output_dir, eval_file))

        if config.evaluate or config.debug:
            break

        dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"Training time {total_time_str}")
    logger.info(f"best epoch {best_epoch}")
    logger.info(f"best {best}")
    logger.info(f"Checkpoints and Logs saved at {config.output_dir}")

    if is_main_process() and config.wandb.enable:
        run.finish()


def eval_after_training(train_config):
    # general config for all
    train_config.wandb.enable = False
    train_config.evaluate = True
    train_config.pretrained_path = join(train_config.output_dir, "ckpt_best.pth")
    train_config.num_frames_test = train_config.num_frames
    train_config.inputs.video_input.num_frames_test = train_config.num_frames

    if train_config.get('num_frames_test_final', False):
        train_config.num_frames_test = train_config.num_frames_test_final
        train_config.batch_size = train_config.batch_size_final
        train_config.inputs.video_input.num_frames_test = train_config.num_frames_test_final
        train_config.model.vision_encoder.num_frames = train_config.num_frames_test_final

    eval_config = copy.deepcopy(train_config)
    eval_config.test_types = list(eval_config.test_file.keys())
    eval_config.output_dir = join(eval_config.output_dir, f"eval_after_training")
    eval_config.result_dir = eval_config.output_dir
    if is_main_process():
        os.makedirs(eval_config.output_dir, exist_ok=True)
        Config.dump(eval_config, os.path.join(eval_config.output_dir, "config.json"))
    logger.info(f"===========> START eval_after_training [{eval_config.test_types}]")
    main(eval_config)


if __name__ == "__main__":
    cfg = setup_main()
    main(cfg)
    # if not cfg.evaluate:
    #     eval_after_training(cfg)
