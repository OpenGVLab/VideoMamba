from configs.data import *
from configs.model import *

# ========================= data ==========================
del available_corpus

train_file = [
    f"{anno_root_downstream}/lsmdc_ret_train.json",
    "your_lsmdc_path/LSMDC",
    "video",
]
test_file = dict(
    test=[
        f"{anno_root_downstream}/lsmdc_ret_test_1000.json",
        "your_lsmdc_path/LSMDC",
        "video",
    ],
)
test_types = ["test"]
num_workers = 6

stop_key = "test/" # used to choose the best ckpt. If None, save the last.
is_paragraph_retrieval = False

# ========================= input ==========================
num_frames = 8
num_frames_test = 8
batch_size = 64
max_txt_l = 96 

inputs = dict(
    image_res=224,
    video_input=dict(
        num_frames="${num_frames}",
        sample_type="rand",
        num_frames_test="${num_frames_test}",
        sample_type_test="middle",
        random_aug=False,
    ),
    max_txt_l=dict(image="${max_txt_l}", video="${max_txt_l}"),
    batch_size=dict(image="${batch_size}", video="${batch_size}"),
    batch_size_test=dict(image="${batch_size}", video="${batch_size}"),
)

# ========================= model ==========================
text_enc = "bert"
model = dict(
    model_cls="UMT_VIDEOMAMBA",
    vision_encoder=dict(
        # backbone
        name="videomamba_middle",
        img_size=224, 
        patch_size=16, 
        depth=32, 
        embed_dim=576, 
        drop_path_rate=0.25,
        ssm_cfg=None, 
        norm_epsilon=1e-5, 
        fused_add_norm=True,
        rms_norm=True, 
        residual_in_fp32=True,
        bimamba=True,
        pool_type="cls+avg",
        kernel_size=1,
        num_frames="${num_frames}",
        ckpt_num_frame=8,
        use_checkpoint=False,
        checkpoint_num=0,
        clip_decoder_embed_dim=576,
        clip_output_dim=512,
        clip_norm_type='l2',
        clip_return_layer=1,
        clip_student_return_interval=1,
        pretrained="your_model_path/videomamba_m16_k400_mask_pt_f8_res224.pth",
        # clip teacher
        clip_teacher="none",
        clip_img_size=224,
        clip_return_interval=1,
        # mask
        video_mask_type="none",
        video_mask_ratio=0.,
        video_double_mask_ratio=0.,
        image_mask_type="none",
        image_mask_ratio=0.,
        image_double_mask_ratio=0.,
        # for ret
        keep_temporal=True,
    ),
    text_encoder="${TextEncoders[${text_enc}]}",
    multimodal=dict(enable=True),
    embed_dim=512,
    temp=0.07,
)

criterion = dict(
    loss_weight=dict(
        vtc=1.0, 
        mlm=1.0, 
        vtm=1.0, 
        uta=0.0,
    ),  # 0: disabled.
    vtm_hard_neg=True,
    mlm_masking_prob=0.5,
    uta_norm_type="l2",
    uta_loss_type="l2",
)

optimizer = dict(
    opt="adamW",
    lr=1e-5,
    opt_betas=[0.9, 0.999],  # default
    weight_decay=0.02,
    max_grad_norm=-1,  # requires a positive float, use -1 to disable
    # use a different lr for some modules, e.g., larger lr for new modules
    different_lr=dict(enable=False, module_names=[], lr=4e-3),
)

scheduler = dict(sched="cosine", epochs=2, min_lr_multi=0.01, warmup_epochs=0.2)

evaluate = False
deep_fusion = False
evaluation = dict(
    eval_frame_ensemble="concat",  # [concat, max, mean, lse]
    eval_x_only=False,
    k_test=128,
    eval_offload=False,  # offload gpu tensors to cpu to save memory.
)

fp16 = True
bf16 = True
gradient_checkpointing = True

# ========================= wandb ==========================
wandb = dict(
    enable=False,
    entity="likunchang",  # username or team name to store the runs, see https://docs.wandb.ai/ref/python/init
    project="umt_videomamba",  # setup in your command line
)
dist_url = "env://"
device = "cuda"
mode = "pt"

# ========================= others ==========================
output_dir = None  # output dir
resume = False  # if True, load optimizer and scheduler states as well
debug = False
log_freq = 1
seed = 42

zero_shot=True
save_latest = True
auto_resume = True
pretrained_path = ""  # path to pretrained model weights, for resume only?
