export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1

JOB_NAME='videomamba_middle_mask_pt_f8_res224'
OUTPUT_DIR="$(dirname $0)/$JOB_NAME"
LOG_DIR="./logs/${JOB_NAME}"
PREFIX='your_ssv2_path'
DATA_PATH='your_ssv2_metadata_path/train.csv'

PARTITION='video5'
GPUS=32
GPUS_PER_NODE=8
CPUS_PER_TASK=16

srun -p $PARTITION \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    python -u run_videomamba_pretraining.py \
    --data_path ${DATA_PATH} \
    --prefix ${PREFIX} \
    --num_sample 1 \
    --split ',' \
    --flip True \
    --mask_type 'attention' \
    --mask_ratio 0.8 \
    --model 'videomamba_middle_pretrain' \
    --clip_teacher 'clip_b16' \
    --clip_loss_ratio 1 \
    --clip_loss_type 'l2' \
    --clip_decoder_embed_dim 576 \
    --clip_output_dim 512 \
    --clip_norm_type 'l2' \
    --clip_return_layer 1 \
    --clip_return_interval 1 \
    --clip_student_return_interval 1 \
    --clip_return_cls \
    --clip_return_attn \
    --tubelet_size 1 \
    --lr 1.5e-4 \
    --drop_path 0.4 \
    --batch_size 64 \
    --num_segments 8 \
    --num_frames 8 \
    --sampling_rate 1 \
    --num_workers 12 \
    --opt adamw \
    --opt_betas 0.9 0.95 \
    --warmup_epochs 40 \
    --save_ckpt_freq 1000 \
    --epochs 801 \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --bf16
