export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1

JOB_NAME='videomamba_tiny_f64_res224'
OUTPUT_DIR="$(dirname $0)/$JOB_NAME"
LOG_DIR="./logs/${JOB_NAME}"
PREFIX='your_coin_path'
DATA_PATH='your_coin_metadata_path'

PARTITION='video5'
GPUS=16
GPUS_PER_NODE=8
CPUS_PER_TASK=16

srun -p $PARTITION \
        --job-name=${JOB_NAME} \
        --gres=gpu:${GPUS_PER_NODE} \
        --ntasks=${GPUS} \
        --ntasks-per-node=${GPUS_PER_NODE} \
        --cpus-per-task=${CPUS_PER_TASK} \
        --kill-on-bad-exit=1 \
        python run_class_finetuning.py \
        --model videomamba_tiny \
        --finetune your_model_path/videomamba_t16_k400_f32_res224.pth \
        --data_path ${DATA_PATH} \
        --prefix ${PREFIX} \
        --data_set 'Kinetics_sparse' \
        --split ',' \
        --nb_classes 180 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 16 \
        --num_sample 2 \
        --input_size 224 \
        --short_side_size 224 \
        --save_ckpt_freq 100 \
        --num_frames 64 \
        --orig_t_size 64 \
        --num_workers 12 \
        --warmup_epochs 5 \
        --tubelet_size 1 \
        --epochs 40 \
        --lr 2e-4 \
        --drop_path 0.1 \
        --aa rand-m5-n2-mstd0.25-inc1 \
        --opt adamw \
        --opt_betas 0.9 0.999 \
        --weight_decay 0.1 \
        --test_num_segment 10 \
        --test_num_crop 3 \
        --dist_eval \
        --test_best \
        --bf16
