export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1
echo "PYTHONPATH: ${PYTHONPATH}"
which_python=$(which python)
echo "which python: ${which_python}"
export PYTHONPATH=${PYTHONPATH}:${which_python}
export PYTHONPATH=${PYTHONPATH}:.
echo "PYTHONPATH: ${PYTHONPATH}"

JOB_NAME='videomamba_middle_res224'
OUTPUT_DIR="$(dirname $0)"

python run_with_submitit_distill.py \
    --root_dir_train your_imagenet_path/train/ \
    --meta_file_train your_imagenet_path/meta/train.txt \
    --root_dir_val your_imagenet_path/val/ \
    --meta_file_val your_imagenet_path/meta/val.txt \
    --model videomamba_middle \
    --teacher_model videomamba_small \
    --teacher_embed_dim 384 \
    --teacher_pretrained_path your_model_path/videomamba_small_res224.pth \
    --batch-size 128 \
    --num_workers 16 \
    --warmup-epochs 20 \
    --lr 5e-4 \
    --warmup-lr 5e-7 \
    --min-lr 5e-6 \
    --weight-decay 0.05 \
    --drop-path 0.5 \
    --clip-grad 5.0 \
    --no-model-ema \
    --output_dir ${OUTPUT_DIR}/ckpt \
    --bf16 \
    --dist-eval