export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1
echo "PYTHONPATH: ${PYTHONPATH}"
which_python=$(which python)
echo "which python: ${which_python}"
export PYTHONPATH=${PYTHONPATH}:${which_python}
export PYTHONPATH=${PYTHONPATH}:.
echo "PYTHONPATH: ${PYTHONPATH}"

JOB_NAME='videomamba_middle_res224to448to576'
OUTPUT_DIR="$(dirname $0)"

python run_with_submitit.py \
    --root_dir_train your_imagenet_path/train/ \
    --meta_file_train your_imagenet_path/meta/train.txt \
    --root_dir_val your_imagenet_path/val/ \
    --meta_file_val your_imagenet_path/meta/val.txt \
    --model videomamba_middle \
    --finetune your_model_path/videomamba_middle_res224to448.pth \
    --input-size 576 \
    --batch-size 32 \
    --num_workers 16 \
    --lr 5e-6 \
    --min-lr 5e-6 \
    --weight-decay 1e-8 \
    --warmup-epochs 2 \
    --epochs 10 \
    --drop-path 0.5 \
    --no-model-ema \
    --output_dir ${OUTPUT_DIR}/ckpt \
    --bf16 \
    --dist-eval