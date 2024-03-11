# Multi-modality Video Understanding

## Datasets

You can find the dataset instructions in [DATASET](DATASET.md). We have provide all the metadata files of our data.

## Model ZOO

You can find all the models and the scripts in [MODEL_ZOO](./MODEL_ZOO.md).

## Pre-Training

We use [CLIP](https://github.com/openai/CLIP) pretrained models as the unmasked teachers by default:
- Follow [extract.ipynb](../video_sm/models/extract_clip/extract.ipynb) to extract visual encoder from CLIP.
- Change `MODEL_PATH` in [clip.py](./models/backbones/videomamba/clip.py).

For training, you can simply run the pretraining scripts as follows:
```shell
# masked pretraining
bash ./exp_pt/videomamba_middle_5m/run.sh
# further unmasked pretraining for 1 epoch
bash ./exp_pt/videomamba_middle_5m_unmasked/run.sh
```

>  **Notes:**
> 1. Set `data_dir` and `your_data_path` like `your_webvid_path` in [data.py](./configs/data.py) before running the scripts.
> 2. Set `vision_encoder.pretrained` in `vision_encoder.pretrained` in the corresponding config files.
> 3. Set `--rdzv_endpoint` to your `MASTER_NODE:MASTER_PORT` in [torchrun.sh](torchrun.sh).
> 4. `save_latest=True` will automatically save the latest checkpoint while training.
> 5. `auto_resume=True` will automatically loaded the best or latest checkpoint while training.
> 6. For unmasked pretraining, please set `pretrained_path` to load the masked pretrained epoch.


## Zero-shot Evaluation

For zero-shot evaluation, you can simply run the pretraining scripts as follows:
```shell
bash ./exp_zs/msrvtt/run.sh
```

>  **Notes:**
> 1. Set `pretrained_path` in the running scripts before running the scripts.
> 2. Set `zero_shot=True` and `evaluate=True` for zero-shot evaluation 

