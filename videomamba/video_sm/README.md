# Single-modality Video Understanding

We currenent release the code and models for:

- [x] **Masked Pretraining**

- [x] **Short-term Video Understaning**
    - K400 and SthSthV2

- [x] **Long-term Video Understaning**
    - Breakfast, COIN and LVU




## Update

- :fire: **03/12/2024**: Pretrained models on ImageNet-1K are released.


## Datasets

You can find the dataset instructions in [DATASET](./DATASET.md).

## Model ZOO

You can find all the models and the scripts in [MODEL_ZOO](./MODEL_ZOO.md).

## Usage

### Masked Pretraining

We use [CLIP](https://github.com/openai/CLIP) pretrained models as the unmasked teachers by default:
- Follow [extract.ipynb](./models/extract_clip/extract.ipynb) to extract visual encoder from CLIP.
- Change `MODEL_PATH` in [clip.py](./models/clip.py).

For training, you can simply run the pretraining scripts as follows:
```shell
bash ./exp/k400/videomamba_middle_mask/run_mask_pretrain.sh
```

> **Notes:**
> 1. Chage `DATA_PATH` to your data path before running the scripts.
> 2. `--sampling_rate` is set to 1 for **sprase sampling**.
> 3. The latest checkpoint will be automatically saved while training, thus we use a large `--save_ckpt_freq`.
> 4. For VideoMamba-M, we use CLIP-B-ViT as the teacher.


### Short-term Video Understanding

For finetuning, you can simply run the fine-tuning scripts as follows:
```shell
bash ./exp/k400/videomamba_middle_mask/run_f8x224.sh
```

> **Notes:**
> 1. Chage `DATA_PATH` And `PREFIX` to your data path before running the scripts.
> 2. Set `--finetune` when using masked pretrained model.
> 3. The best checkpoint will be automatically evaluated with `--test_best`.
> 4. Set `--test_num_segment` and `--test_num_crop` for different evaluation strategies.
> 5. To only run evaluation, just set `--eval`.


### Long-term Video Understanding

For BreakFast and COIN, you can simply run the fine-tuning scripts as follows:
```shell
bash ./exp/breakfast/videomamba_middle_mask/run_f32x224.sh
```

For LVU, there are classification and regression tasks, you can simply run the fine-tuning scripts as follows:
```shell
# classification
bash ./exp/lvu/run_class.sh
# regression
bash ./exp/lvu/run_regression.sh
```
> **Notes:**
> For regression tasks, the data should be preprocessed with normalization as in [ViS4mer](https://github.com/md-mohaiminul/ViS4mer/blob/main/datasets/lvu_dataset.py).


### :warning: Using Trimmed Video

By default, we use `Kinetics_sparse` dataset for different datasets. **However, in [ViS4mer](https://github.com/md-mohaiminul/ViS4mer/blob/main/datasets/lvu_dataset.py), the authors use trimmed clips with sliding window, which may improve the results.** We also provided a dataset with sliding window as follows:
```shell
# classification
bash ./exp/lvu/run_class_trim.sh
# regression
bash ./exp/lvu/run_regression_trim.sh
```

> **Notes:**
> 1. Set `trimmed` for the length of trimmed videos.
> 2. Set `time_stride` for the length of sliding window.