# Image Classification

We currenent release the code and models for:

- [x] **ImageNet-1K pretraining**

- [x] **Large resolution fine-tuning**



## Update

- :fire: **03/12/2024**: Pretrained models on ImageNet-1K are released.



## Model Zoo

See [MODEL_ZOO](./MODEL_ZOO.md).


## Usage

### Normal Training

Simply run the training scripts in [exp](exp) as followed:

```shell
bash ./exp/videomamba_tiny/run224.sh
```

> If the training was interrupted abnormally, you can simply rerun the script for auto-resuming. Sometimes the checkpoint may not be saved properly, you should set the resumed model via `--reusme ${OUTPUT_DIR}/ckpt/checkpoint.pth`.

### Training w/ SD

Simply run the training scripts in [exp_distill](exp_distill) as followed:

```shell
bash ./exp_distill/videomamba_middle/run224.sh
```

> For `teacher_model`, we use a smaller model by default.

### Large Resolution Fine-tuning

Simply run the training scripts in [exp](exp) as followed:

```shell
bash ./exp/videomamba_tiny/run448.sh
```

> Please set pretrained model via `--finetune`.

### Evaluation

Simply add `--eval` in the training scripts.

> It will evaluate the last model by default. You can set other models via `--resume`.

### Generate curves

You can generate the training curves as followed:

```shell
python3 generate_tensoboard.py
```

Note that you should install `tensorboardX`.


