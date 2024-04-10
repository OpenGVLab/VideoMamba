# Model Zoo

## Note

- :warning: <s>The current video models are **fine-tuned without layer decay** due to the [bug](https://github.com/OpenGVLab/VideoMamba/blob/9806e369196c88f3eef92380c74beeb51208d0c3/videomamba/video_sm/optim_factory.py#L24), which may help to improve the performances as in [MAE](https://github.com/facebookresearch/mae). We have fixed the bug but do not plan to retrain them.</s> We have applied it for VideoMamba-M but it does not help.
- For all the pretraining and finetuning, we adopt spaese/uniform sampling.
- `#Frame` $=$ `#input_frame` $\times$ `#crop` $\times$ `#clip`
- `#input_frame` means how many frames are input for model per inference
- `#crop` means spatial crops (e.g., 3 for left/right/center)
- `#clip` means temporal clips (e.g., 4 means repeted sampling four clips with different start indices)

## Masked Pretraining

| Model    | Setting     | Model  | Shell  |
| -------- | ----------- | ------ | ------ |
| VideoMamba-M | K400 800e  |  [aliyun](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/videomamba/videomamba_m16_k400_mask_pt_f8_res224.pth), [:hugs:HF](https://huggingface.co/OpenGVLab/VideoMamba/blob/main/videomamba_m16_k400_mask_pt_f8_res224.pth) | [run.sh](./exp/k400/videomamba_middle_mask/run_mask_pretrain.sh) |
| VideoMamba-M | SthSthV2 200e  |  [aliyun](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/videomamba/videomamba_m16_ssv2_mask_pt_f8_res224.pth), [:hugs:HF](https://huggingface.co/OpenGVLab/VideoMamba/blob/main/videomamba_m16_ssv2_mask_pt_f8_res224.pth) | [run.sh](./exp/ssv2/videomamba_middle_mask/run_mask_pretrain.sh) |


## Short-term Video Understanding

### K400

| Model    | Pretraining  | Resolution    | #Frame   | Top-1  | Model  | Shell  |
| -------- | ------------ | ------------  | -------- | ------ | ------ | ------ |
| VideoMamba-Ti | ImageNet-1K  | 224 | 8x3x4   | 76.9 | [aliyun](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/videomamba/videomamba_t16_k400_f8_res224.pth), [:hugs:HF](https://huggingface.co/OpenGVLab/VideoMamba/blob/main/videomamba_t16_k400_f8_res224.pth) | [run.sh](./exp/k400/videomamba_tiny/run_f8x224.sh) | 
| VideoMamba-Ti | ImageNet-1K  | 224 | 16x3x4   | 78.1 | [aliyun](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/videomamba/videomamba_t16_k400_f16_res224.pth), [:hugs:HF](https://huggingface.co/OpenGVLab/VideoMamba/blob/main/videomamba_t16_k400_f16_res224.pth) | [run.sh](./exp/k400/videomamba_tiny/run_f16x224.sh) | 
| VideoMamba-Ti | ImageNet-1K  | 224 | 32x3x4   | 78.8 | [aliyun](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/videomamba/videomamba_t16_k400_f32_res224.pth), [:hugs:HF](https://huggingface.co/OpenGVLab/VideoMamba/blob/main/videomamba_t16_k400_f32_res224.pth) | [run.sh](./exp/k400/videomamba_tiny/run_f32x224.sh) | 
| VideoMamba-Ti | ImageNet-1K  | 224 | 64x3x4   | 79.6 | [aliyun](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/videomamba/videomamba_t16_k400_f64_res224.pth), [:hugs:HF](https://huggingface.co/OpenGVLab/VideoMamba/blob/main/videomamba_t16_k400_f64_res224.pth) | [run.sh](./exp/k400/videomamba_tiny/run_f64x224.sh) | 
| VideoMamba-Ti | ImageNet-1K  | 384 | 64x3x4   | 80.3 | [aliyun](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/videomamba/videomamba_t16_k400_f64_res224to384.pth), [:hugs:HF](https://huggingface.co/OpenGVLab/VideoMamba/blob/main/videomamba_t16_k400_f64_res224to384.pth) | [run.sh](./exp/k400/videomamba_tiny/run_f64x224to384.sh) | 
| VideoMamba-S | ImageNet-1K  | 224 | 8x3x4   | 79.3 | [aliyun](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/videomamba/videomamba_s16_k400_f8_res224.pth), [:hugs:HF](https://huggingface.co/OpenGVLab/VideoMamba/blob/main/videomamba_s16_k400_f8_res224.pth) | [run.sh](./exp/k400/videomamba_small/run_f8x224.sh) | 
| VideoMamba-S | ImageNet-1K  | 224 | 16x3x4   | 80.8 | [aliyun](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/videomamba/videomamba_s16_k400_f16_res224.pth), [:hugs:HF](https://huggingface.co/OpenGVLab/VideoMamba/blob/main/videomamba_s16_k400_f16_res224.pth) | [run.sh](./exp/k400/videomamba_small/run_f16x224.sh) | 
| VideoMamba-S | ImageNet-1K  | 224 | 32x3x4   | 81.5 | [aliyun](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/videomamba/videomamba_s16_k400_f32_res224.pth), [:hugs:HF](https://huggingface.co/OpenGVLab/VideoMamba/blob/main/videomamba_s16_k400_f32_res224.pth) | [run.sh](./exp/k400/videomamba_small/run_f32x224.sh) | 
| VideoMamba-S | ImageNet-1K  | 224 | 64x3x4   | 81.8 | [aliyun](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/videomamba/videomamba_s16_k400_f64_res224.pth), [:hugs:HF](https://huggingface.co/OpenGVLab/VideoMamba/blob/main/videomamba_s16_k400_f64_res224.pth) | [run.sh](./exp/k400/videomamba_small/run_f64x224.sh) | 
| VideoMamba-S | ImageNet-1K  | 384 | 64x3x4   | 82.7 | [aliyun](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/videomamba/videomamba_s16_k400_f64_res224to384.pth), [:hugs:HF](https://huggingface.co/OpenGVLab/VideoMamba/blob/main/videomamba_s16_k400_f64_res224to384.pth) | [run.sh](./exp/k400/videomamba_small/run_f64x224to384.sh) | 
| VideoMamba-M | ImageNet-1K  | 224 | 8x3x4   | 80.6 | [aliyun](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/videomamba/videomamba_m16_k400_f8_res224.pth), [:hugs:HF](https://huggingface.co/OpenGVLab/VideoMamba/blob/main/videomamba_m16_k400_f8_res224.pth) | [run.sh](./exp/k400/videomamba_middle/run_f8x224.sh) | 
| VideoMamba-M | ImageNet-1K  | 224 | 16x3x4   | 81.9 | [aliyun](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/videomamba/videomamba_m16_k400_f16_res224.pth), [:hugs:HF](https://huggingface.co/OpenGVLab/VideoMamba/blob/main/videomamba_m16_k400_f16_res224.pth) | [run.sh](./exp/k400/videomamba_middle/run_f16x224.sh) | 
| VideoMamba-M | ImageNet-1K  | 224 | 32x3x4   | 82.4 | [aliyun](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/videomamba/videomamba_m16_k400_f32_res224.pth), [:hugs:HF](https://huggingface.co/OpenGVLab/VideoMamba/blob/main/videomamba_m16_k400_f32_res224.pth) | [run.sh](./exp/k400/videomamba_middle/run_f32x224.sh) | 
| VideoMamba-M | ImageNet-1K  | 224 | 64x3x4   | 82.8 | [aliyun](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/videomamba/videomamba_m16_k400_f64_res224.pth), [:hugs:HF](https://huggingface.co/OpenGVLab/VideoMamba/blob/main/videomamba_m16_k400_f64_res224.pth) | [run.sh](./exp/k400/videomamba_middle/run_f64x224.sh) | 
| VideoMamba-M | ImageNet-1K  | 384 | 64x3x4   | 83.3 | [aliyun](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/videomamba/videomamba_m16_k400_f64_res224to384.pth), [:hugs:HF](https://huggingface.co/OpenGVLab/VideoMamba/blob/main/videomamba_m16_k400_f64_res224to384.pth) | [run.sh](./exp/k400/videomamba_middle/run_f64x224to384.sh) | 
| VideoMamba-M | *MASK*  | 224 | 8x3x4   | 82.0 | [aliyun](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/videomamba/videomamba_m16_k400_mask_ft_f8_res224.pth), [:hugs:HF](https://huggingface.co/OpenGVLab/VideoMamba/blob/main/videomamba_m16_k400_mask_ft_f8_res224.pth) | [run.sh](./exp/k400/videomamba_middle_mask/run_f8x224.sh) | 
| VideoMamba-M | *MASK*  | 224 | 16x3x4   | 83.4 | [aliyun](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/videomamba/videomamba_m16_k400_f16_res224.pth), [:hugs:HF](https://huggingface.co/OpenGVLab/VideoMamba/blob/main/videomamba_m16_k400_f16_res224.pth) | [run.sh](./exp/k400/videomamba_middle_mask/run_f16x224.sh) | 
| VideoMamba-M | *MASK*  | 224 | 32x3x4   | 83.9 | [aliyun](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/videomamba/videomamba_m16_k400_mask_ft_f32_res224.pth), [:hugs:HF](https://huggingface.co/OpenGVLab/VideoMamba/blob/main/videomamba_m16_k400_mask_ft_f32_res224.pth) | [run.sh](./exp/k400/videomamba_middle_mask/run_f32x224.sh) | 
| VideoMamba-M | *MASK*  | 224 | 64x3x4   | 84.3 | [aliyun](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/videomamba/videomamba_m16_k400_mask_ft_f64_res224.pth), [:hugs:HF](https://huggingface.co/OpenGVLab/VideoMamba/blob/main/videomamba_m16_k400_mask_ft_f64_res224.pth) | [run.sh](./exp/k400/videomamba_middle_mask/run_f64x224.sh) | 
| VideoMamba-M | *MASK*  | 384 | 64x3x4   | 85.0 | [aliyun](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/videomamba/videomamba_m16_k400_mask_ft_f64_res224to384.pth), [:hugs:HF](https://huggingface.co/OpenGVLab/VideoMamba/blob/main/videomamba_m16_k400_mask_ft_f64_res224to384.pth) | [run.sh](./exp/k400/videomamba_middle_mask/run_f64x224to384.sh) | 



### SthSthV2

| Model    | Pretraining  | Resolution    | #Frame   | Top-1  | Model  | Shell  |
| -------- | ------------ | ------------  | -------- | ------ | ------ | ------ |
| VideoMamba-Ti | ImageNet-1K  | 224 | 8x3x4   | 65.1 | [aliyun](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/videomamba/videomamba_t16_ssv2_f8_res224.pth), [:hugs:HF](https://huggingface.co/OpenGVLab/VideoMamba/blob/main/videomamba_t16_ssv2_f8_res224.pth) | [run.sh](./exp/ssv2/videomamba_tiny/run_f8x224.sh) | 
| VideoMamba-Ti | ImageNet-1K  | 224 | 16x3x4   | 66.0 | [aliyun](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/videomamba/videomamba_t16_ssv2_f16_res224.pth), [:hugs:HF](https://huggingface.co/OpenGVLab/VideoMamba/blob/main/videomamba_t16_ssv2_f16_res224.pth) | [run.sh](./exp/ssv2/videomamba_tiny/run_f16x224.sh) | 
| VideoMamba-Ti | ImageNet-1K  | 288 | 16x3x4   | 66.2 | [aliyun](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/videomamba/videomamba_t16_ssv2_f16_res224to288.pth), [:hugs:HF](https://huggingface.co/OpenGVLab/VideoMamba/blob/main/videomamba_t16_ssv2_f16_res224to288.pth) | [run.sh](./exp/ssv2/videomamba_tiny/run_f16x224to288.sh) | 
| VideoMamba-S | ImageNet-1K  | 224 | 8x3x4   | 66.6 | [aliyun](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/videomamba/videomamba_s16_ssv2_f8_res224.pth), [:hugs:HF](https://huggingface.co/OpenGVLab/VideoMamba/blob/main/videomamba_s16_ssv2_f8_res224.pth) | [run.sh](./exp/ssv2/videomamba_small/run_f8x224.sh) | 
| VideoMamba-S | ImageNet-1K  | 224 | 16x3x4   | 67.7 | [aliyun](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/videomamba/videomamba_s16_ssv2_f16_res224.pth), [:hugs:HF](https://huggingface.co/OpenGVLab/VideoMamba/blob/main/videomamba_s16_ssv2_f16_res224.pth) | [run.sh](./exp/ssv2/videomamba_small/run_f16x224.sh) | 
| VideoMamba-S | ImageNet-1K  | 288 | 16x3x4   | 68.1 | [aliyun](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/videomamba/videomamba_s16_ssv2_f16_res224to288.pth), [:hugs:HF](https://huggingface.co/OpenGVLab/VideoMamba/blob/main/videomamba_s16_ssv2_f16_res224to288.pth) | [run.sh](./exp/ssv2/videomamba_small/run_f16x224to288.sh) | 
| VideoMamba-M | ImageNet-1K  | 224 | 8x3x4   | 67.3 | [aliyun](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/videomamba/videomamba_m16_ssv2_f8_res224.pth), [:hugs:HF](https://huggingface.co/OpenGVLab/VideoMamba/blob/main/videomamba_m16_ssv2_f8_res224.pth) | [run.sh](./exp/ssv2/videomamba_middle/run_f8x224.sh) | 
| VideoMamba-M | ImageNet-1K  | 224 | 16x3x4   | 68.3 | [aliyun](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/videomamba/videomamba_m16_ssv2_f16_res224.pth), [:hugs:HF](https://huggingface.co/OpenGVLab/VideoMamba/blob/main/videomamba_m16_ssv2_f16_res224.pth) | [run.sh](./exp/ssv2/videomamba_middle/run_f16x224.sh) | 
| VideoMamba-M | ImageNet-1K  | 288 | 16x3x4   | 68.4 | [aliyun](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/videomamba/videomamba_m16_ssv2_f16_res224to288.pth), [:hugs:HF](https://huggingface.co/OpenGVLab/VideoMamba/blob/main/videomamba_m16_ssv2_f16_res224to288.pth) | [run.sh](./exp/ssv2/videomamba_middle/run_f16x224to288.sh) | 
| VideoMamba-M | *MASK*  | 224 | 8x3x4   | 70.2 | [aliyun](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/videomamba/videomamba_m16_ssv2_mask_ft_f8_res224.pth), [:hugs:HF](https://huggingface.co/OpenGVLab/VideoMamba/blob/main/videomamba_m16_ssv2_mask_ft_f8_res224.pth) | [run.sh](./exp/ssv2/videomamba_middle_mask/run_f8x224.sh) | 
| VideoMamba-M | *MASK*  | 224 | 16x3x4   | 71.0 | [aliyun](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/videomamba/videomamba_m16_ssv2_f16_res224.pth), [:hugs:HF](https://huggingface.co/OpenGVLab/VideoMamba/blob/main/videomamba_m16_ssv2_f16_res224.pth) | [run.sh](./exp/ssv2/videomamba_middle_mask/run_f16x224.sh) | 
| VideoMamba-M | *MASK*  | 288 | 16x3x4   | 71.4 | [aliyun](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/videomamba/videomamba_m16_ssv2_mask_ft_f16_res224to288.pth), [:hugs:HF](https://huggingface.co/OpenGVLab/VideoMamba/blob/main/videomamba_m16_ssv2_mask_ft_f16_res224to288.pth) | [run.sh](./exp/ssv2/videomamba_middle_mask/run_f16x224to288.sh) | 


## Long-term Video Understanding

### Breakfast

| Model    | Pretraining  | Resolution    | #Frame   | Top-1  | Model  | Shell  |
| -------- | ------------ | ------------  | -------- | ------ | ------ | ------ |
| VideoMamba-Ti | K400  | 224 | 32x3x4   | 94.3 | [aliyun](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/videomamba/videomamba_t16_breakfast_f32_res224.pth), [:hugs:HF](https://huggingface.co/OpenGVLab/VideoMamba/blob/main/videomamba_t16_breakfast_f32_res224.pth) | [run.sh](./exp/breakfast/videomamba_tiny/run_f32x224.sh) | 
| VideoMamba-Ti | K400  | 224 | 64x3x4   | 94.3 | [aliyun](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/videomamba/videomamba_t16_breakfast_f64_res224.pth), [:hugs:HF](https://huggingface.co/OpenGVLab/VideoMamba/blob/main/videomamba_t16_breakfast_f64_res224.pth) | [run.sh](./exp/breakfast/videomamba_tiny/run_f64x224.sh) | 
| VideoMamba-S | K400  | 224 | 32x3x4   | 95.3 | [aliyun](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/videomamba/videomamba_s16_breakfast_f32_res224.pth), [:hugs:HF](https://huggingface.co/OpenGVLab/VideoMamba/blob/main/videomamba_s16_breakfast_f32_res224.pth) | [run.sh](./exp/breakfast/videomamba_small/run_f32x224.sh) | 
| VideoMamba-S | K400  | 224 | 64x3x4   | 97.4 | [aliyun](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/videomamba/videomamba_s16_breakfast_f64_res224.pth), [:hugs:HF](https://huggingface.co/OpenGVLab/VideoMamba/blob/main/videomamba_s16_breakfast_f64_res224.pth) | [run.sh](./exp/breakfast/videomamba_small/run_f64x224.sh) | 
| VideoMamba-M | K400  | 224 | 32x3x4   | 94.8 | [aliyun](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/videomamba/videomamba_m16_breakfast_f32_res224.pth), [:hugs:HF](https://huggingface.co/OpenGVLab/VideoMamba/blob/main/videomamba_m16_breakfast_f32_res224.pth) | [run.sh](./exp/breakfast/videomamba_middle/run_f32x224.sh) | 
| VideoMamba-M | K400  | 224 | 64x3x4   | 95.8 | [aliyun](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/videomamba/videomamba_m16_breakfast_f64_res224.pth), [:hugs:HF](https://huggingface.co/OpenGVLab/VideoMamba/blob/main/videomamba_m16_breakfast_f64_res224.pth) | [run.sh](./exp/breakfast/videomamba_middle/run_f64x224.sh) | 
| VideoMamba-M | *MASK*+K400  | 224 | 32x3x4   | 97.9 | [aliyun](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/videomamba/videomamba_m16_breakfast_mask_ft_f32_res224.pth), [:hugs:HF](https://huggingface.co/OpenGVLab/VideoMamba/blob/main/videomamba_m16_breakfast_mask_ft_f32_res224.pth) | [run.sh](./exp/breakfast/videomamba_middle_mask/run_f32x224.sh) | 
| VideoMamba-M | *MASK*+K400  | 224 | 64x3x4   | 96.9 | [aliyun](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/videomamba/videomamba_m16_breakfast_mask_ft_f64_res224.pth), [:hugs:HF](https://huggingface.co/OpenGVLab/VideoMamba/blob/main/videomamba_m16_breakfast_mask_ft_f64_res224.pth) | [run.sh](./exp/breakfast/videomamba_middle_mask/run_f64x224.sh) | 



### COIN

| Model    | Pretraining  | Resolution    | #Frame   | Top-1  | Model  | Shell  |
| -------- | ------------ | ------------  | -------- | ------ | ------ | ------ |
| VideoMamba-Ti | K400  | 224 | 32x3x10   | 86.2 | [aliyun](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/videomamba/videomamba_t16_coin_f32_res224.pth), [:hugs:HF](https://huggingface.co/OpenGVLab/VideoMamba/blob/main/videomamba_t16_coin_f32_res224.pth) | [run.sh](./exp/coin/videomamba_tiny/run_f32x224.sh) | 
| VideoMamba-Ti | K400  | 224 | 64x3x10   | 87.0 | [aliyun](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/videomamba/videomamba_t16_coin_f64_res224.pth), [:hugs:HF](https://huggingface.co/OpenGVLab/VideoMamba/blob/main/videomamba_t16_coin_f64_res224.pth) | [run.sh](./exp/coin/videomamba_tiny/run_f64x224.sh) | 
| VideoMamba-S | K400  | 224 | 32x3x10   | 88.4 | [aliyun](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/videomamba/videomamba_s16_coin_f32_res224.pth), [:hugs:HF](https://huggingface.co/OpenGVLab/VideoMamba/blob/main/videomamba_s16_coin_f32_res224.pth) | [run.sh](./exp/coin/videomamba_small/run_f32x224.sh) | 
| VideoMamba-S | K400  | 224 | 64x3x10   | 88.7 | [aliyun](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/videomamba/videomamba_s16_coin_f64_res224.pth), [:hugs:HF](https://huggingface.co/OpenGVLab/VideoMamba/blob/main/videomamba_s16_coin_f64_res224.pth) | [run.sh](./exp/coin/videomamba_small/run_f64x224.sh) | 
| VideoMamba-M | K400  | 224 | 32x3x10   | 88.3 | [aliyun](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/videomamba/videomamba_m16_coin_f32_res224.pth), [:hugs:HF](https://huggingface.co/OpenGVLab/VideoMamba/blob/main/videomamba_m16_coin_f32_res224.pth) | [run.sh](./exp/coin/videomamba_middle/run_f32x224.sh) | 
| VideoMamba-M | K400  | 224 | 64x3x10   | 89.5 | [aliyun](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/videomamba/videomamba_m16_coin_f64_res224.pth), [:hugs:HF](https://huggingface.co/OpenGVLab/VideoMamba/blob/main/videomamba_m16_coin_f64_res224.pth) | [run.sh](./exp/coin/videomamba_middle/run_f64x224.sh) | 
| VideoMamba-M | *MASK*+K400  | 224 | 32x3x10   | 89.6 | [aliyun](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/videomamba/videomamba_m16_coin_mask_ft_f32_res224.pth), [:hugs:HF](https://huggingface.co/OpenGVLab/VideoMamba/blob/main/videomamba_m16_coin_mask_ft_f32_res224.pth) | [run.sh](./exp/coin/videomamba_middle_mask/run_f32x224.sh) | 
| VideoMamba-M | *MASK*+K400  | 224 | 64x3x10   | 90.4 | [aliyun](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/videomamba/videomamba_m16_coin_mask_ft_f64_res224.pth), [:hugs:HF](https://huggingface.co/OpenGVLab/VideoMamba/blob/main/videomamba_m16_coin_mask_ft_f64_res224.pth) | [run.sh](./exp/coin/videomamba_middle_mask/run_f64x224.sh) | 


### LVU
For LVU, we originally sample frame from the raw videos sparsely, but the results are not stable due to the limited videos. However, we found that **[ViS4mer](https://github.com/md-mohaiminul/ViS4mer/blob/main/datasets/lvu_dataset.py) uses trimmed clips with sliding window, which may improve the results.** We also provide the related [dataset with sliding window](./datasets/lvu.py). Stay tuned!