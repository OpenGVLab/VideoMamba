# Model Zoo

## ImageNet-1K pretrained (224x224)

| Model         | Top-1 | #Param. | FLOPs | Model                                                        | Shell                                     |
| ------------- | ----- | ------- | ----- | ------------------------------------------------------------ | ----------------------------------------- |
| VideoMamba-Ti | 76.9  | 7M     | 1.1G  | [aliyun](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/videomamba/videomamba_t16_in1k_res224.pth), [:hugs:HF](https://huggingface.co/OpenGVLab/VideoMamba/blob/main/videomamba_t16_in1k_res224.pth) | [run.sh](./exp/videomamba_tiny/run224.sh)      |
| VideoMamba-S  | 81.2  | 26M     | 4.3G  | [aliyun](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/videomamba/videomamba_s16_in1k_res224.pth), [:hugs:HF](https://huggingface.co/OpenGVLab/VideoMamba/blob/main/videomamba_s16_in1k_res224.pth) | [run.sh](./exp/videomamba_small/run224.sh)      |
| VideoMamba-M w/ SD | 82.8  | 74M     | 12.7G  | [aliyun](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/videomamba/videomamba_m16_in1k_res224.pth), [:hugs:HF](https://huggingface.co/OpenGVLab/VideoMamba/blob/main/videomamba_m16_in1k_res224.pth) | [run.sh](./exp_distill/videomamba_middle/run224.sh)      |
| VideoMamba-B w/ SD | 82.7  | 98M     | 16.9G  | [aliyun](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/videomamba/videomamba_b16_in1k_res224.pth), [:hugs:HF](https://huggingface.co/OpenGVLab/VideoMamba/blob/main/videomamba_b16_in1k_res224.pth) | [run.sh](./exp_distill/videomamba_base/run224.sh)      |

> SD refers to Self-Distillation

## Large resolution fine-tuning

| Model         | Resolution | Top-1 | #Param. | FLOPs | Model                                                        | Shell                                     |
| ------------- | ---------- | ----- | ------- | ----- | ------------------------------------------------------------ | ----------------------------------------- |
| VideoMamba-Ti | 448     | 79.3  | 7M     | 4.3G  | [aliyun](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/videomamba/videomamba_t16_in1k_res224to448.pth), [:hugs:HF](https://huggingface.co/OpenGVLab/VideoMamba/blob/main/videomamba_m16_in1k_res224.pth) | [run.sh](./exp/videomamba_tiny/run448.sh)      |
| VideoMamba-Ti | 576     | 79.6  | 7M     | 7.1G  | [aliyun](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/videomamba/videomamba_t16_in1k_res224to448to576.pth), [:hugs:HF](https://huggingface.co/OpenGVLab/VideoMamba/blob/main/videomamba_t16_in1k_res224to448to576.pth) | [run.sh](./exp/videomamba_tiny/run576.sh)      |
| VideoMamba-S | 448     | 83.2  | 26M     | 16.9G  | [aliyun](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/videomamba/videomamba_s16_in1k_res224to448.pth), [:hugs:HF](https://huggingface.co/OpenGVLab/VideoMamba/blob/main/videomamba_s16_in1k_res224to448.pth) | [run.sh](./exp/videomamba_small/run448.sh)      |
| VideoMamba-S | 576     | 83.5  | 26M     | 28.0G  | [aliyun](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/videomamba/videomamba_s16_in1k_res224to448to576.pth), [:hugs:HF](https://huggingface.co/OpenGVLab/VideoMamba/blob/main/videomamba_s16_in1k_res224to448to576.pth) | [run.sh](./exp/videomamba_small/run576.sh)      |
| VideoMamba-M | 448     | 83.8  | 75M     | 50.4G  | [aliyun](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/videomamba/videomamba_m16_in1k_res224to448.pth), [:hugs:HF](https://huggingface.co/OpenGVLab/VideoMamba/blob/main/videomamba_m16_in1k_res224to448.pth) | [run.sh](./exp_distill/videomamba_middle/run448.sh)      |
| VideoMamba-M | 576     | 84.0  | 75M     | 83.1G  | [aliyun](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/videomamba/videomamba_m16_in1k_res224to448to576.pth), [:hugs:HF](https://huggingface.co/OpenGVLab/VideoMamba/blob/main/videomamba_m16_in1k_res224to448to576.pth) | [run.sh](./exp_distill/videomamba_middle/run576.sh)      |