# Model Zoo

All the model weights are saved with the `clip_teacher`, which are loaded from the CLIP vision encoder.

## Pretraining

We load those models with K400 masked pretraining and further pretrain them on multimodality data.

- 5M: CC3M + WebVid2M
- 17M: CC3M + CC12M + COCO + VG + SBU + WebVid2M
- 25M: CC3M + CC12M + COCO + VG + SBU + WebVid10M

| Model    | Setting     | Model  | Script  |
| -------- | ----------- | ------ | ------- |
| VideoMamba-M | 5M          |  [aliyun](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/videomamba/videomamba_m16_5M_f8_res224.pth), [:hugs:HF](https://huggingface.co/OpenGVLab/VideoMamba/blob/main/videomamba_m16_5M_f8_res224.pth) | [script](./exp_pt/videomamba_middle_5m/run.sh)  |
| VideoMamba-M | 17M          |  [aliyun](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/videomamba/videomamba_m16_17M_f8_res224.pth), [:hugs:HF](https://huggingface.co/OpenGVLab/VideoMamba/blob/main/videomamba_m16_17M_f8_res224.pth) | [script](./exp_pt/videomamba_middle_17m/run.sh)  |
| VideoMamba-M | 25M          |  [aliyun](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/videomamba/videomamba_m16_25M_f8_res224.pth), [:hugs:HF](https://huggingface.co/OpenGVLab/VideoMamba/blob/main/videomamba_m16_25M_f8_res224.pth) | [script](./exp_pt/videomamba_middle_25m/run.sh)  |


### Zero-shot Evaluation

<div align="left">
<table width="100%">
    <tr align="center">
        <th rowspan="2">Dataset</th><th rowspan="2">Retrieval</th><th colspan="3">VideoMamba-M</th>
    </tr>
    <tr align="center">
        <th>5M</th><th>17M</th><th>25M</th>
    </tr>
    <tr align="center">
        <th rowspan="2">MSRVTT</th>
        <td>T2V</td>
        <td align='left'>R@1: 32.0<br>R@5: 53.1<br>R@10: 63.6<br></td>
		<td align='left'>R@1: 34.7<br>R@5: 58.9<br>R@10: 68.0<br></td>
		<td align='left'>R@1: 35.6<br>R@5: 58.1<br>R@10: 69.5<br></td>
    </tr>
    <tr align="center">
        <td>V2T</td>
        <td align='left'>R@1: 28.2<br>R@5: 47.6<br>R@10: 56.5<br></td>
		<td align='left'>R@1: 29.5<br>R@5: 49.9<br>R@10: 60.1<br></td>
		<td align='left'>R@1: 29.1<br>R@5: 51.6<br>R@10: 62.2<br></td>
    </tr>
    <tr align="center">
        <th rowspan="2">DiDeMo</th>
        <td>T2V</td>
       <td align='left'>R@1: 36.6<br>R@5: 61.7<br>R@10: 70.3<br></td>
		<td align='left'>R@1: 42.0<br>R@5: 67.3<br>R@10: 76.8<br></td>
		<td align='left'>R@1: 43.1<br>R@5: 68.1<br>R@10: 77.7<br></td>
    </tr>
    <tr align="center">
        <td>V2T</td>
        <td align='left'>R@1: 38.3<br>R@5: 64.7<br>R@10: 73.3<br></td>
		<td align='left'>R@1: 42.3<br>R@5: 68.2<br>R@10: 76.9<br></td>
		<td align='left'>R@1: 43.8<br>R@5: 69.7<br>R@10: 77.8<br></td>
    </tr>
    <tr align="center">
        <th rowspan="2">ActivityNet</th>
        <td>T2V</td>
        <td align='left'>R@1: 35.9<br>R@5: 61.1<br>R@10: 72.3<br></td>
		<td align='left'>R@1: 40.1<br>R@5: 65.7<br>R@10: 76.1<br></td>
		<td align='left'>R@1: 41.0<br>R@5: 67.5<br>R@10: 77.8<br></td>
    </tr>
    <tr align="center">
        <td>V2T</td>
        <td align='left'>R@1: 32.8<br>R@5: 58.8<br>R@10: 69.9<br></td>
		<td align='left'>R@1: 34.2<br>R@5: 61.8<br>R@10: 73.2<br></td>
		<td align='left'>R@1: 37.1<br>R@5: 65.0<br>R@10: 75.1<br></td>
    </tr>
    <tr align="center">
        <th rowspan="2">LSMDC</th>
        <td>T2V</td>
        <td align='left'>R@1: 18.0<br>R@5: 36.1<br>R@10: 43.4<br></td>
		<td align='left'>R@1: 18.4<br>R@5: 35.3<br>R@10: 43.0<br></td>
		<td align='left'>R@1: 20.4<br>R@5: 37.1<br>R@10: 45.7<br></td>
    </tr>
    <tr align="center">
        <td>V2T</td>
        <td align='left'>R@1: 15.9<br>R@5: 31.0<br>R@10: 39.2<br></td>
		<td align='left'>R@1: 16.5<br>R@5: 32.1<br>R@10: 40.0<br></td>
		<td align='left'>R@1: 17.9<br>R@5: 34.6<br>R@10: 42.1<br></td>
    </tr>
    <tr align="center">
        <th rowspan="2">MSVD</th>
        <td>T2V</td>
        <td align='left'>R@1: 38.0<br>R@5: 68.6<br>R@10: 79.0<br></td>
		<td align='left'>R@1: 40.3<br>R@5: 70.0<br>R@10: 79.7<br></td>
		<td align='left'>R@1: 42.0<br>R@5: 71.6<br>R@10: 81.2<br></td>
    </tr>
    <tr align="center">
        <td>V2T</td>
        <td align='left'>R@1: 57.5<br>R@5: 79.9<br>R@10: 85.4<br></td>
		<td align='left'>R@1: 61.8<br>R@5: 81.0<br>R@10: 87.0<br></td>
		<td align='left'>R@1: 62.7<br>R@5: 82.8<br>R@10: 87.6<br></td>
    </tr>
</table>
<br>
</div>
