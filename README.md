# :snake:VideoMamba

<div align="center">

<h2><a href="https://arxiv.org/abs/2403.06977">VideoMamba: State Space Model for Efficient Video Understanding</a></h2>

[Kunchang Li](https://scholar.google.com/citations?user=D4tLSbsAAAAJ), [Xinhao Li](https://leexinhao.github.io/), [Yi Wang](https://scholar.google.com.hk/citations?hl=zh-CN&user=Xm2M8UwAAAAJ), [Yinan He](https://scholar.google.com/citations?user=EgfF_CEAAAAJ), [Yali Wang](https://scholar.google.com/citations?user=hD948dkAAAAJ), [Limin Wang](https://scholar.google.com/citations?user=HEuN8PcAAAAJ) and [Yu Qiao](https://scholar.google.com/citations?user=gFtI-8QAAAAJ&hl)

[![Paper](https://img.shields.io/badge/cs.CV-2403.06977-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2403.06977) [![Paper](https://img.shields.io/badge/%F0%9F%A4%97%20Paper-2403.06977-blue)](https://huggingface.co/papers/2403.06977) [![Paper](https://img.shields.io/badge/%F0%9F%A4%97%20Model-VideoMamba-blue)](https://huggingface.co/OpenGVLab/VideoMamba) [![Space](https://img.shields.io/badge/%F0%9F%A4%97%20Space-VideoMamba-blue)](https://huggingface.co/spaces/OpenGVLab/VideoMamba) [![Paper](https://img.shields.io/badge/Blog-VideoMamba-blue?logo=zhihu&logoColor=blue)](https://zhuanlan.zhihu.com/p/688140932)

![teaser](./assets/comparison.png)
</div>


## Update

- :warning: **2024/03/25**: <s>The current video models are **fine-tuned without layer decay** due to the [bug](https://github.com/OpenGVLab/VideoMamba/blob/9806e369196c88f3eef92380c74beeb51208d0c3/videomamba/video_sm/optim_factory.py#L24), which may help to improve the performances as in [MAE](https://github.com/facebookresearch/mae). We have fixed the bug but do not plan to retrain them.</s> We have applied it for VideoMamba-M but it does not help.
- **2024/03/13**: Fix some bugs and add :hugs:HF model links.
- :fire: **2024/03/12**: **All the code and models are released.**
  - [image_sm](./videomamba/image_sm/README.md): Single-modality Image Tasks
    - Image Classification: [script](./videomamba/image_sm/README.md) & [model](./videomamba/image_sm/MODEL_ZOO.md)
  - [video_sm](./videomamba/video_sm/README.md): Single-modality Video Tasks
    - Short-term Video Understanding: [script](./videomamba/video_sm/README.md#short-term-video-understanding) & [model](./videomamba/video_sm/MODEL_ZOO.md#short-term-video-understanding)
    - Long-term Video Understanding: [script](./videomamba/video_sm/README.md#long-term-video-understanding) & [model](./videomamba/video_sm/MODEL_ZOO.md#long-term-video-understanding)
    - Masked Modeling: [script](./videomamba/video_sm/README.md#masked-pretraining), [model](./videomamba/video_sm/MODEL_ZOO.md#masked-pretraining)
  - [video_mm](./videomamba/video_mm/README.md): Multi-modality Video Tasks
    - Video-Text Retrieval: [script](./videomamba/video_mm/README.md) & [model](./videomamba/video_mm/MODEL_ZOO.md)


## Introduction

![teaser](./assets/framework.png)
Addressing the dual challenges of local redundancy and global dependencies in video understanding, this work innovatively adapts the Mamba to the video domain. The proposed VideoMamba overcomes the limitations of existing 3D convolution neural networks and video transformers. Its linear-complexity operator enables efficient long-term modeling, which is crucial for high-resolution long video understanding. Extensive evaluations reveal VideoMamba's four core abilities: (1) Scalability in the visual domain without extensive dataset pretraining, thanks to a novel self-distillation technique; (2) Sensitivity for recognizing short-term actions even with fine-grained motion differences; (3) Superiority in long-term video understanding, showcasing significant advancements over traditional feature-based models; and (4) Compatibility with other modalities, demonstrating robustness in multi-modal contexts. Through these distinct advantages, VideoMamba sets a new benchmark for video understanding, offering a scalable and efficient solution for comprehensive video understanding. 


## Cite

If you find this repository useful, please use the following BibTeX entry for citation.

```latex
@misc{li2024videomamba,
      title={VideoMamba: State Space Model for Efficient Video Understanding}, 
      author={Kunchang Li and Xinhao Li and Yi Wang and Yinan He and Yali Wang and Limin Wang and Yu Qiao},
      year={2024},
      eprint={2403.06977},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## License

This project is released under the [Apache 2.0 license](./LICENSE)

## Acknowledgement

This repository is built based on [UniFormer](https://github.com/Sense-X/UniFormer), [Unmasked Teacher](https://github.com/OpenGVLab/unmasked_teacher) and [Vim](https://github.com/hustvl/Vim) repository.
