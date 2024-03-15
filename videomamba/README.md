# Usage

- [x] [image_sm](./image_sm/README.md): Single-modality Image Tasks
  - Image Classification: [script](./image_sm/README.md) & [model](./image_sm/MODEL_ZOO.md)
- [x]  [video_sm](./video_sm/README.md): Single-modality Video Tasks
  - Short-term Video Understanding: [script](./video_sm/README.md) & [model](./video_sm/MODEL_ZOO.md)
  - Long-term Video Understanding: [script](./video_sm/README.md) & [model](./video_sm/MODEL_ZOO.md)
  - Masked Modeling: [script](./video_sm/README.md), [model](./video_sm/MODEL_ZOO.md)
- [x] [video_mm](./video_mm/README.md): Multi-modality Video Tasks
  - Video-Text Retrieval: [script](./video_sm/README.md) & [model](./video_sm/MODEL_ZOO.md)

## Installation

- Clone this repo:

  ```shell
  git clone https://github.com/OpenGVLab/VideoMamba
  cd VideoMamba
  ```

- Create Conda environments

  ```shell
  conda create -n mamba python=3.10
  conda activate mamba
  ```


- Install PyTorch 2.1.1+cu118

  ```shell
  pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
  ```

- Install `causal_conv1d` and `mamba`

  ```shell
  pip install -r requirements.txt
  pip install -e causal-conv1d
  pip install -e mamba
  ```
