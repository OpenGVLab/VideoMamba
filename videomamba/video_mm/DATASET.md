# Dataset Preparation

We follow [VINDLU](https://github.com/klauscc/VindLU/) to prepare the datasets, but we **DO NOT** compress the videos and images.  We use the original data and load the JSON files, since there are some communication problems for SQLite in our environment.

:warning: If you do not have enough resources, we suggest you follow the preprocessing of [VINDLU](https://github.com/klauscc/VindLU/blob/main/DATA.md#compressing-videos-and-images).

:label: We use the same **JSON** files provided by [VINDLU](https://drive.google.com/drive/folders/12bC7WotvwyTG4pVvYeU4iZzmBLP1-6d9). However, since some vides are missing in large-scale datasets (like CC3M, CC12M and WebVid10M), we filter out those unavaliable videos.


## Pretraining

- CC3M images, https://github.com/google-research-datasets/conceptual-captions
- CC12M images, https://github.com/google-research-datasets/conceptual-12m
- SBU images, https://www.cs.rice.edu/~vo9/sbucaptions/
- VG images, https://visualgenome.org/api/v0/api_home.html
- COCO images, https://cocodataset.org/#download
- WebVid videos, https://github.com/m-bain/webvid


## Video-Text Retrieval and Video Question Answering

- MSRVTT videos, https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip
- MSVD videos, https://www.cs.utexas.edu/users/ml/clamp/videoDescription/
- ActivityNet videos, http://activity-net.org/download.html
- DiDeMo videos, https://github.com/LisaAnne/LocalizingMoments
- LSMDC videos, https://sites.google.com/site/describingmovies
