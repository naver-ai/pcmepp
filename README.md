# Improved Probabilistic Image-Text Representations (PCME++) (ICLR 2024)

Official Python implementation of PCME++ | [Paper](https://arxiv.org/abs/2305.18171) | [Project page](https://naver-ai.github.io/pcmepp/)

[Sanghyuk Chun](https://sanghyukchun.github.io/home/)

This codebase is built upon the following repositories

- https://github.com/woodfrog/vse_infty
- https://github.com/naver-ai/pcme
- https://github.com/openai/CLIP

## Updates

- 07 Aug, 2023: Code is released!

## Installation

Please check the library version before you run the code:

```
lightning==2.0.1
torch==2.0
torchtext==0.15.1
torchvision==0.15.1
transformers
```

Or, simply run pip install (I strongly recommend making a new virtual environment before you run this):

```
pip3 install -r requirements.txt
```

## Dataset preparation

Step 1. Download COCO 2014 images from the official website: https://cocodataset.org/#download I may assume that your dataset file directory looks like

```
/path/to/dataset
└── images
    ├── train2014 # approximately 82k images are here
    └── val2014   # approximately 40k images are here
```

Step 2. Download annotation files from [this link](https://github.com/naver-ai/pcmepp/releases/download/v0.1.0/coco_annotations.tar.gz) and untar the annotations to the dataset path. It will make your dataset file directory will be

```
/path/to/dataset
└── images
    └── ...
├── id_mapping.json # mapping file for image and captions
├── cxc_annots      # annotations for CxC evaluation of VSE infty codebase
└── precomp         # caption annotations are here
    ├── train_caps.txt
    ├── train_ids.txt
    ├── dev_caps.txt
    ├── dev_ids.txt
    ├── test_caps.txt
    ├── test_ids.txt
    ├── testall_caps.txt
    └── testall_ids.txt
```

## Quick start

- Most of the experiments are reproducible with a single V100. If you want to use multiple GPUs (e.g., larger batch size, or larger model), you should specify `--train__dist_train` option.
- If you would like to run multiple experiments using this repository, it would be better to specify your `expname` using `train__expname`. The default `expname` is `results`, and all logs and weights will be dumped to `results`, if `expname` is not specified.

You can reproduce the main results by the following commands:

```
# PCME++ ViT-B/32 backbone
CUDA_VISIBLE_DEVICES=0 python3 train.py ./configs/pcmepp.yaml --dataloader__data_path /path/to/dataset

# PCME++ ViT-B/16 backbone
CUDA_VISIBLE_DEVICES=0 python3 train.py ./configs/pcmepp.yaml --dataloader__data_path /path/to/dataset --model__backbone_source clip_ViT-B/16

# PCME++ ViT-L/14 backbone
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 train.py ./configs/pcmepp.yaml --dataloader__data_path /path/to/dataset --model__backbone_source clip_ViT-L/14 --model__img_dim 1024 --dataloader__batch_size 16 --train__dist_train
```

This repository also provides `noise ratio` option as follows:

```
# PCME++ ViT-B/32 backbone with noise ratio 20%
CUDA_VISIBLE_DEVICES=0 python3 train.py ./configs/pcmepp.yaml --dataloader__data_path /path/to/dataset --dataloader__noise_ratio 0.2

# PCME++ ViT-B/32 backbone with noise ratio 50%
CUDA_VISIBLE_DEVICES=0 python3 train.py ./configs/pcmepp.yaml --dataloader__data_path /path/to/dataset --dataloader__noise_ratio 0.5
```

You can train the baselines methods using the following commands:

```
# ViT-B/32 backbones. Changing backbone is the same as the PCME++ backbone changes
CUDA_VISIBLE_DEVICES=0 python3 train.py ./configs/others/vse_infty.yaml --dataloader__data_path /path/to/dataset
CUDA_VISIBLE_DEVICES=0 python3 train.py ./configs/others/info_nce.yaml --dataloader__data_path /path/to/dataset
CUDA_VISIBLE_DEVICES=0 python3 train.py ./configs/others/pcmepp_mu_only.yaml --dataloader__data_path /path/to/dataset
CUDA_VISIBLE_DEVICES=0 python3 train.py ./configs/others/pcme.yaml --dataloader__data_path /path/to/dataset

# only exception is InfoNCE + multiple GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 train.py ./configs/others/info_nce.yaml --dataloader__data_path /path/to/dataset --model__backbone_source clip_ViT-L/14 --model__img_dim 1024 --dataloader__batch_size 16 --train__dist_train --train__all_gather_infonce
```

## Official weights

We will provide the official weights for each model in the paper.

## How to cite

```
@inproceedings{chun2024pcmepp,
    title={Improved Probabilistic Image-Text Representations},
    author={Chun, Sanghyuk},
    year={2024},
    booktitle={International Conference on Learning Representations (ICLR)},
}
```

I would like to suggest citing [PCME](https://github.com/naver-ai/pcme) and [ECCV Caption](https://github.com/naver-ai/eccv-caption), too.
```
@inproceedings{chun2021pcme,
    title={Probabilistic Embeddings for Cross-Modal Retrieval},
    author={Chun, Sanghyuk and Oh, Seong Joon and De Rezende, Rafael Sampaio and Kalantidis, Yannis and Larlus, Diane},
    year={2021},
    booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
}

@inproceedings{chun2022eccv_caption,
    title={ECCV Caption: Correcting False Negatives by Collecting Machine-and-Human-verified Image-Caption Associations for MS-COCO}, 
    author={Chun, Sanghyuk and Kim, Wonjae and Park, Song and Chang, Minsuk Chang and Oh, Seong Joon},
    year={2022},
    booktitle={European Conference on Computer Vision (ECCV)},
}
```

## License

```
MIT License

Copyright (c) 2023-present NAVER Cloud Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```
