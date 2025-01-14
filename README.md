# ATM-Traffic: Adaptive Token Modulator for Roadside Traffic Scene Parsing

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Stars](https://img.shields.io/github/stars/IronmanVsThanos/ATM-Traffic)

This is the official implementation of "Cross-domain Traffic Scene Parsing via Vision Foundation Models: A Roadside Data Scarcity Solution"

## Video
sunday：
https://github.com/user-attachments/assets/2dbba8c7-b4a2-4818-af12-d1a478e1383d
# night：
https://github.com/user-attachments/assets/cb8a9c54-7ec3-4745-85d0-fdb9ce4d0be8
# night_rain:
https://github.com/user-attachments/assets/30d53f01-979f-48a6-b97e-e5f062af2356
# snow:
https://github.com/user-attachments/assets/2451c74b-0b4d-49dc-99b3-21f0a4617f20

## Overview

Traffic scene parsing from roadside views faces significant challenges due to limited data availability and poor generalization of existing methods. We propose ATM (Adaptive Token Modulator), a novel approach that:

- Efficiently leverages Vision Foundation Models (VFMs) for roadside traffic scene parsing
- Achieves SOTA performance with only 2.5% trainable parameters
- Shows strong generalization capability in zero-shot and few-shot scenarios
- Performs robustly in challenging conditions (night, rain, etc.)

## Key Features

- **Parameter Efficiency**: Achieves 78.9% mIoU on TSP6K using only 7.7M parameters (2.5% of full model)
- **Zero-shot Performance**: 
  - Cityscapes: 76.28% mIoU
  - TSP6K: 54.57% mIoU  
  - RS2K: 64.10% mIoU
- **Few-shot Learning**: With <10% training data achieves:
  - Cityscapes: 78.58% mIoU
  - TSP6K: 62.35% mIoU
  - RS2K: 68.46% mIoU

## Installation
```bash
# Clone the repository
git clone https://github.com/IronmanVsThanos/ATM-Traffic.git
cd ATM-Traffic
```
# Create and activate conda environment
```bash
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia -y
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
pip install "mmsegmentation>=1.0.0"
pip install "mmdet>=3.0.0"
pip install xformers=='0.0.20' # optional for DINOv2
pip install -r requirements.txt
pip install future tensorboard
# Install dependencies
pip install -r requirements.txt
```


## Dataset Preparation
```bash
├── data
│   ├── cityscapes
│   │   ├── leftImg8bit
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── gtFine
│   │   │   ├── train
│   │   │   ├── val
│   ├── tsp6k
│   │   ├── leftImg8bit
│   │   |   ├── train
│   │   │   ├── val
│   │   ├── labels
│   │   |   ├── train
│   │   │   ├── val
```
## Pretraining Weights
Download: Download pre-trained weights(512*512 and 1024*1024) from:

## Training
```bash
PORT=12345 CUDA_VISIBLE_DEVICES=1,2,3,4 bash tools/dist_train.sh configs/dinov2/atm_dinov2_mask2former_1024x1024_bs4x2.py NUM_GPUS
```
## Evaluation
```bash
CUDA_VISIBLE_DEVICES=1,2,3,4 python tools/test.py configs/dinov2/atm_dinov2_mask2former_1024x1024_bs4x2.py  work_dirs/atm_dinov2_mask2former_1024x1024_bs4x2/iter_40000.pth --backbone ./checkpoints/dinov2_converted_1024x1024.pth
```
## Visulioze
```bash
for img: python tools/visualize.py /path/to/cfg /path/to/checkpoint /path/to/images --backbone /path/to/converted_backbone
for video: python tools/visualize_video.py /path/to/cfg /path/to/checkpoint /path/to/images --backbone /path/to/converted_backbone
```


