# ATM: Adaptive Token Modulator for Vision Foundation Models in Traffic Scene Parsing

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch 1.8+](https://img.shields.io/badge/PyTorch-1.8+-red.svg)

> **Overcoming Roadside View Data Scarcity: Harnessing Vision Foundation Models for Traffic Scene Parsing**

## 🔥 News
- [2024-12] Manuscript under review, codebase will be made publicly available upon paper acceptance.
- [2024-12] Initial repository setup and documentation.

## 📋 Overview

This project introduces an innovative Adaptive Token Modulator (ATM) strategy that effectively leverages Vision Foundation Models (VFMs) to address data scarcity in roadside-view traffic scene parsing. Key features:

- 🚀 **High Efficiency**: Achieves SOTA prefermance with 78.9% mIoU on TSP6K with only 2.5% trainable parameters (7.7M)
- 💡 **Innovation**: First application of VFMs to roadside traffic scene parsing
- 🌟 **Strong Generalization**: Excellent performance in few-shot scenarios
- 🎯 **Practical Value**: Superior performance in challenging conditions (e.g., night scenes)

## 🗂️ Main Contributions

1. Construction of the **RS2K** dataset for roadside-view traffic scene parsing
2. Introduction of the novel **Adaptive Token Modulator (ATM)** strategy
3. Achievement of outstanding cross-scenario generalization capability

## 📊 Videos
https://github.com/user-attachments/assets/2dbba8c7-b4a2-4818-af12-d1a478e1383d
https://github.com/user-attachments/assets/cb8a9c54-7ec3-4745-85d0-fdb9ce4d0be8
https://github.com/user-attachments/assets/30d53f01-979f-48a6-b97e-e5f062af2356


## 📊 Performance

| Dataset | Zero-shot (1% params) | Few-shot (<10% data) |
|--------|----------------------|---------------------|
| Cityscapes | 76.28% | 78.34% |
| TSP6K | 54.57% | 62.35% |
| RS2K | 64.10% | 68.46% |

## 🔜 Coming Soon
- [✔] arxiv
- [✔] video
- [ ] code
- [ ] Pre-trained Models
- [ ] Detailed Documentation

## 🌟 Key Features

- Efficient utilization of limited roadside data
- Knowledge transfer from rich vehicle-mounted perspective datasets
- Low-rank token design for precise feature refinement
- Robust performance across diverse scenarios

## 💬 Citation

If you find our work helpful for your research, please consider citing:
```bibtex
Coming soon

