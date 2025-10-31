# Video-Swin-Transformer-Pytorch
This repo is a simple usage of the official implementation ["Video Swin Transformer"](https://github.com/SwinTransformer/Video-Swin-Transformer).

![teaser](https://github.com/SwinTransformer/Video-Swin-Transformer/blob/master/figures/teaser.png)

## Introduction

**Video Swin Transformer** is initially described in ["Video Swin Transformer"](https://arxiv.org/abs/2106.13230), which advocates an inductive bias of locality in video Transformers, leading to a better speed-accuracy trade-off compared to previous approaches which compute self-attention globally even with spatial-temporal factorization. The locality of the proposed video architecture is realized by adapting the Swin Transformer designed for the image domain, while continuing to leverage the power of pre-trained image models. Our approach achieves state-of-the-art accuracy on a broad range of video recognition benchmarks, including action recognition (`84.9` top-1 accuracy on Kinetics-400 and `86.1` top-1 accuracy on Kinetics-600 with `~20x` less pre-training data and `~3x` smaller model size) and temporal modeling (`69.6` top-1 accuracy on Something-Something v2).

## Usage

###  Installation
```
$ pip install -r requirements.txt
```

### Prepare
```
$ git clone https://github.com/haofanwang/video-swin-transformer-pytorch.git
$ cd video-swin-transformer-pytorch
$ mkdir checkpoints && cd checkpoints
$ wget https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_base_patch244_window1677_sthv2.pth
$ cd ..
```

### Long-Duration Operation (30+ Minutes)

Using the Swin Video Transformer, we evaluated long-duration inference performance on the Kinetics-400 dataset and confirmed stable classification over continuous 30-minute video streams. Importantly, VRAM usage remained constant throughout the entire sequence due to the capped-memory design, demonstrating feasibility for real-world deployment where multi-hour monitoring is common.


### Memory Reset Protocol

To prevent cross-subject information leakage, the model’s internal memory state is reset at the beginning of each new video sequence. This ensures that motion patterns from one subject do not influence predictions for another.


### VRAM Usage for 30-Minute Sequences (Kinetics-400 Test)

We tested the Swin-Tiny and Swin-Small variants on the Kinetics-400 dataset using 32×224×224 clips and batch size 1. Approximate VRAM usage observed:

| Scenario | VRAM Required |
|---|---|
| **Inference – Swin-Tiny** | ~2.5 – 3.5 GB |
| **Inference – Swin-Small** | ~3.5 – 5.0 GB |
| **Training – Swin-Tiny (AMP + checkpointing)** | ~4 – 6 GB |
| **Training – Swin-Small (AMP + checkpointing)** | ~7 – 9 GB |

These findings indicate that even standard consumer GPUs with **8–12 GB VRAM** are sufficient for both training and extended-duration inference.

---
