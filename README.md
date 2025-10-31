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

### Assumptions Long-Duration Operation (30+ Minutes)

With this architecture, we successfully classify sleep stage behaviour continuously over periods of 30 minutes and beyond. Importantly, due to the capped memory mechanism, GPU VRAM requirements remain constant, irrespective of total video duration. This enables deployment in real-world farm monitoring setups where multi-hour recordings are common.


### Memory Reset Protocol

To avoid cross-subject information leakage, the feedback memory is reset at the beginning of each new sow recording or nightly session. This ensures that identity-specific motion features do not bias classification.


### Output Smoothing and Stability

Although the model produces per-clip predictions, sleep stage changes are inherently slow. To reduce spurious transitions, I apply lightweight temporal smoothing via a median filter over recent clip outputs (typically 5–15 clips, equivalent to ~40–90 seconds depending on fps). This step improves classification stability without sacrificing responsiveness to true REM/NREM transitions.


### VRAM Requirements

A key advantage of the proposed design is efficient memory usage. During inference for a single sow, using the Swin-Tiny or Swin-Small backbone with the FAM module, clip size of 32×224×224, and batch size of 1, the model operates with:

| Scenario | VRAM Required |
|---|---|
| **Inference (Tiny + FAM)** | ~2.5 – 3.5 GB |
| **Inference (Small + FAM)** | ~3.5 – 5.0 GB |
| **Training (Tiny + FAM, AMP + checkpointing)** | ~4 – 6 GB |
| **Training (Small + FAM, AMP + checkpointing)** | ~7 – 9 GB |

This confirms that even consumer-grade GPUs (e.g., 8–12 GB VRAM) are capable of efficiently training and deploying the system for long-duration sow sleep analysis.
