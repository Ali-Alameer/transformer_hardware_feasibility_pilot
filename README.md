# Feedback-Attention Memory Enhanced Video Swin Transformer for Long-Duration Sleep Stage Classification in Sows

In this research, we extend the Video Swin Transformer architecture by incorporating a **Feedback-Attention Memory (FAM)** mechanism to enable long-duration video classification for REM and NREM sleep in sows. Sleep is a temporally continuous biological process, and accurate sleep staging requires integrating behavioural context that evolves gradually over time. Conventional transformer-based video models typically operate on short fixed-length clips (e.g., 16–32 frames) and lack built-in mechanisms for preserving long-term spatiotemporal information.



## Proposed Method

To overcome these limitations, we augment the baseline Video Swin Transformer with a lightweight **FAM module** inspired by recurrent-memory and feedback-attention designs. The proposed module maintains a compact set of summary memory tokens derived from previous video clips, enabling the model to incorporate longer-term behavioural context during inference.

### Memory Token Generation

* Each 32-frame clip is processed by the Video Swin Transformer backbone to obtain multi-stage hierarchical features.
* At selected deeper transformer stages, spatiotemporal tokens are pooled to form **128 memory summary tokens per stage**.
* These summaries are appended to a **bounded external memory queue** with fixed length `L_mem`, representing approximately **1–2 minutes** of prior behaviour (depending on FPS).

### Cross-Attention With Historical Memory

For every new incoming clip:

* Current clip tokens perform standard **local-window self-attention** (as in the original Swin Transformer).
* In addition, they perform **cross-attention with the stored memory tokens**, allowing the model to incorporate historical posture and motion patterns.

This mechanism introduces long-range contextual continuity similar to recurrent memory, without modifying the backbone’s temporal window.

### Efficient Memory Management

* The external memory queue is updated sequentially as clips are processed.
* Its maximum size is fixed (`L_mem`), ensuring that **both VRAM usage and compute cost remain bounded**.
* Because clip shape and memory size remain constant, the model supports inference over **arbitrarily long video streams (e.g., ≥30 minutes)** without increasing GPU memory consumption.



## Clip and Sampling Strategy

Sow sleep transitions evolve gradually; therefore, high frame rates are unnecessary. Empirically, a sampling rate of **2–4 FPS** captures essential cues such as posture, immobility, and subtle twitches.

### Overlap and Temporal Resolution

* Clips advance using a **stride of 8 frames**, resulting in **75% overlap**.
* With clip size = 32 frames, this yields classification updates every:

  * **2 seconds at 4 FPS**,
  * **4 seconds at 2 FPS**,
    offering fine-grained temporal resolution suitable for sleep-stage segmentation.


## Long-Duration Operation (30+ Minutes)

To evaluate long-stream performance, we simulated extended video sequences by **concatenating Kinetics-400 clips** into continuous ~30-minute streams (Kinetics-400 contains ~10-second trimmed clips; thus concatenation is required for long-duration simulation). The model processed these streams using a **fixed-length sliding window** of 32 frames per forward pass.

In our experiments, **peak GPU memory usage remained approximately constant** throughout the entire 30-minute sequence because:

* clip size was fixed,
* batch size was 1,
* memory queue length `L_mem` was bounded.

These findings confirm that the streaming design and the proposed FAM architecture are suitable for long-duration deployment.


Here is **all the memory-related information**, rewritten cleanly in **Markdown format**, ready for documentation, a paper, or your GitHub README.

# Memory Usage Summary

## **Long-Term Memory Module (FAM) Overview**

The Feedback-Attention Memory (FAM) module maintains a compact and bounded memory representing 1–2 minutes of past behaviour.
Memory size is fixed, ensuring constant GPU usage during long-duration inference.

## **Memory Queue Specifications**

### **Memory Token Size**

* Number of memory tokens per clip: **128**
* Token dimension: **768**
* Memory per clip:

```
128 tokens × 768 dims × 4 bytes ≈ 0.38 MB
```

## **Total Memory Queue Size**

Given a memory length of **L_mem = 20–30** clips:

```
Memory = 128 × 768 × L_mem × 4 bytes
```

### **Examples**

| L_mem | Total Memory | VRAM Usage |
| ----- | ------------ | ---------- |
| 10    | ~3.9 MB      | Very Low   |
| 20    | ~7.7 MB      | Low        |
| 25    | ~9.8 MB      | Low        |
| 30    | ~11.5 MB     | Low        |

Even at maximum capacity, the FAM module uses **< 12 MB VRAM**, which is negligible compared to backbone features.

## **Cross-Attention Computation Cost**

Cross-attention uses:

```
Query:    current clip tokens  (≈ 300–400 tokens)
Key/Val:  memory tokens        (128 × L_mem)
```

For L_mem = 25:

```
Total memory tokens = 128 × 25 = 3200 tokens
```

The cross-attention cost is lightweight and scales linearly with L_mem.

## **Key Memory Advantages**

* **Constant VRAM usage** during long-duration inference
* **Streaming-compatible** (process hours-long videos)
* **Negligible memory footprint** (<12 MB total)
* **No accumulation of activations**
* Suitable for **edge devices** (Jetson Orin / Xavier)


## **Inference Mode VRAM Breakdown**

| Component                         | VRAM Usage                      |
| --------------------------------- | ------------------------------- |
| Video Swin Transformer (backbone) | 2–10 GB (depends on Swin-T/S/B) |
| FAM Memory Queue                  | **< 12 MB**                     |
| Temporary buffers                 | 0.5–1 GB                        |
| **Total (Swin-T)**                | **~2.5–3.5 GB**                 |
| **Total (Swin-S)**                | **~4.5–6 GB**                   |
| **Total (Swin-B)**                | **~8–10 GB**                    |


## **Why Memory Stays Constant**

* Clip size is fixed: **32 frames**
* Batch size = **1**
* Memory queue length is fixed: **L_mem**
* No dynamic feature caching
* No growing hidden states (unlike RNN/LSTM)

Therefore:

> **GPU memory stays constant during 30+ minutes of continuous streaming.**

## **Memory Lifetime**

* Each new clip adds a summary to the memory queue.
* When queue exceeds L_mem, oldest memory entries are removed.
* Sliding window over time = stable, bounded memory footprint.
