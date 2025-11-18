# Feedback-Attention Memory Enhanced Video Swin Transformer for Long-Duration Sleep Stage Classification in Sows

In this research, we extend the Video Swin Transformer architecture by incorporating a **Feedback-Attention Memory (FAM)** mechanism to enable long-duration video classification for REM and NREM sleep in sows. Sleep is a temporally continuous biological process, and accurate sleep staging requires integrating behavioural context that evolves gradually over time. Conventional transformer-based video models typically operate on short fixed-length clips (e.g., 16–32 frames) and lack built-in mechanisms for preserving long-term spatiotemporal information.

![teaser](./images/architecture.png)



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


## VRAM Usage for 32×224×224 Clip Processing (Kinetics-400 Simulation)

We evaluated the Swin-Tiny and Swin-Small variants during both inference and training (AMP + gradient checkpointing), using **batch size = 1** and clip size = **32 × 224 × 224**.

| Scenario                                        | Approx. VRAM Usage |
| ----------------------------------------------- | ------------------ |
| **Inference – Swin-Tiny**                       | ~2.5 – 3.5 GB      |
| **Inference – Swin-Small**                      | ~3.5 – 5.0 GB      |
| **Training – Swin-Tiny (AMP + checkpointing)**  | ~4 – 6 GB          |
| **Training – Swin-Small (AMP + checkpointing)** | ~7 – 9 GB          |

These measurements suggest that **consumer GPUs with 8–12 GB of VRAM** are generally sufficient for:

* training the baseline models under our configuration, and
* performing extended-duration inference using the proposed FAM-enhanced architecture.
