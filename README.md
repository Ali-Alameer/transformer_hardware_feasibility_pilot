# Feedback-Attention Memory Enhanced Video Swin Transformer for Long-Duration Video Classification

We conducted this pilot study to inform a research proposal recently submitted to the ERC Synergy Grant. In this research, we extend the Video Swin Transformer architecture by incorporating a **Feedback-Attention Memory (FAM)** mechanism to enable **long-duration video classification for continuous behavioural monitoring**. Many real-world video understanding tasks—such as livestock behaviour analysis, smart surveillance, and industrial process monitoring—require integrating visual context that evolves gradually over long time periods. Conventional transformer-based video models typically operate on short fixed-length clips (e.g., 16–32 frames) and lack built-in mechanisms for preserving long-term spatiotemporal information across long video streams.

Our approach explicitly models temporal continuity while maintaining **bounded but configurable memory usage** and predictable compute costs.

## Proposed Method

To overcome these limitations, we augment the baseline Video Swin Transformer with a lightweight **Feedback-Attention Memory (FAM)** module inspired by recurrent-memory and feedback-attention designs. The proposed module maintains a compact set of summary memory tokens derived from previously processed video clips, enabling the model to incorporate longer-term behavioural context during inference.

### Memory Token Generation

* Each 32-frame clip is processed by the Video Swin Transformer backbone to obtain multi-stage hierarchical features.
* At selected deeper transformer stages, spatiotemporal tokens are pooled to form **128 memory summary tokens per stage**.
* These summaries are appended to a **bounded external memory queue** with fixed length `L_mem`.
### Cross-Attention With Historical Memory

For every new incoming clip:

* Current clip tokens perform standard **local-window self-attention**, as in the original Swin Transformer.
* In addition, they perform **cross-attention with stored memory tokens**, allowing the model to incorporate historical motion patterns, posture persistence, and gradual activity transitions.

This design introduces long-range temporal continuity similar to recurrent memory, without expanding the backbone’s temporal window.

### Efficient Memory Management

* The external memory queue is updated sequentially as clips are processed.
* Its maximum size is fixed (`L_mem`), ensuring **predictable and bounded VRAM usage**.
* Because clip shape and memory size remain constant, the model supports inference over **arbitrarily long video streams (e.g., ≥30 minutes)** without unbounded GPU memory growth.

## Long-Duration Operation (30+ Minutes)

To evaluate long-stream performance, we simulated extended video sequences by **concatenating Kinetics-400 clips** into continuous ~30-minute streams. The model processed these streams using a **fixed-length sliding window** of 32 frames per forward pass.

Our memory-aware architecture supports temporal reasoning over **30-minute video sequences**. Pilot experiments on a custom dataset (supporting calculations and code are provided in this GitHub repository) confirmed that **30-minute inference requires approximately 2.5–3.5 GB VRAM per stream**, excluding parallel batch replication. Accordingly, a single **96 GB GPU** can process **approximately 32–40 video streams in parallel**, enabling large-scale behavioural analysis deployments.

Peak GPU memory usage remained approximately constant throughout the entire 30-minute sequence because:

* clip size was fixed,
* batch size per stream was 1,
* memory queue length `L_mem` was bounded.

These findings confirm that the streaming design and the proposed FAM architecture are suitable for **long-duration, high-throughput video monitoring systems**.

## Total Memory Queue Size

Given a memory length of **L_mem = 25–30** clips, the total memory footprint is:

### Examples

| L_mem | Total Memory | VRAM Usage |
| ----: | -----------: | ---------: |
|    10 |      ~1.2 GB |   Moderate |
|    20 |      ~2.5 GB |       High |
|    25 |      ~3.1 GB |       High |
|    30 |  ~3.8–4.0 GB |       High |

This memory usage includes **stored memory tokens and associated attention buffers**, and represents a deliberate trade-off to enable richer long-term temporal reasoning.


## Cross-Attention Computation Cost

Cross-attention operates on:

* **Query:** current clip tokens (approximately 300–400 tokens)
* **Key/Value:** memory tokens (128 × `L_mem`)

For `L_mem = 25`:

* Total memory tokens: **3,200**
* Cross-attention cost scales **linearly with `L_mem`**

Despite the increased memory capacity, the computation remains tractable due to the bounded queue and fixed clip size.


## Key Memory Advantages

* **Bounded and predictable VRAM usage**
* **Streaming-compatible**, supporting hours-long video streams
* **Configurable memory–accuracy trade-off**
* **No accumulation of unbounded activations**
* Well-suited for **datacenter-scale GPUs (e.g., 80–96 GB VRAM)**


## Inference Mode VRAM Breakdown (Per Stream)

| Component                        | VRAM Usage               |
| -------------------------------- | ------------------------ |
| Video Swin Transformer backbone  | 2–6 GB (Swin-T / Swin-S) |
| FAM Memory Queue (`L_mem=25–30`) | **~3–4 GB**              |
| Temporary buffers                | 0.5–1 GB                 |
| **Total per stream**             | **~2.5–3.5 GB**          |


## Why Memory Stays Constant

* Clip size is fixed at **32 frames**
* Batch size per stream is **1**
* Memory queue length is fixed (`L_mem`)
* No dynamic feature caching
* No growing hidden states, unlike recurrent neural networks

**As a result, GPU memory remains constant during 30+ minutes of continuous streaming.**

## Memory Lifetime

* Each new clip contributes a summary to the memory queue.
* When the queue exceeds `L_mem`, the oldest entries are removed.
* The system operates as a sliding temporal window with a stable, bounded memory footprint.

