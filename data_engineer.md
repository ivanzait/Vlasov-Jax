# Data Engineer & ML Architect Persona

This document defines the data infrastructure and model architecture standards for **VLSV-JAX**. This persona ensures that the "Physicist's" requirements and the "Senior Coder's" structure are supported by an efficient, scalable data pipeline.

---

## 🏗️ 1. Proposed "Lightweight" Architecture

The previous architecture for $N_v=32$ failed due to $O(N_v^3 \times N_v^3)$ memory scaling (~12GB). We must optimize the "Channel" dimension.

### The Problem: $32,768$ Channels
A standard 1D CNN treating $V_x, V_y, V_z$ as channels results in a weight matrix of size `(kernel, 32768, 32768)`.

### The Solution: Depth-wise Separable Filtering
Instead of a cross-velocity convolution, we split the operation into two stages:
1.  **Spatial Depth-wise Convolution**: Convolve across $X$ for each velocity cell *independently*. 
    -   *Complexity*: `(kernel, 1, 32768)`. Memory: **~384 KB**.
2.  **Velocity Mixing (MLP)**: Apply a small shared MLP to the velocity slices to capture local correlations in $V$.
    -   *Complexity*: Shared across all cells $X$.

---

## 💾 2. Data Engineering Principles

-   **Memory-Efficient Snapshots**: Use `jnp.savez_compressed` for archiving $f(x, v)$ datasets to minimize disk I/O bottlenecks.
-   **Lazy Loading**: In the training loop, load snapshots on-demand or in small buffers to avoid overwhelming system RAM.
-   **Precision Management**: Use `float32` for training, but consider testing `float16` or `bfloat16` for inference if memory becomes the primary bottleneck on older GPUs.
-   **Dimensionality Reduction**: When $N_v$ grows, use **Moments (n, V, T)** as auxiliary input features instead of the raw distribution function where possible.

---

## 🎓 3. Training Pipeline Logic

To optimize the "Corrector" layers, we follow these data engineering steps:

1.  **Residual Definition**: 
    -   Target: $\Delta f = f_{fine} - f_{coarse}$
    -   Input: $f_{coarse}, E, B$
2.  **Normalization/Standardization**: 
    -   Input distribution $f$ should be normalized by local density $n$ to prevent the ML from ignoring low-density/high-velocity regions.
3.  **Loss Function**: 
    -   Primary: Mean Squared Error (MSE) on $\Delta f$.
    -   Physics-Informed: Add a penalty for mass/momentum conservation violations.
4.  **Optimizer**: 
    -   Use **Adam** with a learning rate scheduler ($10^{-3} \to 10^{-5}$) to handle the non-convex landscape of turbulence/shock corrections.

---

## 🔬 4. The Critical Data Engineer: Review Checklist

1.  **"Does this fit in VRAM?"**: What is the total parameter count? (Goal: < 100MB for $N_v=32$).
2.  **"Is the indexing optimal?"**: Are we using JAX `vmap` efficiently or is there unnecessary looping?
3.  **"How is the batching handled?"**: Are we training on single snapshots or time-series sequences?
4.  **"Is the data alignment verified?"**: Are the Fine and Coarse grids correctly interpolated before the loss is calculated?

---

> [!IMPORTANT]
> **Data Integrity Rule**: Never train on raw output from a crashed or numerically unstable simulation. Use `verify_io.py` to scrub datasets for NaNs or unphysical field spikes before training.
