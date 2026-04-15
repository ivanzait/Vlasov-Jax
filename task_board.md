# VLSV-JAX Task Board & Knowledge Ledger

This document tracks the evolution of the Physics-ML augmented solver, capturing simulation checkpoints and shared expertise.

---

## 🚀 1. Current Project Status
- **Active Fingerprint**: `NX=64, NV=32(coarse)/64(fine), DT=0.05, BC=('static','copy'), 2-cell inflow buffer`
- **ML Regime**: Log-Space Multiplicative Corrections ($f \cdot \exp(g)$), Online-Only
- **Architecture**: Boundary Masking (Buffer=2), Tuning Clamps, Dynamic Kernel
- **Initial Condition**: Warm-Start (Zero-Bias Weights), $\sigma=1e-4$
- **Current Objective**: Experimentation phase (Stable Boundary Sharpening confirmed).

---

## 🏛️ 2. The Brain: Shared Knowledge

### ⚡ Physicist's Insights
- **The Tail Variance Discovery**: Distribution functions in shock simulations span 10 orders of magnitude. Standard MSE loss ignores the tails. **Log-space transformations** are mandatory to homogenize the learning signal and guarantee $f > 0$.
- **Boundary Interaction**: Internal ML padding (CNN) at boundaries can lead to unphysical discontinuities. Hard-enforcement of BCs *between* Strang sub-steps is required for stability when ML is active.

### 🏛️ Senior Coder's Insights
- **Multiplicative Residuals**: Switching from $f + \Delta f$ to $f \cdot \exp(g)$ is the most robust way to integrate NNs into a kinetic solver. It leverages the natural exponential form of the distribution function.
- **Layered Orchestration**: Maintaining a decoupled `nn_models.py` allows us to swap architectures (CNN vs. MLP) without touching the sensitive Faraday/Ohm's Law solvers in `solver_maxwell.py`.
- **Online-Only Training**: `train_corrector.py` is a utility library, not a script. All weight updates happen inside `run_maxwell.py` via the solver-in-the-loop hook. No offline `python train_corrector.py` workflow exists.
- **2-Cell BC Buffer**: Static inflow BC must clamp `f[0:2]` not just `f[0]` — otherwise the CNN sees an artificial 3% density notch between the frozen ghost cell and the Lorentz-accelerated `f[1]`.

---

## 📖 3. Evolution Ledger (Fingerprint Log)

| Date | Step | Config | ML Mode | Loss | Result |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 2026-04-11 | 30 | 64x32, static-copy | Log-Space (5e-2) | 12.56 | **Success**: Stable boundaries, active tail learning. |
| 2026-04-11 | 30 | 64x32, copy-static | Log-Space (1e-2) | 0.47 | **Marginal**: ML scaling too low to see shock. |
| 2026-04-13 | 30 | 64x32, static-copy, NO ML | — | — | **Baseline audit**: n[1] notch confirmed (-3% at step 30). Root cause: 1-cell static BC. |
| 2026-04-13 | 30 | 64x64, static-copy, NO ML | — | — | **Fine-res baseline**: Mass drift -0.09%, energy stable. Clean shock. Saved to data/fine_data/. |
| 2026-04-13 | 30 | 64x32, static-copy, 2-cell buffer, NO ML | — | — | **Coarse baseline**: n[0]=n[1]=1.0000 retained. Drift -0.14%. Saved to data/coarse_data/. |
| 2026-04-13 | — | Architecture Refactor | — | — | **Senior Coder Audit**: 10 issues fixed. Split initialization, cleaned ML API, improved training hook. |
| 2026-04-13 | 30 | 64x32, Warm-Start | Online Train | 11.5 | **Success**: Training stabilized via Zero-Bias. Mass conservation excellent (+0.07%). Pipeline automated. |
| 2026-04-13 | — | Solver Refinement | — | — | **Physics Refine**: Field self-consistency added (mid-step Ohm's Law). Axis normalization in dashboards. |
| 2026-04-13 | — | handle Refactor | — | — | **Logic Refine**: Decoupled ADVECTION and ACCELERATION handles implemented. Support for dynamic kernels. |
| 2026-04-13 | 30 | Masked Sharpening | Online Train | 9.68 | **Success**: Adv_Clamp=5e-2, Kernel=3. Masking Buffer=2 restored stability. Drift -1.22%. Shock sharpened. |
| 2026-04-14 | 30 | 64x32, DeepONet | Online Train | 1.32 | **Success**: DeepONet stabilized via tanh-clamp. Loss normalization fixed. Training is now physically consistent. |
---

## 🗺️ 4. Roadmap & Next Steps

- `[x]` **Mass Conservation**: Interior drift -0.14% — **physical** (boundary outflow). Not ML-induced.
- `[x]` **Inflow BC Notch**: 2-cell buffer eliminates artificial boundary step. CNN no longer sees artifact.
- `[x]` **Online-Only Architecture**: All training via `run_maxwell.py` hook.
- `[x]` **First Online Training Run**: Stabilized training executed via Warm-Start.
- `[x]` **Decoupled Handles**: ML_ADVECTION_CLAMP and ML_ACCELERATION_CLAMP implemented.
- `[x]` **Boundary Masking**: Ghost cell exclusion (Buffer=2) implemented. Restored stability.
- `[x]` **Sharpening Experiment**: Proven effective with kernel=3 and clamp=5e-2. 
- `[x]` **DeepONet Integration**: Successfully implemented Branch-Trunk architecture ($D=32$).
- `[x]` **Stabilization Fix**: Added `tanh` safety bounds to DeepONet and fixed critical gradient overscaling.
- `[ ]` **Long-Term Drift**: Monitor if -1% drift accumulates over NT > 100 with dual-corrector active.
- `[ ]` **Hyperparameter Sweep**: Optimize Branch hidden dims and Trunk MLP depth for shock resolution.
