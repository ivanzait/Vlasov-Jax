# VLSV-JAX: Differentiable 1D-3V Hybrid Vlasov Plasma Solver

[![JAX](https://img.shields.io/badge/Accelerated_by-JAX-blue.svg)](https://github.com/google/jax)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**VLSV-JAX** is a high-performance, fully differentiable modular framework for plasma kinetic simulations. Built on top of **JAX**, it enables deeply fused, multi-step integrations optimized for GPU/TPU accelerators.

> [!TIP]
> This solver is designed specifically for **Physics-Informed Machine Learning (Physics-ML)** workflows, providing exact gradients through the entire simulation timeline for discovery and optimization.

---

## 🚀 Key Features

*   **Multi-Regime Physics**: Specialized modules for **Hybrid** (Ion kinetics) and **Electrostatic** (Electron kinetics).
*   **Neural Correction Engine**: Built-in specialized MLP hooks for learning the numerical residuals between coarse and fine resolution simulations.
*   **Modern Architecture**: Uses JAX-native **Pytree** state management for end-to-end differentiability.
*   **Numerical Precision**:
    *   **SLICE-3D**: Semi-Lagrangian scheme for conservative velocity rotations.
    *   **Ghost-Cell Boundaries**: 2nd-order Central differences enforced through synchronized 2-cell padding.
*   **Offline Physics-ML Pipeline**: Integrated dataset handling for enriched features ($f, E, B$ fields + spatial gradients).
*   **Differentiable by Design**: Fully compatible with `jax.grad`, `jax.vmap`, and `jax.jit`.

---

## 🏗️ Project Architecture

The codebase is modularized to decouple physics logic from numerical infrastructure:

### 1. Hybrid Vlasov-Maxwell Solver (`src/solver/`)
*   [`simulator.py`](file:///Users/ivanzait/Documents/Documents_LM4500/Codes/VLSV-JAX-2/Vlasov-Jax/src/solver/simulator.py): Main module entry point.
*   [`vlasov_solver.py`](file:///Users/ivanzait/Documents/Documents_LM4500/Codes/VLSV-JAX-2/Vlasov-Jax/src/solver/vlasov_solver.py): High-level Strang-split orchestrator.
*   [`init_simulation.py`](file:///Users/ivanzait/Documents/Documents_LM4500/Codes/VLSV-JAX-2/Vlasov-Jax/src/solver/init_simulation.py): Parameters and grid verification.
*   [`field_solver.py`](file:///Users/ivanzait/Documents/Documents_LM4500/Codes/VLSV-JAX-2/Vlasov-Jax/src/solver/field_solver.py): Functional Maxwell/Faraday kernels.
*   [`boundary.py`](file:///Users/ivanzait/Documents/Documents_LM4500/Codes/VLSV-JAX-2/Vlasov-Jax/src/solver/boundary.py): Ghost cell synchronization.
*   [`state.py`](file:///Users/ivanzait/Documents/Documents_LM4500/Codes/VLSV-JAX-2/Vlasov-Jax/src/solver/state.py): JAX Pytree data structure (`SimulationState`).

### 2. Neural Correction (`src/ml/`)
*   [`ml_dataset.py`](file:///Users/ivanzait/Documents/Documents_LM4500/Codes/VLSV-JAX-2/Vlasov-Jax/src/ml/ml_dataset.py): Purified, metadata-aware dataset engine.
*   [`ml_models.py`](file:///Users/ivanzait/Documents/Documents_LM4500/Codes/VLSV-JAX-2/Vlasov-Jax/src/ml/ml_models.py): Dynamized 3-layer MLP architecture.
*   [`train_offline.py`](file:///Users/ivanzait/Documents/Documents_LM4500/Codes/VLSV-JAX-2/Vlasov-Jax/src/ml/train_offline.py): Physics-weighted training pipeline.
*   [`ml_quantification.py`](file:///Users/ivanzait/Documents/Documents_LM4500/Codes/VLSV-JAX-2/Vlasov-Jax/src/ml/ml_quantification.py): Performance scorecard.

### 3. Setup & Utils (`setup/`)
*   [`init_shock.py`](file:///Users/ivanzait/Documents/Documents_LM4500/Codes/VLSV-JAX-2/Vlasov-Jax/setup/init_shock.py): Shock tube IC generator.
*   [`plot_shock.py`](file:///Users/ivanzait/Documents/Documents_LM4500/Codes/VLSV-JAX-2/Vlasov-Jax/setup/plot_shock.py): Spatial visualization suite.

---

## 🔄 Simulation Cycle (Darwin Hybrid)

VLSV-JAX uses a **Strang-Splitting** sequence to maintain 2nd-order accuracy in time:

```mermaid
graph TD
    A[Start Step t] --> B["Advect X (dt/2)"]
    B --> C["Calculate Fields (E, J) via field_solver"]
    C --> D["Accelerate V (dt) via vlasov_solver"]
    D --> E["Advance Magnetic Field (dt) via field_solver"]
    E --> F["Advect X (dt/2)"]
    F --> G["Synchronize Ghosts (apply_bc)"]
    G --> H["End Step t+dt"]
    H --> A
```

---

## 📊 Data Format & Persistence

Simulation data is stored in **`.npz`** format for broad compatibility. Each file contains:
*   **`f`**: 4D Distribution Function $[NX_{total}, NV, NV, NV]$.
*   **`B_x, B_y, B_z`**: 1D Magnetic field components.
*   **`E_x, E_y, E_z`**: 1D Electric field components.
*   **`x, v`**: Physical grid coordinates.

> [!NOTE]
> Saved data strictly contains only the **physical domain** (ghost cells are automatically sliced out during persistence).

---

## 📖 Quick Start

### Running a Hybrid Shock Simulation
```bash
./run_simulation.sh config_fine
```

### Performing Offline Training (Neural Correction)
```bash
./run_training.sh
```

### Verifying ML Performance
```bash
./run_verification.sh
```
*Trained weights are persistent in `data/ml_weights/`.*

---

## 📈 ML Correction Benchmarks

The current **Deep High-Capacity Model** achieved the following improvements on unseen shock test data:
*   **Log-Distribution Fidelity**: **61.3%** error reduction.
*   **Bulk Velocity ($V_x$)**: **61.4%** error reduction.
*   **Physical Consistency**: Maintained density conservation within 3.1% of baseline.

---

## 🌍 Roadmap
See [taskboard.md](file:///Users/ivanzait/Documents/Documents_LM4500/Codes/VLSV-JAX-2/Vlasov-Jax/taskboard.md) for current milestones and ML integration progress.
