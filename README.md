# VLSV-JAX: Differentiable 1D-3V Hybrid Vlasov Plasma Solver

[![JAX](https://img.shields.io/badge/Accelerated_by-JAX-blue.svg)](https://github.com/google/jax)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**VLSV-JAX** is a high-performance, fully differentiable modular framework for plasma kinetic simulations. Built on top of **JAX**, it enables deeply fused, multi-step integrations optimized for GPU/TPU accelerators.

> [!TIP]
> This solver is designed specifically for **Physics-Informed Machine Learning (Physics-ML)** workflows, providing exact gradients through the entire simulation timeline for discovery and optimization.

---

## 🚀 Key Features

*   **Multi-Regime Physics**: Specialized modules for **Hybrid** (Ion kinetics) and **Electrostatic** (Electron kinetics).
*   **Modern Architecture**: Uses JAX-native **Pytree** state management for end-to-end differentiability.
*   **Numerical Precision**:
    *   **SLICE-3D**: Semi-Lagrangian scheme for conservative velocity rotations.
    *   **Ghost-Cell Boundaries**: 2nd-order Central differences enforced through synchronized 2-cell padding.
*   **Differentiable by Design**: Fully compatible with `jax.grad`, `jax.vmap`, and `jax.jit`.

---

## 🏗️ Project Architecture

The codebase is modularized to decouple physics logic from numerical infrastructure:

### 1. Hybrid Vlasov-Maxwell Solver
*   [`simulator.py`](file:///Users/ivanzait/Documents/Documents_LM4500/Codes/VLSV-JAX-2/Vlasov-Jax/simulator.py): Main entry point for simulations.
*   [`init_simulation.py`](file:///Users/ivanzait/Documents/Documents_LM4500/Codes/VLSV-JAX-2/Vlasov-Jax/init_simulation.py): Centralized setup for physical parameters and grid verification.
*   [`vlasov_solver.py`](file:///Users/ivanzait/Documents/Documents_LM4500/Codes/VLSV-JAX-2/Vlasov-Jax/vlasov_solver.py): Implementation of the high-level Strang-split orchestrator.
*   [`field_solver.py`](file:///Users/ivanzait/Documents/Documents_LM4500/Codes/VLSV-JAX-2/Vlasov-Jax/field_solver.py): Functional Maxwell/Faraday/Moment kernels.
*   [`boundary.py`](file:///Users/ivanzait/Documents/Documents_LM4500/Codes/VLSV-JAX-2/Vlasov-Jax/boundary.py): Ghost cell synchronization and BC enforcement.
*   [`state.py`](file:///Users/ivanzait/Documents/Documents_LM4500/Codes/VLSV-JAX-2/Vlasov-Jax/state.py): Core JAX Pytree data structure (`SimulationState`).
*   [`init_shock.py`](file:///Users/ivanzait/Documents/Documents_LM4500/Codes/VLSV-JAX-2/Vlasov-Jax/init_shock.py): Initial Condition generator for shock physics.

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
python3 simulator.py --config config_coarse
```
*Outputs (plots, diagnostics) are saved to the directories defined in your config.*

---

## 🌍 Roadmap
See [taskboard.md](file:///Users/ivanzait/Documents/Documents_LM4500/Codes/VLSV-JAX-2/Vlasov-Jax/taskboard.md) for current milestones and ML integration progress.
