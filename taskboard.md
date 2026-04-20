# VLSV-JAX Taskboard

This board tracks the evolution of the VLSV-JAX framework from a monolithic solver to a differentiable, ML-ready physicist's toolkit.

## 🏆 Phase 1 & 2: Modular Stabilization (Completed)
- [x] **State Formalization**: Transitioned to JAX-native `SimulationState` Pytrees.
- [x] **Modular Architecture**: Decoupled physics into `vlasov_solver`, `field_solver`, and `boundary` modules.
- [x] **Advanced Boundaries**: Implemented 2-cell ghost layers with Electric Field synchronization.
- [x] **Renaming Refactor**: Optimized filenames for clarity (`simulator.py`, `vlasov_solver.py`, `state.py`).

## 🧠 Phase 3: Neural Correction Engine (Completed)
- [x] **Dataset Engine**: Implemented `ml_dataset.py` with enriched EM fields, spatial gradients, and randomized 60/20/20 partitioning.
- [x] **Deep MLP Architecture**: Developed a high-capacity 3-layer `[256, 256, 128]` MLP in `ml_models.py`.
- [x] **Physics-Weighted Loss**: Implemented tail-enhancing velocity weighting and strong moment-consistency constraints ($\lambda=5.0$).
- [x] **Offline Benchmark**: achieved **61% reduction** in distribution log-residuals and **61% reduction** in velocity MAE on unseen test data.
- [x] **Verification Dashboard**: Specialized 4-row dashboard with log-scale Phase-Space visualization.

- [x] Phase 11: Downstream Normalization Pivot
    - [x] Update physical constants in `init_shock.py` (n_down=1, B_down=1)
    - [x] Purge legacy `data/` and `plots/` directories
- [ ] Phase 12: New Production Baseline
    - [ ] Run Fine-resolution ($64^3$) downstream-normalized simulation
    - [ ] Run Coarse-resolution ($32^3$) downstream-normalized simulation
    - [ ] Retrain Deep ResMLP on the new physically consistent dataset

## 🚀 Roadmap (Future Phases)
- [ ] **Dataset Engine**: Automated "Fine vs Coarse" downsampling utility for generating training targets.
- [ ] **Online Training Loop**: Integrated `optax` loop for "Solver-in-the-Loop" training.
- [ ] **Multi-Regime Diagnostics**: Advanced 1D phase-space visualization with ML-residual overlays.

---
*Last Updated: 2026-04-17*
