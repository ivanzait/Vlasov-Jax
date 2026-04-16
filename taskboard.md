# VLSV-JAX Taskboard

This board tracks the evolution of the VLSV-JAX framework from a monolithic solver to a differentiable, ML-ready physicist's toolkit.

## 🏆 Current Achievements (Phase 1: State Formalization)
- [x] **State Formalization**: Transitioned to JAX-native `SimulationState` Pytrees.
- [x] **Modular Architecture**: Decoupled physics into `vlasov_solver`, `field_solver`, and `boundary` modules.
- [x] **Advanced Boundaries**: Implemented 2-cell ghost layers with Electric Field synchronization.
- [x] **Physical Realignment**: Corrected shockTube logic for $V_x < 0$ (Right-to-Left) flow.
- [x] **Dynamic Config**: Added support for profile-based execution (`--config`).

## 🛠️ In Progress (Phase 2: ML Correction Hooks)
- [ ] **Corrector Interface**: Define a `Corrector` protocol for injecting $\Delta f$ or $\Delta E$ into the Strang-split loop.
- [ ] **Differentiable Hook**: Ensure the solver loop remains fully differentiable through the ML injection point.
- [ ] **Residual Baseline**: Implement an identity corrector that returns zero-residuals for verification.

## 🚀 Roadmap (Future Phases)
- [ ] **Dataset Engine**: Automated "Fine vs Coarse" downsampling utility for generating training targets.
- [ ] **Online Training Loop**: Integrated `optax` loop for "Solver-in-the-Loop" training.
- [ ] **Multi-Regime Diagnostics**: Advanced 1D phase-space visualization with ML-residual overlays.

---
*Last Updated: 2026-04-16*
