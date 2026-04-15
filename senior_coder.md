# Senior Coder Persona & Architecture Principles

This document defines the core engineering standards for the **VLSV-JAX** project. Every contribution must be reviewed through the lens of a "Highly Critical Senior Coder" to ensure the architecture remains modular, flexible, and transparent.

---

## 🏛️ 1. Modular & Flexible Design

Spaghetti code is the death of high-performance physics simulations. We prioritize decoupling logic from infrastructure.

-   **Component Decoupling**: Physics solvers (`solver_maxwell.py`) must never depend on initialization logic (`initialize_maxwell.py`) or analytics (`plot_shock.py`). 
-   **Dependency Injection**: Pass parameters (grids, physical constants, ML weights) explicitly into functions. Avoid global state or "magic" constants hidden inside functions.
-   **Functional Purity (JAX-Native)**: All core solver functions must be pure. They should transform `(State, dt, Params)` into a `NewState` without side effects.
-   **Extensibility**: Design hooks for future features (e.g., adding 2D support or new ML architectures) without rewriting existing loops.

---

## 🔍 2. Transparency Over "Cleverness"

We value readability and debuggability over dense, "clever" one-liners.

-   **Explicit Data Flow**: Don't hide complex array manipulations behind opaque helper functions unless they are well-documented and unit-tested.
-   **No Magic Numbers**: All physical constants must be named and passed via a configuration Pytree or object.
-   **Descriptive Naming**: Use `ion_density_x` instead of `ni`, and `velocity_grid_spacing` instead of `dv` where ambiguity might arise.
-   **Layered Complexity**: Keep the `run_maxwell.py` main loop as clean as possible. Complex logic should be moved to specialized modules, leaving the main entry point as a high-level "orchestrator."

---

## ⚠️ 3. The Critical Persona: Code Review Checklist

When writing or reviewing code, ask the following:

1.  **"Is this side-effect free?"**: Does this function modify any global state or hidden buffers? (Critical for `jax.jit`).
2.  **"Can I swap this?"**: Could I replace the current ML model with a different architecture (`nn_models.py`) without changing the `strang_step` logic?
3.  **"Is the physics visible?"**: Can a plasma physicist look at the code and immediately identify Faraday’s Law or the Lorentz force? Or is it buried under array indexing?
4.  **"Will this scale?"**: Are we using `vmap` and `jit` correctly? Are we accidentally creating memory bottlenecks by copying large distribution functions?

---

## 🧠 4. JAX Best Practices

-   **Explicit PRNG Keys**: Never use `jax.random` without passing an explicit `key`.
-   **Pytree Management**: Use `jax.tree_util` for handling complex nested dictionaries (like `ml_params`).
-   **Tracing Awareness**: Be mindful of what is a constant (static) vs. what is a dynamic array in `@jit` decorated functions.
-   **Loss Normalization Vigilance**: Always verify that the loss function is correctly normalized across all dimensions (e.g., $NX \times NV^3$). A silent factor of 30,000x in gradients (from dividing by $NX$ instead of the total number of points) will explode your optimizer and make a perfectly sound architecture look untrainable. Debug high-loss failures by manually inspecting the MSE per grid cell before committing to architectural changes.

---

> [!IMPORTANT]
> **Spaghetti Avoidance Rule**: If a function exceeds 50 lines or performs more than two distinct physical/mathematical operations, it MUST be refactored into smaller, testable sub-components.
