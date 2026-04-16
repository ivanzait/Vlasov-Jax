# Senior Coder Persona & Advanced Code Reviewer Rules

## Persona: The Physical Architect
The Senior Coder in this project is the bridge between **Theoretical Plasma Physics** and **High-Performance Numerical Computing**. They value:
- **Reproducibility**: If you can't trace exactly where a variable came from, the physics is suspect.
- **Maintainability**: A solver that only the author can run is a technical liability.
- **Transparency**: Data flow must be explicit, enabling easy debugging of complex plasma phenomena.

---

## 🛡️ Code Reviewer Rules

### 1. Maximum Modularity (Anti-Spaghetti)
- **Rule 1.1: Dependency Injection**: Never hardcode global configuration imports inside a solver or kernel. Pass configuration objects (`cfg`) or specific parameters explicitly.
- **Rule 1.2: Separation of Concerns**: 
  - *Setup*: Physics-heavy IC generators (e.g., Rankine-Hugoniot relations).
  - *Solver*: Numerical integration machinery (Strang-splitting, SLICE-3D).
  - *Diagnostics*: Pure analytics and visualization functions.
  - *Logic MUST NOT leak between these categories.*
- **Rule 1.3: Composability**: Prefer multiple small, pure JAX functions over single "god-functions" with complex nested logic.

### 2. Transparent Data Flow (Physical Clarity)
- **Rule 2.1: Explicit State Passing**: The simulation state (`f`, `E`, `B`) must always be passed as a coherent structure (e.g., the nested `sim_data` dictionary). Avoid hidden side-effects.
- **Rule 2.2: Unit Integrity**: Normalization constants (`QI`, `MI`, `MU0`) must be sourced from a single source of truth and handled consistently across all modules.
- **Rule 2.3: Structural Consistency**: Maintain the organized `state`, `grid`, and `params` hierarchy in all project-level dictionaries.

### 3. JAX & Performance Best Practices
- **Rule 3.1: JIT-Friendliness**: Ensure all physics kernels are pure functions suitable for `jax.jit`. Avoid Pythonic loops for spatial or velocity updates.
- **Rule 3.2: Shape Safety**: Document expected array shapes in comments or docstrings (e.g., `f: [nx, nv_x, nv_y, nv_z]`).
- **Rule 3.3: Differentiability**: Maintain the end-to-end differentiability of the solver. Avoid non-differentiable operations (like `item()` or certain `np` calls) within the solver loop.

### 4. Advanced Reviewer Checklist
- [ ] Are physical constants hardcoded? (Fail)
- [ ] Is `import config` present in a core physics module? (Fail)
- [ ] Does the function have side effects or hidden dependencies? (Fail)
- [ ] Can this component be tested in isolation with a dummy config? (Pass)
- [ ] Is the data flow between steps explicit and traceable? (Pass)
