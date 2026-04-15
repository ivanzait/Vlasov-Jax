"""
initialize_maxwell.py — Grid Construction, IC Setup, and Pre-run Verification

Normalization (Hybrid Vlasov-Maxwell, Darwin Approximation):
    Length    — ion skin depth        d_i  = c / ω_pi  = 1
    Velocity  — Alfvén velocity       V_A  = B_0 / √(μ₀ m_i n_0) = 1
    Time      — inv. ion cyclotron    Ω_ci⁻¹ = m_i / (q_i B_0) = 1
    B-field   — background field      B_0 = 1
    Density   — upstream density      n_0 = 1
    q/m       — normalized to 1       (q_i = m_i = μ₀ = 1)
"""

import os
import jax.numpy as jnp

from solver_maxwell import HybridMaxwellSolver
from plot_shock import plot_step_maxwell, plot_initial_verification
from setup_shock import setup_shock_hybrid
import config


# ==========================================
# 1. GRID PARAMETER BUILDER (Pure)
# ==========================================

def build_grid_params(nx, nv, dx, dv, B_0, m_i, q_i, mu_0):
    """
    Derives all geometric and temporal parameters from resolution and physics.
    Pure function: no side effects, fully testable.

    Returns a dict with keys: lx, lv, dt, omega_ci
    """
    lx = dx * nx
    lv = nv * dv / 2
    omega_ci = q_i * B_0 / m_i          # Ion cyclotron frequency (normalized = 1)
    dt = 0.05 / omega_ci                  # CFL timestep: 0.05 Ω_ci⁻¹
    return {'lx': lx, 'lv': lv, 'dt': dt, 'omega_ci': omega_ci}


# ==========================================
# 2. INITIAL STATE VERIFIER
# ==========================================

def verify_initial_state(solver, f, B_x, B_y, B_z):
    """
    Checks total pressure balance and velocity resolution adequacy.
    Prints human-readable warnings. Returns moment arrays for plotting.

    Returns:
        dict with keys: n_i, T_i, P_gas, P_mag, P_tot
    """
    n_i, Vi_x, Vi_y, Vi_z, T_i = solver.get_moments(f)

    P_gas = n_i * T_i
    P_mag = (B_x**2 + B_y**2 + B_z**2) / 2.0
    P_tot = P_gas + P_mag

    p_mean = jnp.mean(P_tot)
    p_std  = jnp.std(P_tot)
    print(f" Mean Total Pressure: {p_mean:.6f} +/- {p_std:.6f} ({100*p_std/p_mean:.2f}% variation)")

    # Resolution check: dv vs thermal velocity
    vth_min = float(jnp.min(jnp.sqrt(jnp.maximum(T_i, 1e-4))))
    dv = solver.dv
    if dv > 0.5 * vth_min:
        alert = "CRITICAL" if dv > vth_min else "CAUTION"
        print(f"\n [!] RESOLUTION {alert}: dv ({dv:.4f}) is large relative to min(vth) ({vth_min:.4f}).")
        if alert == "CRITICAL":
            print(f"     Thermal velocity under-resolved. Expect significant (>1%) pressure variation.")
        else:
            print(f"     Resolution is marginal. Consider decreasing dv for better pressure balance.")

    # Velocity domain check: lv must cover drift + 3 sigma
    vi_max = float(jnp.max(jnp.abs(Vi_x) + jnp.abs(Vi_y) + jnp.abs(Vi_z)))
    if solver.lv < (vi_max + 3 * vth_min):
        print(f"\n [!] RANGE WARNING: lv ({solver.lv:.2f}) may clip distribution tails "
              f"(needs ≥ {vi_max + 3*vth_min:.2f}).")

    return {'n_i': n_i, 'T_i': T_i, 'P_gas': P_gas, 'P_mag': P_mag, 'P_tot': P_tot}


# ==========================================
# 3. SIMULATION INITIALIZER (Orchestrator)
# ==========================================

def initialize_simulation(nx=None, nv=None, nt=None, dx=None, dv=None,
                           plot_dir=None, use_ml=None):
    """
    Orchestrates the full pre-simulation setup:
      1. Merge caller overrides with config defaults
      2. Derive grid/timing parameters via build_grid_params()
      3. Construct solver and velocity-space tensors
      4. Generate Rankine-Hugoniot initial condition
      5. Capture static BC buffers
      6. Verify IC via verify_initial_state() and plot

    Returns a dict consumed by run_maxwell.py.
    """
    # 0. Config defaults (caller can override any parameter)
    nx      = nx       or config.NX
    nv      = nv       or config.NV
    nt      = nt       or config.NT
    dx      = dx       or config.DX
    dv      = dv       or config.DV
    plot_dir = plot_dir or config.PLOT_DIR
    use_ml  = use_ml if use_ml is not None else config.USE_ML

    # 1. Grid and timing parameters
    gp = build_grid_params(nx, nv, dx, dv,
                           config.B_BACKGROUND, config.MI, config.QI, config.MU_0)

    # 2. Solver construction
    os.makedirs(plot_dir, exist_ok=True)
    solver = HybridMaxwellSolver(nx, nv, gp['lx'], gp['lv'],
                                 bc_x=config.BC_X, bc_v=config.BC_V)
    x, v   = solver.x, solver.v
    dx, dv = solver.dx, solver.dv     # use solver's computed values (may differ slightly)

    # 3. Velocity-space coordinate tensors (broadcast-ready)
    VX   = v[None, :, None, None]
    VY   = v[None, None, :, None]
    VZ   = v[None, None, None, :]
    V_sq = VX**2 + VY**2 + VZ**2

    # 4. Initial condition (Rankine-Hugoniot shock)
    ml_status = "ENABLED" if use_ml else "BYPASSED"
    print(f" Initializing Hybrid Shock Tube (nx={nx}, nv={nv}, ML={ml_status})...")
    print(f" BC_x={config.BC_X}, BC_v={config.BC_V}")
    X = x[:, None, None, None]
    f, B_x, B_y, B_z = setup_shock_hybrid(X, VX, VY, VZ, gp['lx'], x)

    # 5. Capture initial 2-cell static BC buffers
    solver.set_static_boundaries(f, B_y, B_z)

    # 6. Verification and diagnostics
    print(" Verifying Initial State...")
    moments = verify_initial_state(solver, f, B_x, B_y, B_z)
    plot_initial_verification(x, moments['n_i'], moments['T_i'],
                              moments['P_gas'], moments['P_mag'], moments['P_tot'],
                              save_dir=plot_dir)

    E_x, E_y, E_z, _, _, _ = solver.get_fields(f, B_x, B_y, B_z)
    plot_step_maxwell(0, x, v, f, B_x, B_y, B_z, E_x, E_y, E_z,
                      gp['lx'], dx, dv, save_dir=plot_dir)

    return {
        'f':      f,
        'B':      (B_x, B_y, B_z),
        'E':      (E_x, E_y, E_z),
        'solver': solver,
        'dt':     gp['dt'],
        'nt':     nt,
        'bc_x':   config.BC_X,
        'bc_v':   config.BC_V,
        'grids':  (x, v, dx, dv, V_sq),
        'lx':     gp['lx'],
    }
