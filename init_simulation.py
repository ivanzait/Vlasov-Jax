import os
import jax.numpy as jnp
from vlasov_solver import HybridMaxwellSolver
from plot_shock import plot_step_maxwell, plot_initial_verification
from init_shock import setup_shock_hybrid
from state import SimulationState

# ==========================================
# HYBRID (ION/ALFVENIC) NORMALIZATION
# ==========================================

def get_omega_ci(B_mag, qi, mi):
    """ Ion cyclotron frequency """
    return qi * B_mag / mi

def get_v_alfven(B_mag, n, mi, mu0):
    """ Alfven velocity V_A = B / sqrt(mu_0 * rho) """
    rho = n * mi
    return B_mag / jnp.sqrt(mu0 * rho)

def get_d_i(n, mi, qi, mu0):
    """ Ion skin depth d_i """
    return jnp.sqrt(mi / (mu0 * n * qi**2))


def initialize_simulation(cfg):
    """
    Sets up the physical parameters, grid, and initial condition for a 1D shock simulation.
    Uses parameters from the passed configuration object (cfg).
    """
    # 1. Base Physical parameters
    nx, nv = cfg.NX, cfg.NV
    dx, dv = cfg.DX, cfg.DV
    nt, dt = cfg.NT, cfg.DT
    bc_x, bc_v = cfg.BC_X, cfg.BC_V
    plot_dir = cfg.PLOT_DIR
    
    qi, mi, mu0 = cfg.QI, cfg.MI, cfg.MU0
    
    lx = dx * nx
    lv = nv * dv / 2    

    # 2. Solver and Grids
    os.makedirs(plot_dir, exist_ok=True)
    solver = HybridMaxwellSolver(nx, nv, lx, lv, bc_x=bc_x, bc_v=bc_v, qi=qi, mi=mi)
    
    x, v, dx, dv = solver.x, solver.v, solver.dx, solver.dv
    
    # 3. Preparation of 4D Grids for Initial Conditions
    # We use broadcasting to create 4D representations of the grids
    # Shape: [nx_total, nv, nv, nv]
    VX_4d = v[None, :, None, None]
    VY_4d = v[None, None, :, None]
    VZ_4d = v[None, None, None, :]
    
    V_sq = VX_4d**2 + VY_4d**2 + VZ_4d**2
    
    # Generate ICs on the total grid (NX + padded ghosts)
    X_4d = x[:, None, None, None]
    f, B_x_1d, B_y_1d, B_z_1d = setup_shock_hybrid(X_4d, VX_4d, VY_4d, VZ_4d, lx, x)
    B_x, B_y, B_z = B_x_1d, B_y_1d, B_z_1d
    
    # Calculate initial E fields via Ohm's Law
    E_x, E_y, E_z, J_x, J_y, J_z = solver.get_fields(f, B_x, B_y, B_z)
    
    # Capture initial boundaries if using 'static' configuration
    if 'static' in solver.bc_x:
        solver.set_static_boundaries(f, B_y, B_z, E_x, E_y, E_z)

    # 4. Pack into SimulationState
    state = SimulationState(f, B_x, B_y, B_z, E_x, E_y, E_z)

    # 5. Zero-step Verification
    print(" Verifying Initial State...")
    n_i, Vi_x, Vi_y, Vi_z, T_i = solver.get_moments(state.f)
    
    P_gas = n_i * T_i
    P_mag = (state.B_x**2 + state.B_y**2 + state.B_z**2) / 2.0
    P_tot = P_gas + P_mag
    
    p_mean = jnp.mean(P_tot)
    p_std = jnp.std(P_tot)
    variation_pct = 100 * p_std / p_mean
    print(f" Mean Total Pressure: {p_mean:.6f} +/- {p_std:.6f} ({variation_pct:.2f}% variation)")

    # 6. Resolution Sensitivity Warning (vth vs dv)
    vth_min_val = float(jnp.min(jnp.sqrt(jnp.maximum(T_i, 1e-4))))
    
    if dv > 0.5 * vth_min_val:
        v_alert = "CRITICAL" if dv > vth_min_val else "CAUTION"
        print(f"\n [!] RESOLUTION {v_alert}: dv ({dv:.4f}) is large relative to min(vth) ({vth_min_val:.4f}).")
        if v_alert == "CRITICAL":
            print(f"     Thermal velocity is under-resolved. Expect significant (>1%) pressure variation.")
        else:
            print(f"     Resolution is marginal. Consider decreasing dv for better pressure balance.")

    # 7. Grid Clipping Check (lv vs V_drift)
    vi_max = float(jnp.max(jnp.abs(Vi_x) + jnp.abs(Vi_y) + jnp.abs(Vi_z)))
    if lv < (vi_max + 3 * vth_min_val):
        print(f"\n [!] RANGE WARNING: lv ({lv:.2f}) may be too small for drift velocity ({vi_max:.2f}).")
        print(f"     Tails of the distribution function may be clipped, causing mass/energy loss.")

    plot_initial_verification(x, n_i, T_i, P_gas, P_mag, P_tot, save_dir=plot_dir)
    
    # Initial fields for plotting
    plot_step_maxwell(0, x, v, state.f, state.B_x, state.B_y, state.B_z, state.E_x, state.E_y, state.E_z, lx, dx, dv, save_dir=plot_dir)

    return {
        'state': state,
        'grid': {
            'x': x,
            'v': v,
            'dx': dx,
            'dv': dv,
            'lx': lx,
            'lv': lv,
            'V_sq': V_sq
        },
        'params': {
            'nx': nx,
            'nv': nv,
            'nt': nt,
            'dt': dt,
            'bc_x': bc_x,
            'bc_v': bc_v,
        },
        'solver': solver
    }

