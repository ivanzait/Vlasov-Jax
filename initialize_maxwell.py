import os
import jax.numpy as jnp
from solver_maxwell import HybridMaxwellSolver
from plot_shock import plot_step_maxwell, plot_initial_verification
from setup_shock import setup_shock_hybrid

# ==========================================
# HYBRID (ION/ALFVENIC) NORMALIZATION
# ==========================================
# Normalization for Maxwell solver (Darwin Hybrid Vlasov-Maxwell):
# magnetic field ~ B_0 = 1.0
# density ~ n_0 = 1.0
# distance ~ Ion skin depth (d_i = c / w_pi) = 1.0
# velocity ~ Alfven velocity (V_a = B_0 / sqrt(mu_0 m_i n_0)) = 1.0
# time ~ Inverse ion cyclotron frequency (Omega_ci^-1) = 1.0

q_i = 1.0         # Normalized ion charge
m_i = 1.0         # Normalized ion mass
mu_0 = 1.0        # Normalized vacuum permeability

def get_omega_ci(B_mag, m=m_i, q=q_i):
    """ Ion cyclotron frequency """
    return q * B_mag / m

def get_v_alfven(B_mag, n, m=m_i, mu=mu_0):
    """ Alfven velocity V_A = B / sqrt(mu_0 * rho) """
    rho = n * m
    return B_mag / jnp.sqrt(mu * rho)

def get_d_i(n, m=m_i, q=q_i, mu=mu_0):
    """ Ion skin depth d_i """
    return jnp.sqrt(m / (mu * n * q**2))





def initialize_simulation(nx=64, nv=32, nt=10, bc_x=('periodic', 'periodic'), bc_v='copy', plot_dir="plots_maxwell"):
    """
    Sets up the physical parameters, grid, and initial condition for a 1D shock simulation.
    """
    # 1. Base Physical Parameters
    B_0 = 1.0
    omega_ci = get_omega_ci(B_0)      
    t_ci = 1.0 / omega_ci             

    # 2. Simulation Geometry
    dx = 0.5
    lx = dx * nx
    dv = 0.6

    lv = nv * dv / 2    
    dt = 0.05 * t_ci


    # 3. Solver and Grids
    os.makedirs(plot_dir, exist_ok=True)
    solver = HybridMaxwellSolver(nx, nv, lx, lv, bc_x=bc_x, bc_v=bc_v)
    
    x, v, dx, dv = solver.x, solver.v, solver.dx, solver.dv
    X = x[:, None, None, None]
    VX = v[None, :, None, None]
    VY = v[None, None, :, None]
    VZ = v[None, None, None, :]
    V_sq = VX**2 + VY**2 + VZ**2
    
    # 4. Initialization
    print(f" Initializing Hybrid Shock Tube (nx={nx}, nv={nv}, BC_x={bc_x}, BC_v={bc_v})...")
    f, B_x, B_y, B_z = setup_shock_hybrid(X, VX, VY, VZ, lx, x)

    # 5. Static Boundary Capture
    # Capture initial values if 'static' BCs are requested
    solver.set_static_boundaries(f, B_y, B_z)

    # 6. Zero-step Verification
    print(" Verifying Initial State...")
    n_i, Vi_x, Vi_y, Vi_z, T_i = solver.get_moments(f)
    
    P_gas = n_i * T_i
    P_mag = (B_x**2 + B_y**2 + B_z**2) / 2.0
    P_tot = P_gas + P_mag
    
    p_mean = jnp.mean(P_tot)
    p_std = jnp.std(P_tot)
    variation_pct = 100 * p_std / p_mean
    print(f" Mean Total Pressure: {p_mean:.6f} +/- {p_std:.6f} ({variation_pct:.2f}% variation)")

    # 7. Resolution Sensitivity Warning (vth vs dv)
    # The moment integration n = sum(f)*dv^3 is sensitive to vth/dv.
    # Convert to float for reliable printing comparison
    vth_min_val = float(jnp.min(jnp.sqrt(jnp.maximum(T_i, 1e-4))))
    
    if dv > 0.5 * vth_min_val:
        v_alert = "CRITICAL" if dv > vth_min_val else "CAUTION"
        print(f"\n [!] RESOLUTION {v_alert}: dv ({dv:.4f}) is large relative to min(vth) ({vth_min_val:.4f}).")
        if v_alert == "CRITICAL":
            print(f"     Thermal velocity is under-resolved. Expect significant (>1%) pressure variation.")
        else:
            print(f"     Resolution is marginal. Consider decreasing dv for better pressure balance.")

    # 8. Grid Clipping Check (lv vs V_drift)
    # Ensure the velocity domain covers the distribution function tails
    vi_max = float(jnp.max(jnp.abs(Vi_x) + jnp.abs(Vi_y) + jnp.abs(Vi_z)))
    if lv < (vi_max + 3 * vth_min_val):
        print(f"\n [!] RANGE WARNING: lv ({lv:.2f}) may be too small for drift velocity ({vi_max:.2f}).")
        print(f"     Tails of the distribution function may be clipped, causing mass/energy loss.")

    plot_initial_verification(x, n_i, T_i, P_gas, P_mag, P_tot, save_dir=plot_dir)
    
    # Initial fields for plotting
    E_x, E_y, E_z, J_x, J_y, J_z = solver.get_fields(f, B_x, B_y, B_z)
    plot_step_maxwell(0, x, v, f, B_x, B_y, B_z, E_x, E_y, E_z, lx, dx, dv, save_dir=plot_dir)

    return {
        'f': f,
        'B': (B_x, B_y, B_z),
        'E': (E_x, E_y, E_z),
        'solver': solver,
        'dt': dt,
        'nt': nt,
        'bc_x': bc_x,
        'bc_v': bc_v,
        'grids': (x, v, dx, dv, V_sq),
        'lx': lx
    }
