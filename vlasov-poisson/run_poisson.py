import os
import sys
import time
import jax.numpy as jnp

# Add root directory to path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from solver_poisson import FourierSolver, TVDSolver
from setup_two_stream import setup_two_stream
from plot_two_stream import plot_step
from config_poisson import *


if __name__ == "__main__":
    
    # ==========================================
    # SIMULATION PARAMETERS
    # ==========================================
    nx = 128
    nv = 64
    lx = 2 * jnp.pi / 0.5  # Approx 12.56
    lv = 6.0               # Wide enough for shock acceleration
    dt = 0.05
    nt = 250

    # Initialize
    os.makedirs("plots_two_stream", exist_ok=True)
    
    # Choose solver: TVDSolver (for shocks) or FourierSolver (spectrally exact for smooth flows)
    solver = TVDSolver(nx, nv, lx, lv)
    
    x, v, dx, dv, kx = solver.x, solver.v, solver.dx, solver.dv, solver.kx

    ### Create coordinate grids that broadcast appropriately to the 4D (x, vx, vy, vz)
    X = x[:, None, None, None]
    VX = v[None, :, None, None]
    VY = v[None, None, :, None]
    VZ = v[None, None, None, :]
    V_sq = VX**2 + VY**2 + VZ**2
    
    print(" Setup: Two-Stream Instability")
    f = setup_two_stream(X, VX, VY, VZ, v_drift=1.0, vth_beam=0.3, k_pert=0.3, pert_amp=0.05)
    
    Ekin_0 = jnp.sum(f * (V_sq / 2)) * (dx * dv**3)

    ### Main cycle
    
    # The initial condition is step 0:
    energy = jnp.sum(f * (V_sq / 2)) * (dx * dv**3)
    print(f"Step 0: Energy = {energy:.6f}")
    Ex = solver.get_electric_field(f)
    plot_step(0, x, v, f, Ex, lx, dx, dv, save_dir="plots_two_stream")
    
    t_start = time.time()
    for i in range(1, nt + 1):
        # Explicit single step using the chosen solver
        f = solver.strang_step(f, dt)
        
        # Calculate diagnostics and plot every 10 steps
        if i % 10 == 0:
            f.block_until_ready()
            t_end = time.time()
            dt_ten = t_end - t_start
            
            energy = jnp.sum(f * (V_sq / 2)) * (dx * dv**3)
            print(f"Step {i}: Energy = {energy:.6f} | Time (10 steps) = {dt_ten:.4f}s")
            
            Ex = solver.get_electric_field(f)
            plot_step(i, x, v, f, Ex, lx, dx, dv, save_dir="plots_two_stream")
            
            t_start = time.time()
    
    # ------------------------------------------
    # Memory Cleanup
    # ------------------------------------------
    import jax
    import gc
    
    f.block_until_ready()  # Ensure final computations complete before flushing
    jax.clear_caches()
    jax.clear_backends()
    
    del f, solver, Ex
    gc.collect()
    print("Simulation complete. Memory cleaned.")