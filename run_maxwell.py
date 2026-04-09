import os
import time
import jax.numpy as jnp

from solver_maxwell import HybridMaxwellSolver
from plot_shock import plot_step_maxwell
from initialize_maxwell import initialize_simulation


if __name__ == "__main__":
    
    # 1. Initialize Simulation (Parameters, Grids, Initial State, Verification)
    # bc_x: (left_bc, right_bc) | bc_v: universal velocity BC
    # Options: 'periodic', 'copy', 'static'
    sim_data = initialize_simulation(nx=64, nv=64, nt=70, 
                                     bc_x=('copy', 'static'), 
                                     bc_v='copy')
    

    f = sim_data['f']
    B_x, B_y, B_z = sim_data['B']
    E_x, E_y, E_z = sim_data['E']
    solver = sim_data['solver']
    dt = sim_data['dt']
    nt = sim_data['nt']
    lx = sim_data['lx']
    x, v, dx, dv, V_sq = sim_data['grids']


    ### Main cycle
    
    # The initial condition is step 0:
    energy = jnp.sum(f * (V_sq / 2)) * (dx * dv**3)
    b_energy = jnp.sum(B_x**2 + B_y**2 + B_z**2) * dx / 2.0
    print(f"Step 0: Kinetic Energy = {energy:.6f}, Magnetic Energy = {b_energy:.6f}")
    

    E_x, E_y, E_z, J_x, J_y, J_z = solver.get_fields(f, B_x, B_y, B_z)
    plot_step_maxwell(0, x, v, f, B_x, B_y, B_z, E_x, E_y, E_z, lx, dx, dv, save_dir="plots_maxwell")
    
    
    t_start = time.time()
    for i in range(1, nt + 1):
        # 1. Advance the solver
        f, B_x, B_y, B_z, E_x, E_y, E_z = solver.strang_step(f, B_x, B_y, B_z, dt)
        
        # 2. Apply Modular Boundary Conditions (Asymmetric Spatial & Velocity 'Copy')
        f, B_y, B_z = solver.apply_bc(f, B_y, B_z)
        
        # Calculate diagnostics and plot every 10 steps
        if i % 10 == 0:
            f.block_until_ready() # Ensure JAX operations are done before timing
            t_end = time.time()
            dt_ten = t_end - t_start
            
            energy = jnp.sum(f * (V_sq / 2)) * (dx * dv**3)
            b_energy = jnp.sum(B_x**2 + B_y**2 + B_z**2) * dx / 2.0
            total_energy = energy + b_energy            
            print(f"Step {i}: Total Energy = {total_energy:.6f} | Time (10 steps) = {dt_ten:.4f}s")            
            plot_step_maxwell(i, x, v, f, B_x, B_y, B_z, E_x, E_y, E_z, lx, dx, dv, save_dir="plots_maxwell")
            
            t_start = time.time()
    

    # ------------------------------------------
    # Memory Cleanup
    # ------------------------------------------
    import jax
    import gc
    
    f.block_until_ready()  # Ensure final computations complete before flushing
    jax.clear_caches()
    
    del f, solver, E_x, B_x
    gc.collect()
    print("Simulation complete. Memory cleaned.")
