import os
import time
import argparse
import importlib
import jax.numpy as jnp

from vlasov_solver import HybridMaxwellSolver
from plot_shock import plot_step_maxwell
from init_simulation import initialize_simulation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VLSV-JAX Hybrid Maxwell Simulation.")
    parser.add_argument("--config", type=str, default="config", help="Name of the configuration module (default: config)")
    args = parser.parse_args()

    # Dynamic import of the configuration
    try:
        cfg = importlib.import_module(args.config)
        print(f" Loaded configuration: {args.config}")
    except ImportError:
        print(f" Error: Configuration module '{args.config}' not found.")
        exit(1)

    # 1. Initialize Simulation (Parameters, Grids, Initial State, Verification)
    # Uses the dynamically loaded config
    # 1. Initialize Simulation (Parameters, Grids, Initial State, Verification)
    # Returns sim_data containing the SimulationState Pytree
    sim_data = initialize_simulation(cfg)
    
    state = sim_data['state']
    solver = sim_data['solver']
    
    dt = sim_data['params']['dt']
    nt = sim_data['params']['nt']
    
    x = sim_data['grid']['x']
    v = sim_data['grid']['v']
    dx = sim_data['grid']['dx']
    dv = sim_data['grid']['dv']
    lx = sim_data['grid']['lx']
    V_sq = sim_data['grid']['V_sq']


    ### Main cycle
    # Energy and diagnostics based on the SimulationState
    f_phys = solver.get_physical(state.f)
    Bx_phys = solver.get_physical(state.B_x)
    By_phys = solver.get_physical(state.B_y)
    Bz_phys = solver.get_physical(state.B_z)
    x_phys = solver.get_physical(x)
    
    energy = jnp.sum(f_phys * (V_sq / 2)) * (dx * dv**3)
    b_energy = jnp.sum(Bx_phys**2 + By_phys**2 + Bz_phys**2) * dx / 2.0
    print(f"Step 0: Kinetic Energy = {energy:.6f}, Magnetic Energy = {b_energy:.6f}")
    

    plot_step_maxwell(0, x_phys, v, f_phys, Bx_phys, By_phys, Bz_phys, 
                      solver.get_physical(state.E_x), solver.get_physical(state.E_y), solver.get_physical(state.E_z), 
                      lx, dx, dv, save_dir=cfg.PLOT_DIR)
    
    # 2. Data Persistence (Initial State - Physical Only)
    os.makedirs(cfg.DATA_DIR, exist_ok=True)
    save_path = os.path.join(cfg.DATA_DIR, "step_0000.npz")
    jnp.savez(save_path, f=f_phys, B_x=Bx_phys, B_y=By_phys, B_z=Bz_phys, 
              E_x=solver.get_physical(state.E_x), E_y=solver.get_physical(state.E_y), E_z=solver.get_physical(state.E_z), 
              x=x_phys, v=v, dx=dx, dv=dv)
    
    
    t_start = time.time()
    for i in range(1, nt + 1):
        # 3. Advance the solver using the unified state object
        state = solver.strang_step(state, dt)
        
        # 4. Apply Modular Boundary Conditions (Synchronize ghosts)
        state = solver.apply_bc_state(state)
        
        # Save diagnostics and plot based on config
        if i % cfg.PLOT_EVERY == 0:
            state.f.block_until_ready()
            t_end = time.time()
            dt_ten = t_end - t_start
            
            f_phys = solver.get_physical(state.f)
            Bx_phys = solver.get_physical(state.B_x)
            By_phys = solver.get_physical(state.B_y)
            Bz_phys = solver.get_physical(state.B_z)
            
            energy = jnp.sum(f_phys * (V_sq / 2)) * (dx * dv**3)
            b_energy = jnp.sum(Bx_phys**2 + By_phys**2 + Bz_phys**2) * dx / 2.0
            total_energy = energy + b_energy            
            print(f"Step {i}: Total Energy = {total_energy:.6f} | Time (10 steps) = {dt_ten:.4f}s")            
            
            plot_step_maxwell(i, x_phys, v, f_phys, Bx_phys, By_phys, Bz_phys, 
                              solver.get_physical(state.E_x), solver.get_physical(state.E_y), solver.get_physical(state.E_z), 
                              lx, dx, dv, save_dir=cfg.PLOT_DIR)
            
            t_start = time.time()
        
        # Data Persistence (Stride-based - Physical Only)
        if i % cfg.SAVE_STRIDE == 0:
            save_path = os.path.join(cfg.DATA_DIR, f"step_{i:04d}.npz")
            jnp.savez(save_path, f=solver.get_physical(state.f), 
                      B_x=solver.get_physical(state.B_x), B_y=solver.get_physical(state.B_y), B_z=solver.get_physical(state.B_z), 
                      E_x=solver.get_physical(state.E_x), E_y=solver.get_physical(state.E_y), E_z=solver.get_physical(state.E_z), 
                      x=x_phys, v=v, dx=dx, dv=dv)
    

    # ------------------------------------------
    # Memory Cleanup
    # ------------------------------------------
    import jax
    import gc
    
    state.f.block_until_ready()  # Ensure final computations complete before flushing
    jax.clear_caches()
    gc.collect()
    
    print("Simulation complete. Memory cleaned.")
