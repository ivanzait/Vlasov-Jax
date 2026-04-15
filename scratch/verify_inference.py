import os
import sys
import jax
import jax.numpy as jnp
import numpy as np

# Add project root to path
sys.path.append(os.getcwd())

import config
from initialize_maxwell import initialize_simulation
from nn_models import init_network_params, apply_advection_correction, apply_acceleration_correction

def verify_inference():
    print("=== Inference Verification ===")
    
    # 1. Initialize simulation twice (to get identical starting points)
    sim_data_baseline = initialize_simulation(nt=2, plot_dir="plots_test")
    sim_data_ml = initialize_simulation(nt=2, plot_dir="plots_test")
    
    solver = sim_data_baseline['solver']
    dt = sim_data_baseline['dt']
    
    # Baseline run (No ML)
    f_b = sim_data_baseline['f']
    Bx_b, By_b, Bz_b = sim_data_baseline['B']
    for i in range(1, 3):
        f_b, Bx_b, By_b, Bz_b, E_x, E_y, E_z = solver.strang_step(
            f_b, Bx_b, By_b, Bz_b, dt, ml_params=None, adv_func=None, acc_func=None)
        f_b, By_b, Bz_b = solver.apply_bc(f_b, By_b, Bz_b)
    
    # ML run (Inference with random weights)
    f_m = sim_data_ml['f']
    Bx_m, By_m, Bz_m = sim_data_ml['B']
    key = jax.random.PRNGKey(42)
    ml_params = init_network_params(key, config.NX, config.NV)
    
    for i in range(1, 3):
        f_m, Bx_m, By_m, Bz_m, E_x, E_y, E_z = solver.strang_step(
            f_m, Bx_m, By_m, Bz_m, dt, 
            ml_params=ml_params, 
            adv_func=apply_advection_correction, 
            acc_func=apply_acceleration_correction
        )
        f_m, By_m, Bz_m = solver.apply_bc(f_m, By_m, Bz_m)
    
    # Calculate difference
    diff = jnp.abs(f_b - f_m)
    max_diff = float(jnp.max(diff))
    mean_diff = float(jnp.mean(diff))
    
    print(f"Max difference in f after 2 steps: {max_diff:.2e}")
    print(f"Mean difference in f after 2 steps: {mean_diff:.2e}")
    
    if max_diff > 0:
        print("SUCCESS: Inference Path is ACTIVE and modifying the distribution function.")
    else:
        print("FAILURE: ML run produced identical results to baseline.")

if __name__ == "__main__":
    verify_inference()
