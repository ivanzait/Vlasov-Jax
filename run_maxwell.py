"""
run_maxwell.py — Main Entry Point for the VLSV-JAX Hybrid Solver

Controls the simulation loop, online ML training hook, and I/O.
All configuration lives in config.py — no edits needed here for routine runs.

Usage:
    python run_maxwell.py
"""

import os
import gc
import jax
import time
import jax.numpy as jnp

from solver_maxwell import HybridMaxwellSolver
from plot_shock import plot_step_maxwell
from initialize_maxwell import initialize_simulation
from nn_models import (init_network_params, apply_advection_correction,
                       apply_acceleration_correction, apply_deeponet_correction)
from data_io import save_snapshot, load_model_weights, save_model_weights
from train_corrector import maybe_train_step, init_adam_state
import config

if __name__ == "__main__":

    # -------------------------------------------------------
    # 0. Initialize simulation (grid, IC, BC, verification)
    # -------------------------------------------------------
    sim_data = initialize_simulation()

    USE_ML     = config.USE_ML
    ML_MODE    = config.ML_MODE
    MODEL_PATH = config.MODEL_PATH

    f                    = sim_data['f']
    B_x, B_y, B_z       = sim_data['B']
    E_x, E_y, E_z       = sim_data['E']
    solver               = sim_data['solver']
    dt                   = sim_data['dt']
    nt                   = sim_data['nt']
    lx                   = sim_data['lx']
    x, v, dx, dv, V_sq  = sim_data['grids']
    nx, nv               = len(x), len(v)

    # Self-describing snapshot metadata
    grid_info = {'nx': nx, 'nv': nv, 'dx': dx, 'dv': dv,
                 'lx': lx, 'lv': solver.lv}

    # -------------------------------------------------------
    # 1. Machine Learning setup
    # -------------------------------------------------------
    key       = jax.random.PRNGKey(42)
    ml_params = None
    ml_adv    = None
    ml_acc    = None
    m_adam    = None
    v_adam    = None
    t_adam    = 0

    if USE_ML:
        print(f" [ML] Initializing correction layers...")
        acc_mode = getattr(config, 'ML_ACC_MODE', 'mlp')
        ml_params = init_network_params(
            key, nx, nv,
            kernel_size=config.ML_ADVECTION_KERNEL,
            accel_hidden=config.ML_ACCEL_HIDDEN,
            std=1e-4,
            acc_mode=acc_mode,
            deeponet_d=getattr(config, 'ML_DEEPONET_D', 32),
            deeponet_trunk_hidden=getattr(config, 'ML_DEEPONET_TRUNK_HIDDEN', 64),
            deeponet_branch_hidden=getattr(config, 'ML_DEEPONET_BRANCH_HIDDEN', 32),
            deeponet_kernel=getattr(config, 'ML_DEEPONET_KERNEL', 5),
        )
        # Choose advection corrector variant: cell-wise multiplicative or face flux correction
        if getattr(config, 'ML_ADVECT_FLUX_CORRECTION', False):
            from nn_models import apply_advection_flux_correction
            ml_adv = apply_advection_flux_correction if getattr(config, 'ML_USE_CNN', True) else None
        else:
            ml_adv = apply_advection_correction if getattr(config, 'ML_USE_CNN', True) else None
        # Choose acceleration corrector variant: legacy MLP or DeepONet
        if getattr(config, 'ML_USE_MLP', True):
            if acc_mode == 'deeponet':
                from functools import partial
                ml_acc = partial(apply_deeponet_correction, v_grid=solver.v)
                print(f" [ML] Acceleration corrector: DeepONet (D={getattr(config, 'ML_DEEPONET_D', 32)})")
            else:
                ml_acc = apply_acceleration_correction
                print(f" [ML] Acceleration corrector: MLP (hidden={config.ML_ACCEL_HIDDEN})")
        else:
            ml_acc = None

        if ML_MODE == 'inference':
            # Prefer explicit MODEL_PATH, fall back to config ML_WEIGHTS_PATH
            if os.path.exists(MODEL_PATH):
                print(f" [ML] Loading pre-trained weights from {MODEL_PATH}...")
                ml_params = load_model_weights(MODEL_PATH, ml_params)
            elif os.path.exists(config.ML_WEIGHTS_PATH):
                print(f" [ML] Loading pre-trained weights from {config.ML_WEIGHTS_PATH}...")
                ml_params = load_model_weights(config.ML_WEIGHTS_PATH, ml_params)
            else:
                print(f" [ML] Warning: no pretrained weights found. Using initialized weights.")
        elif ML_MODE in ['training', 'train']:
            print(f" [ML] Initializing Online Training (Adam optimizer)...")
            m_adam, v_adam = init_adam_state(ml_params)
    else:
        print(" [ML] Mode: Bypassed (USE_ML=False)")

    # -------------------------------------------------------
    # 2. Snapshot I/O config
    # -------------------------------------------------------
    save_every = config.SAVE_EVERY
    save_dir   = config.SAVE_DIR
    os.makedirs(save_dir, exist_ok=True)

    # -------------------------------------------------------
    # 3. Step 0 diagnostics
    # -------------------------------------------------------
    energy   = jnp.sum(f * (V_sq / 2)) * (dx * dv**3)
    b_energy = jnp.sum(B_x**2 + B_y**2 + B_z**2) * dx / 2.0
    print(f"Step 0: Kinetic Energy = {energy:.6f}, Magnetic Energy = {b_energy:.6f}")

    # Interior-only mass baseline (excludes 2-cell inflow buffer + copy outflow)
    N0_interior = float(jnp.sum(f[2:-1]) * dx * dv**3)
    print(f"Step 0: Interior Mass (N_interior) = {N0_interior:.8f} [baseline]")

    # -------------------------------------------------------
    # 4. Main simulation loop
    # -------------------------------------------------------
    t_start = time.time()
    for i in range(1, nt + 1):

        # Advance one Strang-split step with optional ML corrections
        f, B_x, B_y, B_z, E_x, E_y, E_z = solver.strang_step(
            f, B_x, B_y, B_z, dt,
            ml_params=ml_params,
            adv_func=ml_adv,
            acc_func=ml_acc,
            adv_flux=getattr(config, 'ML_ADVECT_FLUX_CORRECTION', False),
            adv_clamp=config.ML_ADVECTION_CLAMP,
            acc_clamp=config.ML_ACCELERATION_CLAMP,
            boundary_buffer=config.ML_BOUNDARY_BUFFER
        )
        f, B_y, B_z = solver.apply_bc(f, B_y, B_z)

        # Diagnostics every 10 steps
        if i % 10 == 0:
            f.block_until_ready()
            t_end        = time.time()
            energy       = jnp.sum(f * (V_sq / 2)) * (dx * dv**3)
            b_energy     = jnp.sum(B_x**2 + B_y**2 + B_z**2) * dx / 2.0
            total_energy = energy + b_energy
            N_interior   = float(jnp.sum(f[2:-1]) * dx * dv**3)
            mass_drift   = 100.0 * (N_interior - N0_interior) / N0_interior

            print(f"Step {i:04d}: Total Energy = {total_energy:.6f} | "
                  f"N_interior = {N_interior:.6f} | "
                  f"Mass Drift = {mass_drift:+.4f}% | "
                  f"Time (10 steps) = {t_end - t_start:.4f}s")
            plot_step_maxwell(i, x, v, f, B_x, B_y, B_z, E_x, E_y, E_z,
                              lx, dx, dv, save_dir=config.PLOT_DIR)   # Fix #7
            t_start = time.time()

        # Snapshot save + optional online training update
        if i % save_every == 0:
            save_snapshot(i, f, (B_x, B_y, B_z), (E_x, E_y, E_z),
                          grid_info=grid_info, params=ml_params, save_dir=save_dir)

            if USE_ML and ML_MODE in ['training', 'train']:
                ml_params, m_adam, v_adam, t_adam = maybe_train_step(
                    step=i, f=f, B=(B_x, B_y, B_z), E=(E_x, E_y, E_z),
                    ml_params=ml_params, m_adam=m_adam, v_adam=v_adam, t_adam=t_adam,
                    fine_dir="data/fine_data", nx=nx, nv=nv, dt=dt, nt=nt,
                    lr_adv=config.ML_LR_CNN,
                    lr_acc=config.ML_LR_MLP,
                    adv_clamp=config.ML_ADVECTION_CLAMP,
                        acc_clamp=config.ML_ACCELERATION_CLAMP,
                        boundary_buffer=config.ML_BOUNDARY_BUFFER,
                        dx=dx, v=v
                )

    # -------------------------------------------------------
    # 5. Post-run: save weights and clean up
    # -------------------------------------------------------
    if USE_ML and ML_MODE in ['training', 'train']:
        save_model_weights(ml_params, MODEL_PATH)
        print(f" [ML] Online training complete. Weights saved to {MODEL_PATH}")

    f.block_until_ready()
    jax.clear_caches()
    del f, solver, E_x, B_x   # Fix #8: removed duplicate import jax / import gc
    gc.collect()
    print("Simulation complete. Memory cleaned.")
