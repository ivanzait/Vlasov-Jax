"""
verify_training.py — 3-Column Verification Dashboard

Compares the evolved state of the simulation across three regimes:
  1. FINE GROUND TRUTH  (data/fine_data/)
  2. COARSE BASELINE     (data/coarse_data/)
  3. ML-CORRECTED        (data/corrected_data/)

Features:
- Global Axis Scanning: Ensures y-limits for Density and Vx are consistent across all frames.
- Field Alignment: Downsamples fine-resolution fields for visual comparison.
"""

import os
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from data_io import load_snapshot
from train_corrector import coarsen_data
import config

def extract_diagnostics(f, v_grid, dv):
    """Calculates moments and phase-space marginals for plotting."""
    n = jnp.sum(f, axis=(1, 2, 3)) * (dv**3)
    n_safe = jnp.maximum(n, 1e-6)
    
    vx_grid = v_grid[None, :, None, None]
    Vx = jnp.sum(f * vx_grid, axis=(1, 2, 3)) * (dv**3) / n_safe
    
    f_vx = jnp.sum(f, axis=(2, 3)) * (dv**2)
    
    return {'n': n, 'Vx': Vx, 'f_vx': f_vx}

def scan_global_limits(shot_ids):
    """
    Scans all relevant snapshots across all regimes to find global y-limits 
    for Density and bulk velocity Vx.
    """
    print(f" [Scanning] Finding global axis limits for snapshots {shot_ids}...")
    n_min, n_max = float('inf'), float('-inf')
    v_min, v_max = float('inf'), float('-inf')
    
    regimes = ["data/fine_data", "data/coarse_data", "data/corrected_data"]
    dv = config.DV
    v_grid = jnp.linspace(-config.LV, config.LV, config.NV)
    
    for sid in shot_ids:
        for regime in regimes:
            path = os.path.join(regime, f"snapshot_{sid:05d}.npz")
            if not os.path.exists(path): continue
            
            data = load_snapshot(path)
            f = data['f']
            
            # Coarsen if it's fine data to get compatible moment scales
            if regime == "data/fine_data":
                f = coarsen_data(f, config.NX, config.NV)
            
            diag = extract_diagnostics(f, v_grid, dv)
            
            n_min = min(n_min, float(diag['n'].min()))
            n_max = max(n_max, float(diag['n'].max()))
            v_min = min(v_min, float(diag['Vx'].min()))
            v_max = max(v_max, float(diag['Vx'].max()))
            
    # Add a 5% buffer
    n_buf = (n_max - n_min) * 0.05
    v_buf = (v_max - v_min) * 0.05
    
    return (n_min - n_buf, n_max + n_buf), (v_min - v_buf, v_max + v_buf)

def plot_triple_comparison(shot_id, n_lim=None, v_lim=None):
    """
    Generates a 3-column verification dashboard for a specific snapshot ID.
    Compare three evolved states at step T.
    """
    nx, nv = config.NX, config.NV
    dx, dv = config.DX, config.DV
    lv = nv * dv / 2
    x = jnp.linspace(0, nx*dx, nx, endpoint=False)
    v = jnp.linspace(-lv, lv, nv)
    
    # Define paths
    path_fine    = os.path.join("data/fine_data",      f"snapshot_{shot_id:05d}.npz")
    path_coarse  = os.path.join("data/coarse_data",    f"snapshot_{shot_id:05d}.npz")
    path_correct = os.path.join("data/corrected_data", f"snapshot_{shot_id:05d}.npz")
    
    if not all([os.path.exists(p) for p in [path_fine, path_coarse, path_correct]]):
        print(f" [!] Error: Snapshots for ID {shot_id} missing in one or more directories.")
        return

    # 1. Load Data
    data_f = load_snapshot(path_fine)
    data_c = load_snapshot(path_coarse)
    data_m = load_snapshot(path_correct)
    
    # Coarsen fine data for baseline alignment
    f_fine_c = coarsen_data(data_f['f'], nx, nv)
    
    # 2. Extract Diagnostics
    diag_f = extract_diagnostics(f_fine_c, v, dv)
    diag_c = extract_diagnostics(data_c['f'], v, dv)
    diag_m = extract_diagnostics(data_m['f'], v, dv)
    
    # 3. Plotting
    fig, axes = plt.subplots(4, 3, figsize=(18, 20), sharex='col', sharey='row')
    
    columns = [
        (diag_f, data_f['B'], data_f['E'], "FINE (Ground Truth)"),
        (diag_c, data_c['B'], data_c['E'], "COARSE (Baseline)"),
        (diag_m, data_m['B'], data_m['E'], "COARSE + ML (In-Loop)")
    ]
    
    for col_idx, (diag, B, E, title) in enumerate(columns):
        # Row 0: B-Fields (Transverse)
        ax = axes[0, col_idx]
        bx_plot = B[1] if B[1].shape[0] == nx else B[1][::B[1].shape[0]//nx]
        bz_plot = B[2] if B[2].shape[0] == nx else B[2][::B[2].shape[0]//nx]
        
        ax.plot(x, bx_plot, 'b-', label='By')
        ax.plot(x, bz_plot, 'r-', label='Bz')
        ax.set_title(title, fontweight='bold', fontsize=14)
        if col_idx == 0: ax.set_ylabel("Magnetic Fields", fontsize=12)
        ax.grid(True, alpha=0.3)
        if col_idx == 0: ax.legend()
        
        # Row 1: E-Fields
        ex_plot = E[0] if E[0].shape[0] == nx else E[0][::E[0].shape[0]//nx]
        ey_plot = E[1] if E[1].shape[0] == nx else E[1][::E[1].shape[0]//nx]
        ez_plot = E[2] if E[2].shape[0] == nx else E[2][::E[2].shape[0]//nx]
        
        ax = axes[1, col_idx]
        ax.plot(x, ey_plot, 'b-', label='Ey')
        ax.plot(x, ez_plot, 'r-', label='Ez')
        ax.plot(x, ex_plot, 'k--', label='Ex', alpha=0.3)
        if col_idx == 0: ax.set_ylabel("Electric Fields", fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Row 2: Ion Moments (Density n and Vx)
        ax = axes[2, col_idx]
        ax.plot(x, diag['n'], 'k-', label='Density n')
        if n_lim: ax.set_ylim(n_lim)
        
        ax_v = ax.twinx()
        ax_v.plot(x, diag['Vx'], 'g-', label='Vx')
        if v_lim: ax_v.set_ylim(v_lim)
        
        if col_idx == 2: ax_v.set_ylabel("Vx (Bulk Velocity)", color='g')
        if col_idx == 0: ax.set_ylabel("Density n", fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Row 3: Phase Space f(x, vx)
        ax = axes[3, col_idx]
        im = ax.pcolormesh(x, v, diag['f_vx'].T + 1e-15, shading='auto', 
                          cmap='inferno', norm=LogNorm(vmin=1e-8, vmax=diag_f['f_vx'].max()))
        ax.set_xlabel("x (d_i)")
        if col_idx == 0: ax.set_ylabel("vx (Velocity)", fontsize=12)
        
        if col_idx == 2:
            fig.colorbar(im, ax=axes[3, :], location='right', label='f(x, vx) [Log]')

    plt.tight_layout()
    output_dir = "plots/verification"
    os.makedirs(output_dir, exist_ok=True)
    save_path = f"{output_dir}/dashboard_{shot_id:05d}.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f" [OK] Dashboard saved to {save_path}")

if __name__ == "__main__":
    shot_ids = [10, 20, 30]
    print("=== Generating Triple Comparison Dashboards ===")
    
    # 1. First scan for global limits to ensure Row 2 consistency
    n_lim, v_lim = scan_global_limits(shot_ids)
    print(f" [Limits] Density: {n_lim[0]:.3f} to {n_lim[1]:.3f}")
    print(f" [Limits] Velocity: {v_lim[0]:.3f} to {v_lim[1]:.3f}")
    
    # 2. Plot with fixed limits
    for sid in shot_ids:
        plot_triple_comparison(sid, n_lim=n_lim, v_lim=v_lim)
