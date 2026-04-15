import matplotlib.pyplot as plt
import jax.numpy as jnp
import os
import glob
from data_io import load_snapshot



def plot_step_maxwell(i, x, v, f, B_x, B_y, B_z, E_x, E_y, E_z, lx, dx, dv, save_dir="plots_maxwell"):
    """
    Plots the electromagnetic fields and moments for the Hybrid Maxwell solver.
    """
    f_vx_x = jnp.sum(f, axis=(2, 3)) * (dv**2)  # Integrate over vy, vz
    
    # Calculate moments for subplot 3
    n_i = jnp.sum(f, axis=(1, 2, 3)) * (dv**3)
    n_i_safe = jnp.maximum(n_i, 1e-6)
    
    vx_grid = v[None, :, None, None]
    vy_grid = v[None, None, :, None]
    vz_grid = v[None, None, None, :]
    
    Vi_x = jnp.sum(f * vx_grid, axis=(1, 2, 3)) * (dv**3) / n_i_safe
    Vi_y = jnp.sum(f * vy_grid, axis=(1, 2, 3)) * (dv**3) / n_i_safe
    Vi_z = jnp.sum(f * vz_grid, axis=(1, 2, 3)) * (dv**3) / n_i_safe

    fig, axes = plt.subplots(4, 1, figsize=(10, 14), sharex=True)
    
    # 1. B fields
    axes[0].plot(x, B_x, label='Bx', color='black', linestyle='--')
    axes[0].plot(x, B_y, label='By', color='blue')
    axes[0].plot(x, B_z, label='Bz', color='red')
    axes[0].set_title(f"Magnetic Fields (Step {i})")
    axes[0].set_ylabel("B (B_0)")
    axes[0].legend(loc="upper right")
    axes[0].grid(True)
    
    # 2. E fields
    axes[1].plot(x, E_x, label='Ex', color='black', linestyle='--')
    axes[1].plot(x, E_y, label='Ey', color='blue')
    axes[1].plot(x, E_z, label='Ez', color='red')
    axes[1].set_title("Electric Fields")
    axes[1].set_ylabel("E (V_A B_0)")
    axes[1].legend(loc="upper right")
    axes[1].grid(True)
    
    # 3. Density and Velocity
    ax3 = axes[2]
    # Density on left axis
    ax3.plot(x, n_i, color='black', linestyle=':', linewidth=2.5, label='Density (n)')
    ax3.set_ylabel("Density (n_0)")
    ax3.set_title("Ion Moments")
    
    # Velocity on right axis
    ax3_rhs = ax3.twinx()
    ax3_rhs.plot(x, Vi_x, label='Vx', color='green', linestyle='-', linewidth=2.5)
    ax3_rhs.plot(x, Vi_y, label='Vy', color='magenta', linestyle='-', linewidth=2.5)
    ax3_rhs.plot(x, Vi_z, label='Vz', color='cyan', linestyle='-', linewidth=2.5)
    ax3_rhs.set_ylabel("Velocity (V_A)")
    
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_rhs.get_legend_handles_labels()
    ax3_rhs.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax3.grid(True)
    
    # 4. Phase Space f(x, vx)
    im = axes[3].pcolormesh(x, v, f_vx_x.T, shading='auto', cmap='jet')
    axes[3].set_title("Marginal Phase Space f(x, vx)")
    axes[3].set_xlabel("x (d_i)")
    axes[3].set_ylabel("vx (V_A)")
    #fig.colorbar(im, ax=axes[3])
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/step_{i:04d}.png", dpi=150)
    plt.close(fig)


def plot_initial_verification(x, n, T, P_gas, P_mag, P_tot, save_dir="plots_maxwell"):
    """
    Plots the initial state profiles for verification of pressure balance.
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    
    # 1. Density and Temperature
    axes[0].plot(x, n, label='Density (n)', color='black')
    axes[0].set_ylabel("n")
    ax0_rhs = axes[0].twinx()
    ax0_rhs.plot(x, T, label='Temperature (T)', color='orange')
    ax0_rhs.set_ylabel("T")
    axes[0].set_title("Initial Density and Temperature")
    
    lines1, labels1 = axes[0].get_legend_handles_labels()
    lines2, labels2 = ax0_rhs.get_legend_handles_labels()
    ax0_rhs.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    axes[0].grid(True)
    
    # 2. Pressures
    axes[1].plot(x, P_gas, label='Gas Pressure (nT)', color='green')
    axes[1].plot(x, P_mag, label='Mag Pressure (B^2/2)', color='blue')
    axes[1].set_title("Initial Pressure Components")
    axes[1].set_ylabel("Pressure")
    axes[1].legend(loc="upper right")
    axes[1].grid(True)
    
    # 3. Total Pressure balance
    axes[2].plot(x, P_tot, label='Total Pressure (P_gas + P_mag)', color='red', linewidth=2)
    axes[2].set_title("Initial Total Pressure (Should be Constant)")
    axes[2].set_xlabel("x (d_i)")
    axes[2].set_ylabel("Total Pressure")
    axes[2].legend(loc="upper right")
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/initial_verification.png", dpi=150)
    plt.close(fig)
    print(f"Verification plot saved to {save_dir}/initial_verification.png")


import config

if __name__ == "__main__":
    # --- Configuration ---
    snapshot_dir = config.SAVE_DIR
    output_base_dir = config.PLOT_DIR
    
    # 1. Setup output directory
    os.makedirs(output_base_dir, exist_ok=True)
    
    # 2. Find snapshots
    snapshot_pattern = os.path.join(snapshot_dir, "snapshot_*.npz")
    snapshot_files = sorted(glob.glob(snapshot_pattern))
    
    if not snapshot_files:
        print(f" [!] No snapshots found in {snapshot_dir}. Ensure simulation has run.")
    else:
        print(f" [Plotter] Found {len(snapshot_files)} snapshots. Starting batch processing...")
        
        # 3. Processing Loop
        for i, file_path in enumerate(snapshot_files):
            print(f"   --> Processing {os.path.basename(file_path)} ({i+1}/{len(snapshot_files)})")
            
            data = load_snapshot(file_path)
            step = data['step']
            f = data['f']
            B_x, B_y, B_z = data['B']
            E_x, E_y, E_z = data['E']
            
            # --- Dynamic Grid Resolution ---
            # Prioritize snapshot metadata, fallback to config
            meta = data.get('metadata', {})
            nx = int(meta.get('nx', f.shape[0]))
            nv = int(meta.get('nv', f.shape[1]))
            dx = float(meta.get('dx', config.DX))
            dv = float(meta.get('dv', config.DV))
            lx = float(meta.get('lx', dx * nx))
            lv = float(meta.get('lv', dv * nv / 2))
            
            x = jnp.linspace(0, lx, nx, endpoint=False)
            v = jnp.linspace(-lv, lv, nv)
            
            plot_step_maxwell(
                step, x, v, f, 
                B_x, B_y, B_z, 
                E_x, E_y, E_z, 
                lx, dx, dv, 
                save_dir=output_base_dir
            )

        print(f"\n [Done] All plots saved to {output_base_dir}/")
