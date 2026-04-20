import matplotlib.pyplot as plt
import jax.numpy as jnp
import os
import argparse
from matplotlib.colors import LogNorm, LinearSegmentedColormap

# Global publication-ready styling
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 12,
    'font.family': 'serif',
    'mathtext.fontset': 'dejavuserif',
    'axes.linewidth': 1.5
})

# 6-stop 'HyperPlasma' colormap: Dark Blue -> Purple -> Green -> Yellow -> Orange -> Red
# Designed for maximum visibility of low-density beams and high-energy tails
cmd_colors = ['#080035', '#7F00FF', '#007FFF', '#00FF00', '#FFFF00', '#FF7F00', '#FF0000']
custom_cmap = LinearSegmentedColormap.from_list("HyperPlasma", cmd_colors, N=256)


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
    axes[0].plot(x, B_x, label='$B_x$', color='black', linestyle='--')
    axes[0].plot(x, B_y, label='$B_y$', color='blue')
    axes[0].plot(x, B_z, label='$B_z$', color='red')
    axes[0].set_title(f"Magnetic Fields (Step {i})")
    axes[0].set_ylabel(r"$B / B_0$")
    axes[0].legend(loc="upper right", frameon=True)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(x[0], x[-1])
    
    # 2. E fields
    axes[1].plot(x, E_x, label='$E_x$', color='black', linestyle='--')
    axes[1].plot(x, E_y, label='$E_y$', color='blue')
    axes[1].plot(x, E_z, label='$E_z$', color='red')
    axes[1].set_title("Electric Fields")
    axes[1].set_ylabel(r"$E / (V_A B_0)$")
    axes[1].legend(loc="upper right", frameon=True)
    axes[1].grid(True, alpha=0.3)
    
    # 3. Density and Velocity
    ax3 = axes[2]
    # Density on left axis
    ax3.plot(x, n_i, color='black', linestyle=':', linewidth=2.5, label='Density ($n$)')
    ax3.set_ylabel(r"Density $n / n_0$")
    ax3.set_title("Ion Moments")
    
    # Velocity on right axis
    ax3_rhs = ax3.twinx()
    ax3_rhs.plot(x, Vi_x, label='$V_x$', color='green', linestyle='-', linewidth=2.5)
    ax3_rhs.plot(x, Vi_y, label='$V_y$', color='magenta', linestyle='-', linewidth=2.5)
    ax3_rhs.plot(x, Vi_z, label='$V_z$', color='cyan', linestyle='-', linewidth=2.5)
    ax3_rhs.set_ylabel(r"Velocity $V / V_A$")
    
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_rhs.get_legend_handles_labels()
    ax3_rhs.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax3.grid(True)
    
    # 4. Phase Space f(x, vx)
    # Using LogNorm for distribution details and the unique custom_cmap
    im = axes[3].pcolormesh(x, v, f_vx_x.T, shading='auto', cmap=custom_cmap, norm=LogNorm(vmin=1e-4, vmax=0.5))
    axes[3].set_title(r"Marginal Phase Space $f(x, V_x)$")
    axes[3].set_xlabel(r"$x / d_i$")
    axes[3].set_ylabel(r"$V_x / V_A$")
    fig.colorbar(im, ax=axes[3], label="$f$ (log scale)")
    
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
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/initial_verification.png", dpi=150)
    plt.close(fig)
    print(f"Verification plot saved to {save_dir}/initial_verification.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot simulation results from .npz files.")
    parser.add_argument("--data_dir", type=str, default="data/test", help="Directory containing .npz files.")
    parser.add_argument("--step", type=int, default=0, help="Timestep to plot.")
    parser.add_argument("--out_dir", type=str, default="plots_maxwell", help="Directory to save the plots.")
    
    args = parser.parse_args()
    
    file_path = os.path.join(args.data_dir, f"step_{args.step:04d}.npz")
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
    else:
        print(f"Loading data from {file_path}...")
        data = jnp.load(file_path)
        
        # Unpack data
        f = data['f']
        B_x, B_y, B_z = data['B_x'], data['B_y'], data['B_z']
        E_x, E_y, E_z = data['E_x'], data['E_y'], data['E_z']
        x, v = data['x'], data['v']
        dx, dv = data['dx'], data['dv']
        
        # Reconstruct lx
        lx = float(x[-1] + dx)
        
        # Ensure output directory exists
        os.makedirs(args.out_dir, exist_ok=True)
        
        print(f"Plotting step {args.step}...")
        plot_step_maxwell(args.step, x, v, f, B_x, B_y, B_z, E_x, E_y, E_z, lx, dx, dv, save_dir=args.out_dir)
        print(f"Plot saved to {args.out_dir}/step_{args.step:04d}.png")
