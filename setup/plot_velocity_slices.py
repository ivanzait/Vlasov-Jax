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

# 7-stop 'HyperPlasma' colormap
cmd_colors = ['#080035', '#7F00FF', '#007FFF', '#00FF00', '#FFFF00', '#FF7F00', '#FF0000']
custom_cmap = LinearSegmentedColormap.from_list("HyperPlasma", cmd_colors, N=256)

def plot_velocity_slices(step, v, f, out_dir="plots_slices"):
    """
    Plots 2D velocity distribution slices (Vx-Vy, Vy-Vz, Vx-Vz) 
    at the spatial center of the simulation.
    """
    nx = f.shape[0]
    mid_x = nx // 2
    
    # Extract distribution at spatial center
    # f shape: (NX, NV, NV, NV)
    f_mid = f[mid_x]
    
    # Marginalize (integrate) over the 3rd dimension for each 2D projection
    f_xy = jnp.sum(f_mid, axis=2)
    f_yz = jnp.sum(f_mid, axis=0)
    f_xz = jnp.sum(f_mid, axis=1)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    vmax = 0.01
    vmin = 1e-4

    # 1. Vx-Vy
    im0 = axes[0].pcolormesh(v, v, f_xy.T, shading='auto', cmap=custom_cmap, norm=LogNorm(vmin=vmin, vmax=vmax))
    axes[0].set_title(r"$f(V_x, V_y)$")
    axes[0].set_xlabel(r"$V_x / V_A$")
    axes[0].set_ylabel(r"$V_y / V_A$")
    fig.colorbar(im0, ax=axes[0])
    
    # 2. Vy-Vz
    im1 = axes[1].pcolormesh(v, v, f_yz.T, shading='auto', cmap=custom_cmap, norm=LogNorm(vmin=vmin, vmax=vmax))
    axes[1].set_title(r"$f(V_y, V_z)$")
    axes[1].set_xlabel(r"$V_y / V_A$")
    axes[1].set_ylabel(r"$V_z / V_A$")
    fig.colorbar(im1, ax=axes[1])
    
    # 3. Vx-Vz
    im2 = axes[2].pcolormesh(v, v, f_xz.T, shading='auto', cmap=custom_cmap, norm=LogNorm(vmin=vmin, vmax=vmax))
    axes[2].set_title(r"$f(V_x, V_z)$")
    axes[2].set_xlabel(r"$V_x / V_A$")
    axes[2].set_ylabel(r"$V_z / V_A$")
    fig.colorbar(im2, ax=axes[2])
    
    plt.suptitle(f"Velocity Distribution at $x \sim L/2$ (Step {step})", y=1.05)
    plt.tight_layout()
    
    os.makedirs(out_dir, exist_ok=True)
    save_path = f"{out_dir}/v_slices_step_{step:04d}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Velocity slices saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot velocity slices at spatial center.")
    parser.add_argument("--data_dir", type=str, required=True, help="Data directory.")
    parser.add_argument("--step", type=int, default=0, help="Timestep.")
    parser.add_argument("--out_dir", type=str, default="plots/v_slices", help="Output directory.")
    
    args = parser.parse_args()
    
    file_path = os.path.join(args.data_dir, f"step_{args.step:04d}.npz")
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
    else:
        print(f"Loading {file_path} for velocity slicing...")
        data = jnp.load(file_path)
        plot_velocity_slices(args.step, data['v'], data['f'], args.out_dir)
