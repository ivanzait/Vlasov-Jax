import matplotlib.pyplot as plt
import jax.numpy as jnp

def plot_step(i, x, v, f, Ex, lx, dx, dv, save_dir="plots"):
    """
    Plots the electric field Ex and the marginal 1D phase-space density f(x, vx).
    """
    f_vx_x = jnp.sum(f, axis=(2, 3)) * (dv**2)  # Integrate over vy, vz
            
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    
    # Plot Ex
    ax1.plot(x, Ex, color='blue')
    ax1.set_title(f"Electric Field Ex at Step {i}")
    ax1.set_ylabel("Ex")
    ax1.set_xlim([0, lx])
    ax1.set_ylim([-0.6, 0.6])
    ax1.grid(True)
    
    # Plot f(x, vx)
    im = ax2.pcolormesh(x, v, f_vx_x.T, shading='auto', cmap='viridis')
    ax2.set_title("Phase Space Marginal f(x, vx)")
    ax2.set_xlabel("x")
    ax2.set_ylabel("vx")
    fig.colorbar(im, ax=ax2)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/step_{i:04d}.png")
    plt.close(fig)
