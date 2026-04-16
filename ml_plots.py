import jax
import jax.numpy as jnp
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
from ml_dataset import downsample_velocity, get_gradients
from ml_models import MLP, get_n_v_from_f

def load_full_snapshot_and_predict(step, params, coarse_dir='data/coarse', epsilon=1e-12):
    """
    Loads a full coarse snapshot and runs ML inference.
    """
    path = os.path.join(coarse_dir, f"step_{step:04d}.npz")
    with np.load(path) as data:
        f_coarse = jnp.array(data['f'])
        e_coarse = jnp.stack([data['E_x'], data['E_y'], data['E_z']], axis=-1)
        b_coarse = jnp.stack([data['B_x'], data['B_y'], data['B_z']], axis=-1)
        dx = float(data['dx'])
        x = jnp.array(data['x'])
        v = jnp.array(data['v'])
        dv = v[1] - v[0]

    # Calculate Gradients for MLP input
    de_dx = jnp.stack([get_gradients(e_coarse[..., i], dx) for i in range(3)], axis=-1)
    db_dx = jnp.stack([get_gradients(b_coarse[..., i], dx) for i in range(3)], axis=-1)

    nx = f_coarse.shape[0]
    f_flat = f_coarse.reshape(nx, -1)
    e_flat = e_coarse.reshape(nx, 3)
    b_flat = b_coarse.reshape(nx, 3)
    de_flat = de_dx.reshape(nx, 3)
    db_flat = db_dx.reshape(nx, 3)

    inputs = jnp.concatenate([f_flat, e_flat, b_flat, de_flat, db_flat], axis=1)
    
    # ML Inference
    pred_log_residuals = MLP.forward(params, inputs)
    f_pred = f_coarse * jnp.exp(pred_log_residuals.reshape(f_coarse.shape))
    
    return x, v, dv, f_coarse, f_pred, e_coarse, dx

def create_4row_verification_plot(step=50):
    # 1. Load Weights
    weight_path = 'ml_weights/mlp_final_phys.pkl'
    if not os.path.exists(weight_path):
        print("Error: Weights not found.")
        return
        
    with open(weight_path, 'rb') as f:
        params = pickle.load(f)

    # 2. Reconstruct ML state
    x, v, dv, f_coarse, f_ml, e_coarse, dx = load_full_snapshot_and_predict(step, params)
    
    # 3. Load Fine reference
    with np.load(f'data/fine/step_{step:04d}.npz') as data_f:
        f_f = downsample_velocity(jnp.array(data_f['f']))
        e_f = jnp.stack([data_f['E_x'], data_f['E_y'], data_f['E_z']], axis=-1)
        # Assuming NX is same for purpose of overlay, or we slice if needed.
    
    # 4. Calculate Moments
    n_f, vx_f = get_n_v_from_f(f_f.reshape(-1, 32**3), v, dv)
    n_c, vx_c = get_n_v_from_f(f_coarse.reshape(-1, 32**3), v, dv)
    n_m, vx_m = get_n_v_from_f(f_ml.reshape(-1, 32**3), v, dv)

    # 5. Build the 4-Row Dashboard
    fig = plt.figure(figsize=(15, 18))
    gs = fig.add_gridspec(4, 3)
    
    # Shared X-axis for top 3 rows
    ax_e = fig.add_subplot(gs[0, :])
    ax_n = fig.add_subplot(gs[1, :], sharex=ax_e)
    ax_v = fig.add_subplot(gs[2, :], sharex=ax_e)
    
    # Row 1: E(x) - Overlaid
    ax_e.plot(x, e_f[:, 0], 'k-', label='Fine (Ex)', alpha=0.8)
    ax_e.plot(x, e_coarse[:, 0], 'r--', label='Coarse (Ex)', alpha=0.8)
    ax_e.set_title(f"Electric Field Ex(x) - Step {step}")
    ax_e.set_ylabel("Ex")
    ax_e.grid(True, alpha=0.3)
    ax_e.legend()

    # Row 2: n(x) - Overlaid
    ax_n.plot(x, n_f, 'k-', linewidth=2, label='Fine Density (n)')
    ax_n.plot(x, n_c, 'r--', label='Coarse Baseline')
    ax_n.plot(x, n_m, 'b-', label='ML-Improved')
    ax_n.set_title("Density Profile n(x)")
    ax_n.set_ylabel("n")
    ax_n.grid(True, alpha=0.3)
    ax_n.legend()

    # Row 3: V(x) - Overlaid
    ax_v.plot(x, vx_f, 'k-', linewidth=2, label='Fine Velocity (Vx)')
    ax_v.plot(x, vx_c, 'r--', label='Coarse Baseline')
    ax_v.plot(x, vx_m, 'b-', label='ML-Improved')
    ax_v.set_title("Velocity Profile Vx(x)")
    ax_v.set_ylabel("Vx")
    ax_v.grid(True, alpha=0.3)
    ax_v.legend()

    # Row 4: f(Vx, x) - Side-by-Side
    # Marginalize over vy, vz
    f_f_ps = jnp.sum(f_f, axis=(2, 3)) * (dv**2)
    f_c_ps = jnp.sum(f_coarse, axis=(2, 3)) * (dv**2)
    f_m_ps = jnp.sum(f_ml, axis=(2, 3)) * (dv**2)
    
    v_mesh, x_mesh = jnp.meshgrid(v, x)
    
    axes_ps = [fig.add_subplot(gs[3, 0]), fig.add_subplot(gs[3, 1]), fig.add_subplot(gs[3, 2])]
    ps_data = [f_f_ps, f_c_ps, f_m_ps]
    ps_titles = ["Fine f(x, vx)", "Coarse f(x, vx)", "ML-Improved f(x, vx)"]
    
    from matplotlib.colors import LogNorm
    for i in range(3):
        # Using LogNorm for the phase space density to see the tails
        im = axes_ps[i].pcolormesh(x, v, ps_data[i].T, shading='auto', cmap='jet', norm=LogNorm(vmin=1e-5, vmax=1.0))
        axes_ps[i].set_title(ps_titles[i])
        axes_ps[i].set_xlabel("x")
        if i == 0: axes_ps[i].set_ylabel("vx")
        fig.colorbar(im, ax=axes_ps[i])

    plt.tight_layout()
    save_path = f"plots/ml_4row_dashboard_step_{step}.png"
    plt.savefig(save_path, dpi=150)
    print(f"4-row dashboard saved to {save_path}")

if __name__ == "__main__":
    os.makedirs('plots', exist_ok=True)
    create_4row_verification_plot(step=50)
    create_4row_verification_plot(step=30)
