import jax
import jax.numpy as jnp
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
from ml_dataset import load_simulation_data, downsample_velocity, get_gradients
from ml_models import MLP, get_n_v_from_f

def load_full_snapshot_and_predict(step, params, coarse_dir='data/coarse', epsilon=1e-12):
    """
    Loads a full coarse snapshot using the Pure Dictionary loader and runs ML inference.
    """
    data_c = load_simulation_data(coarse_dir, [step])
    f_coarse = data_c['f'][0]
    e_coarse = data_c['E'][0]
    b_coarse = data_c['B'][0]
    
    meta = data_c['metadata']
    dx, v, dv, x = meta['dx'], meta['v'], meta['dv'], meta['x']

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
    weight_path = 'data/ml_weights/mlp_final_phys.pkl'
    if not os.path.exists(weight_path):
        print("Error: Weights not found.")
        return
        
    with open(weight_path, 'rb') as f:
        params = pickle.load(f)

    # 2. Reconstruct ML state
    x, v, dv, f_coarse, f_ml, e_coarse, dx = load_full_snapshot_and_predict(step, params)
    
    # 3. Load Fine reference using Pure loader
    data_f_full = load_simulation_data('data/fine', [step])
    f_f = downsample_velocity(data_f_full['f'][0])
    e_f = data_f_full['E'][0]
    
    # 4. Calculate Moments
    n_f, vx_f = get_n_v_from_f(f_f.reshape(-1, 32**3), v, dv)
    n_c, vx_c = get_n_v_from_f(f_coarse.reshape(-1, 32**3), v, dv)
    n_m, vx_m = get_n_v_from_f(f_ml.reshape(-1, 32**3), v, dv)

    # 5. Build the 4-Row Dashboard
    fig = plt.figure(figsize=(15, 18))
    gs = fig.add_gridspec(4, 3)
    
    ax_e = fig.add_subplot(gs[0, :])
    ax_n = fig.add_subplot(gs[1, :], sharex=ax_e)
    ax_v = fig.add_subplot(gs[2, :], sharex=ax_e)
    
    ax_e.plot(x, e_f[:, 0], 'k-', label='Fine (Ex)', alpha=0.8)
    ax_e.plot(x, e_coarse[:, 0], 'r--', label='Coarse (Ex)', alpha=0.8)
    ax_e.set_title(f"Electric Field Ex(x) - Step {step}")
    ax_e.set_ylabel("Ex")
    ax_e.grid(True, alpha=0.3)
    ax_e.legend()

    ax_n.plot(x, n_f, 'k-', linewidth=2, label='Fine Density (n)')
    ax_n.plot(x, n_c, 'r--', label='Coarse Baseline')
    ax_n.plot(x, n_m, 'b-', label='ML-Improved')
    ax_n.set_title("Density Profile n(x)")
    ax_n.set_ylabel("n")
    ax_n.grid(True, alpha=0.3)
    ax_n.legend()

    ax_v.plot(x, vx_f, 'k-', linewidth=2, label='Fine Velocity (Vx)')
    ax_v.plot(x, vx_c, 'r--', label='Coarse Baseline')
    ax_v.plot(x, vx_m, 'b-', label='ML-Improved')
    ax_v.set_title("Velocity Profile Vx(x)")
    ax_v.set_ylabel("Vx")
    ax_v.grid(True, alpha=0.3)
    ax_v.legend()

    f_f_ps = jnp.sum(f_f, axis=(2, 3)) * (dv**2)
    f_c_ps = jnp.sum(f_coarse, axis=(2, 3)) * (dv**2)
    f_m_ps = jnp.sum(f_ml, axis=(2, 3)) * (dv**2)
    
    axes_ps = [fig.add_subplot(gs[3, 0]), fig.add_subplot(gs[3, 1]), fig.add_subplot(gs[3, 2])]
    ps_data = [f_f_ps, f_c_ps, f_m_ps]
    ps_titles = ["Fine f(x, vx)", "Coarse f(x, vx)", "ML-Improved f(x, vx)"]
    
    from matplotlib.colors import LogNorm
    for i in range(3):
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
