import jax
import jax.numpy as jnp
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from .ml_dataset import load_simulation_data, downsample_velocity, get_gradients
from .ml_models import MLP, get_n_v_from_f

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
cmd_colors = ['#080035', '#7F00FF', '#007FFF', '#00FF00', '#FFFF00', '#FF7F00', '#FF0000']
custom_cmap = LinearSegmentedColormap.from_list("HyperPlasma", cmd_colors, N=256)

import argparse
from .ml_configs import get_config, get_input_dim

def load_full_snapshot_and_predict(step, params, config_name, coarse_dir='data/coarse', epsilon=1e-12):
    """
    Loads a full coarse snapshot and runs ML inference using the specific feature config.
    """
    exp_config = get_config(config_name)
    data_c = load_simulation_data(coarse_dir, [step])
    f_coarse = data_c['f'][0]
    e_coarse = data_c['E'][0]
    b_coarse = data_c['B'][0]
    
    meta = data_c['metadata']
    dx, v, dv, x = meta['dx'], meta['v'], meta['dv'], meta['x']

    nx = f_coarse.shape[0]
    feature_list = []
    
    if exp_config.get('f'):
        feature_list.append(f_coarse.reshape(nx, -1))
    
    if exp_config.get('E'):
        feature_list.append(e_coarse.reshape(nx, 3))
        
    if exp_config.get('B'):
        feature_list.append(b_coarse.reshape(nx, 3))
        
    if exp_config.get('grad_E'):
        de_dx = jnp.stack([get_gradients(e_coarse[..., i], dx) for i in range(3)], axis=-1)
        feature_list.append(de_dx.reshape(nx, 3))
        
    if exp_config.get('grad_B'):
        db_dx = jnp.stack([get_gradients(b_coarse[..., i], dx) for i in range(3)], axis=-1)
        feature_list.append(db_dx.reshape(nx, 3))
        
    inputs = jnp.concatenate(feature_list, axis=1)
    
    # ML Inference
    pred_log_residuals = MLP.forward(params, inputs)
    f_pred = f_coarse * jnp.exp(pred_log_residuals.reshape(f_coarse.shape))
    
    return x, v, dv, f_coarse, f_pred, e_coarse, dx

def create_4row_verification_plot(config_name='baseline', step=50):
    # 1. Load Weights for specific config
    weight_path = f'data/ml_weights/mlp_{config_name}.pkl'
    if not os.path.exists(weight_path):
        print(f"Error: Weights not found at {weight_path}")
        return
        
    with open(weight_path, 'rb') as f:
        params = pickle.load(f)

    # 2. Reconstruct ML state
    x, v, dv, f_coarse, f_ml, e_coarse, dx = load_full_snapshot_and_predict(step, params, config_name)
    
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
    
    ax_e.plot(x, e_f[:, 0], 'k-', linewidth=2.5, label='Fine Reference', alpha=0.9)
    ax_e.plot(x, e_coarse[:, 0], 'r--', linewidth=1, label='Coarse Input', alpha=0.8)
    ax_e.set_title(f"Electric Field Profile - Step {step}")
    ax_e.set_ylabel(r"$E_x / (V_A B_0)$")
    ax_e.grid(True, alpha=0.3)
    ax_e.legend(frameon=True)
    ax_e.set_xlim(x[0], x[-1])

    ax_n.plot(x, n_f, 'k-', linewidth=2.5, label='Fine Reference')
    ax_n.plot(x, n_c, 'r--', linewidth=1, label='Coarse Baseline')
    ax_n.plot(x, n_m, 'b-', linewidth=2, label='ML-Improved')
    ax_n.set_title("Density Profile $n(x)$")
    ax_n.set_ylabel(r"Density $n / n_0$")
    ax_n.grid(True, alpha=0.3)
    ax_n.legend(frameon=True)

    ax_v.plot(x, vx_f, 'k-', linewidth=2.5, label='Fine Reference')
    ax_v.plot(x, vx_c, 'r--', linewidth=1, label='Coarse Baseline')
    ax_v.plot(x, vx_m, 'b-', linewidth=2, label='ML-Improved')
    ax_v.set_title("Velocity Profile $V_x(x)$")
    ax_v.set_ylabel(r"Velocity $V_x / V_A$")
    ax_v.grid(True, alpha=0.3)
    ax_v.legend(frameon=True)

    f_f_ps = jnp.sum(f_f, axis=(2, 3)) * (dv**2)
    f_c_ps = jnp.sum(f_coarse, axis=(2, 3)) * (dv**2)
    f_m_ps = jnp.sum(f_ml, axis=(2, 3)) * (dv**2)
    
    axes_ps = [fig.add_subplot(gs[3, 0]), fig.add_subplot(gs[3, 1]), fig.add_subplot(gs[3, 2])]
    ps_data = [f_f_ps, f_c_ps, f_m_ps]
    ps_titles = ["Fine Reference", "Coarse Baseline", "ML-Improved"]
    
    from matplotlib.colors import LogNorm
    for i in range(3):
        im = axes_ps[i].pcolormesh(x, v, ps_data[i].T, shading='auto', cmap=custom_cmap, norm=LogNorm(vmin=1e-4, vmax=0.5))
        axes_ps[i].set_title(ps_titles[i])
        axes_ps[i].set_xlabel(r"$x / d_i$")
        if i == 0: axes_ps[i].set_ylabel(r"$V_x / V_A$")
        fig.colorbar(im, ax=axes_ps[i])

    plt.tight_layout()
    save_path = f"plots/ml_4row_{config_name}_step_{step}.png"
    plt.savefig(save_path, dpi=150)
    print(f"4-row dashboard saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ML Verification Plots.")
    parser.add_argument("--config", type=str, default="baseline", help="baseline, no_grad, or all")
    args = parser.parse_args()
    
    os.makedirs('plots', exist_ok=True)
    
    configs = [args.config] if args.config != 'all' else ['baseline', 'no_grad']
    
    steps = [50, 70, 90]
    for cfg in configs:
        for step in steps:
            create_4row_verification_plot(config_name=cfg, step=step)
