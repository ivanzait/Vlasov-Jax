import jax
import jax.numpy as jnp
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from .ml_dataset import load_simulation_data, downsample_velocity, upsample_velocity, get_gradients
from .ml_models import MLP, get_n_v_from_f

def evaluate_generalization(weight_path='data/ml_weights/mlp_final_phys.pkl', test_dir='data/coarse_nv16', step=50):
    """
    Evaluates the trained MLP on a simulation with a DIFFERENT resolution than 32^3.
    Demonstrates the effectiveness of the Resolution Adapter for generalization.
    """
    if not os.path.exists(weight_path):
        print(f"Error: Weights not found at {weight_path}")
        return

    with open(weight_path, 'rb') as f:
        params = pickle.load(f)

    print(f"--- Evaluating Generalization for {test_dir} (Step {step}) ---")
    
    # 1. Load Data (Super-Coarse or Unseen)
    data_c = load_simulation_data(test_dir, [step])
    f_coarse = data_c['f'][0]
    e_coarse = data_c['E'][0]
    b_coarse = data_c['B'][0]
    meta = data_c['metadata']
    dx, v_c, dv_c, x = meta['dx'], meta['v'], meta['dv'], meta['x']
    
    coarse_nv = f_coarse.shape[-1]
    
    # 2. Apply Resolution Adapter (Upsample to 32^3 for MLP)
    print(f"  [Adapter] Upsampling {coarse_nv}^3 source to 32^3 canonical grid...")
    f_coarse_32 = upsample_velocity(f_coarse, target_nv=32)
    
    # 3. Build Features (Using Canonical 32^3)
    nx = f_coarse_32.shape[0]
    de_dx = jnp.stack([get_gradients(e_coarse[..., i], dx) for i in range(3)], axis=-1)
    db_dx = jnp.stack([get_gradients(b_coarse[..., i], dx) for i in range(3)], axis=-1)
    
    inputs = jnp.concatenate([
        f_coarse_32.reshape(nx, -1),
        e_coarse.reshape(nx, 3),
        b_coarse.reshape(nx, 3),
        de_dx.reshape(nx, 3),
        db_dx.reshape(nx, 3)
    ], axis=1)
    
    # 4. Predict in Canonical Space
    pred_log_residuals = MLP.forward(params, inputs)
    f_ml_32 = f_coarse_32 * jnp.exp(pred_log_residuals.reshape(f_coarse_32.shape))
    
    # 5. Downsample ML correction back to Target Resolution for verification
    print(f"  [Adapter] Downsampling correction back to {coarse_nv}^3 for verification...")
    f_ml_target = downsample_velocity(f_ml_32, target_nv=coarse_nv)
    
    # 6. Load Fine Reference (Ground Truth at Target Resolution)
    data_f_full = load_simulation_data('data/fine', [step])
    f_f_target = downsample_velocity(data_f_full['f'][0], target_nv=coarse_nv)
    
    # 7. Quantification
    # Note: v and dv must match the target resolution grid
    n_f, vx_f = get_n_v_from_f(f_f_target.reshape(-1, coarse_nv**3), v_c, dv_c)
    n_c, vx_c = get_n_v_from_f(f_coarse.reshape(-1, coarse_nv**3), v_c, dv_c)
    n_m, vx_m = get_n_v_from_f(f_ml_target.reshape(-1, coarse_nv**3), v_c, dv_c)
    
    # helper for correlation
    def get_corr(a, b):
        return float(np.corrcoef(a, b)[0, 1])

    l2_baseline = jnp.sqrt(jnp.mean((jnp.log(f_f_target+1e-12) - jnp.log(f_coarse+1e-12))**2))
    l2_ml = jnp.sqrt(jnp.mean((jnp.log(f_f_target+1e-12) - jnp.log(f_ml_target+1e-12))**2))
    imp_l2 = (l2_baseline - l2_ml) / l2_baseline * 100
    
    corr_n_baseline = get_corr(n_f, n_c)
    corr_n_ml       = get_corr(n_f, n_m)
    corr_v_baseline = get_corr(vx_f, vx_c)
    corr_v_ml       = get_corr(vx_f, vx_m)

    print("\n--- Generalization Results ---")
    print(f"L2 (log-f) Baseline: {l2_baseline:.4f}")
    print(f"L2 (log-f) ML-Corrected: {l2_ml:.4f} ({imp_l2:.1f}% Improvement)")
    print(f"\n--- Momentum Correlation (Pearson r) ---")
    print(f"Density (n)   | Baseline: {corr_n_baseline:.4f} | ML-Corrected: {corr_n_ml:.4f}")
    print(f"Velocity (Vx) | Baseline: {corr_v_baseline:.4f} | ML-Corrected: {corr_v_ml:.4f}")
    
    # 8. Visualization
    plt.rcParams.update({'font.size': 14, 'font.family': 'serif'})
    # ... (Plotting code similar to ml_plots.py but adjusted for generalization)
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    axes[0].plot(x, n_f, 'k-', label='Fine Reference')
    axes[0].plot(x, n_c, 'r--', label='Coarse Baseline')
    axes[0].plot(x, n_m, 'b-', label='ML-Improved')
    axes[0].set_title(f"Density Generalization on {coarse_nv}^3 Grid")
    axes[0].set_ylabel(r"$n / n_0$")
    axes[0].legend()
    
    axes[1].plot(x, vx_f, 'k-', label='Fine Reference')
    axes[1].plot(x, vx_c, 'r--', label='Coarse Baseline')
    axes[1].plot(x, vx_m, 'b-', label='ML-Improved')
    axes[1].set_title(f"Velocity Generalization on {coarse_nv}^3 Grid")
    axes[1].set_ylabel(r"$V_x / V_A$")
    axes[1].set_xlabel(r"$x / d_i$")
    axes[1].legend()
    
    plt.tight_layout()
    save_path = f"plots/generalization_{coarse_nv}_step_{step}.png"
    plt.savefig(save_path, dpi=150)
    print(f"Generalization plot saved to {save_path}")

if __name__ == "__main__":
    os.makedirs('plots', exist_ok=True)
    # Test on Native 32^3
    evaluate_generalization(test_dir='data/coarse', step=50)
    # Test on Super-Coarse 16^3
    evaluate_generalization(test_dir='data/coarse_nv16', step=50)
