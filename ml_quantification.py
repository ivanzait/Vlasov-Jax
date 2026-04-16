import jax
import jax.numpy as jnp
import pickle
import numpy as np
import os
from ml_models import MLP, get_n_v_from_f

def quantify_ml_improvement():
    # 1. Load weights and test data
    weight_path = 'ml_weights/mlp_final_phys.pkl'
    test_path = 'ml_weights/test_data_split.pkl'
    
    if not os.path.exists(weight_path) or not os.path.exists(test_path):
        print("Error: Training results not found. Please run train_offline.py first.")
        return

    with open(weight_path, 'rb') as f:
        params = pickle.load(f)
    with open(test_path, 'rb') as f:
        test_data = pickle.load(f)
        
    inputs = test_data['inputs']
    labels_log_true = test_data['labels']
    v = test_data['v']
    dv = test_data['dv']
    
    print(f"--- ML Quantification Scorecard (Test Set: {inputs.shape[0]} samples) ---")
    
    # 2. Inference
    pred_log_residuals = MLP.forward(params, inputs)
    
    # 3. Reconstruct Distributions
    f_coarse = inputs[:, :32**3]
    f_pred = f_coarse * jnp.exp(pred_log_residuals)
    f_true = f_coarse * jnp.exp(labels_log_true)
    
    # 4. Calculate Moment Errors
    print("Calculating moment errors (n, V)...")
    n_pred, v_pred = get_n_v_from_f(f_pred, v, dv)
    n_true, v_true = get_n_v_from_f(f_true, v, dv)
    n_coarse, v_coarse = get_n_v_from_f(f_coarse, v, dv)
    
    # Baseline Errors
    err_n_base = jnp.mean(jnp.abs(n_coarse - n_true))
    err_v_base = jnp.mean(jnp.abs(v_coarse - v_true))
    
    # ML Errors
    err_n_ml = jnp.mean(jnp.abs(n_pred - n_true))
    err_v_ml = jnp.mean(jnp.abs(v_pred - v_true))
    
    # 5. Output Scorecard
    print("\n| Metric         | Coarse Baseline | ML-Recovered | Improvement (%) |")
    print("|----------------|-----------------|--------------|-----------------|")
    print(f"| L2 (log-f)     | {jnp.mean(labels_log_true**2):.6f}        | {jnp.mean((pred_log_residuals - labels_log_true)**2):.6f}     | {(1 - jnp.mean((pred_log_residuals - labels_log_true)**2)/jnp.mean(labels_log_true**2))*100:.1f}%            |")
    print(f"| MAE Density (n)| {err_n_base:.6f}        | {err_n_ml:.6f}     | {(1 - err_n_ml/err_n_base)*100:.1f}%            |")
    print(f"| MAE Velocity (V)| {err_v_base:.6f}        | {err_v_ml:.6f}     | {(1 - err_v_ml/err_v_base)*100:.1f}%            |")

if __name__ == "__main__":
    quantify_ml_improvement()
