import jax
import jax.numpy as jnp
import pickle
import numpy as np
import os
import argparse
from .ml_models import MLP, get_n_v_from_f
from .ml_configs import get_config

def quantify_ml_improvement(config_name='baseline'):
    # 1. Load weights and test data for specific config
    weight_path = f'data/ml_weights/mlp_{config_name}.pkl'
    test_path = f'data/ml_weights/test_data_{config_name}.pkl'
    
    if not os.path.exists(weight_path) or not os.path.exists(test_path):
        print(f"Error: Training results for '{config_name}' not found.")
        print(f"Expected: {weight_path} and {test_path}")
        return

    with open(weight_path, 'rb') as f:
        params = pickle.load(f)
    with open(test_path, 'rb') as f:
        test_data = pickle.load(f)
        
    inputs = test_data['inputs']
    labels_log_true = test_data['labels']
    v = test_data['v']
    dv = test_data['dv']
    
    print(f"\n--- ML Quantification Scorecard [{config_name}] (Test Set: {inputs.shape[0]} samples) ---")
    
    # 2. Inference
    pred_log_residuals = MLP.forward(params, inputs)
    
    # 3. Reconstruct Distributions
    # Slice the distribution from the inputs (always the first 32^3 values)
    f_coarse = inputs[:, :32**3]
    f_pred = f_coarse * jnp.exp(pred_log_residuals)
    f_true = f_coarse * jnp.exp(labels_log_true)
    
    # 4. Calculate Moment Errors
    n_pred, v_pred = get_n_v_from_f(f_pred, v, dv)
    n_true, v_true = get_n_v_from_f(f_true, v, dv)
    n_coarse, v_coarse = get_n_v_from_f(f_coarse, v, dv)
    
    # Baseline Errors (MAE)
    err_n_base = jnp.mean(jnp.abs(n_coarse - n_true))
    err_v_base = jnp.mean(jnp.abs(v_coarse - v_true))
    
    # ML Errors (MAE)
    err_n_ml = jnp.mean(jnp.abs(n_pred - n_true))
    err_v_ml = jnp.mean(jnp.abs(v_pred - v_true))
    
    # Relative Errors (%) (Global L1-norm based)
    # This avoids inflation from small denominators in near-zero regions
    eps = 1e-12
    sum_n_true = jnp.sum(jnp.abs(n_true)) + eps
    sum_v_true = jnp.sum(jnp.abs(v_true)) + eps
    
    rel_n_base = (jnp.sum(jnp.abs(n_coarse - n_true)) / sum_n_true) * 100
    rel_n_ml = (jnp.sum(jnp.abs(n_pred - n_true)) / sum_n_true) * 100
    
    rel_v_base = (jnp.sum(jnp.abs(v_coarse - v_true)) / sum_v_true) * 100
    rel_v_ml = (jnp.sum(jnp.abs(v_pred - v_true)) / sum_v_true) * 100

    # L2 Errors (log-f)
    l2_base = jnp.mean(labels_log_true**2)
    l2_ml = jnp.mean((pred_log_residuals - labels_log_true)**2)
    
    # 5. Output Scorecard
    print(f"| Metric         | Coarse Baseline | ML-Recovered | Improvement (%) | Base Rel. Err. | ML Rel. Err. |")
    print(f"|----------------|-----------------|--------------|-----------------|----------------|--------------|")
    # L2 log-f doesn't have a simple relative error in the same sense as moments
    print(f"| L2 (log-f)     | {l2_base:.6f}        | {l2_ml:.6f}     | {(1 - l2_ml/l2_base)*100:.1f}%            | -              | -            |")
    print(f"| MAE Density (n)| {err_n_base:.6f}        | {err_n_ml:.6f}     | {(1 - err_n_ml/err_n_base)*100:.1f}%            | {rel_n_base:.2f}%          | {rel_n_ml:.2f}%        |")
    print(f"| MAE Velocity (V)| {err_v_base:.6f}        | {err_v_ml:.6f}     | {(1 - err_v_ml/err_v_base)*100:.1f}%            | {rel_v_base:.2f}%          | {rel_v_ml:.2f}%        |")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantify ML model performance.")
    parser.add_argument("--config", type=str, default="baseline", help="baseline, no_grad, or all")
    args = parser.parse_args()
    
    if args.config == 'all':
        quantify_ml_improvement('baseline')
        quantify_ml_improvement('no_grad')
    else:
        quantify_ml_improvement(args.config)
