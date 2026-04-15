
import jax
import jax.numpy as jnp
from jax import random
import os
import sys

# Add current directory to path to import local modules
sys.path.append(os.getcwd())

# Dummy config setup
import config_train as config
sys.modules['config'] = config

from nn_models import (init_network_params, apply_deeponet_correction, 
                       init_deeponet_params, _branch_forward, _trunk_forward)
from train_corrector import loss_fn

def count_params(tree):
    return sum(x.size for x in jax.tree_util.tree_leaves(tree))

def test_deeponet_standalone():
    print("--- 1. Testing Standalone DeepONet ---")
    nx, nv = 64, 32
    D = 32
    key = random.PRNGKey(42)
    
    # Test initialization with larger scale for numerical stability in test
    params = init_deeponet_params(key, D=D, std=0.01)
    acc_params_count = count_params(params)
    print(f"DeepONet Acc Params: {acc_params_count}")

    # Test forward pass shape with non-uniform inputs
    # Add some variation to avoid constant-input artifacts
    x_noise = jnp.sin(jnp.linspace(0, 4*jnp.pi, nx))
    f = jnp.ones((nx, nv, nv, nv)) * 0.1 * (1.1 + x_noise[:, None, None, None])
    E = jnp.zeros(nx) + 0.1 * x_noise
    B = jnp.ones(nx) + 0.05 * x_noise
    v_grid = jnp.linspace(-5, 5, nv)
    dt = 0.05
    
    g_log = apply_deeponet_correction(
        f, E, E, E, B, B, B, dt, params, 
        boundary_buffer=2, v_grid=v_grid
    )
    
    print(f"Output shape: {g_log.shape}")
    print(f"Output stats: min={jnp.min(g_log):.4e}, max={jnp.max(g_log):.4e}, mean={jnp.mean(g_log):.4e}")
    assert g_log.shape == (nx, nv, nv, nv)
    
    # Test boundary buffer
    print(f"Boundary values (buffer zone max abs):")
    print(f"  Left (idx 0:2): {jnp.max(jnp.abs(g_log[0:2])):.4e}")
    print(f"  Right (idx -2:): {jnp.max(jnp.abs(g_log[-2:])):.4e}")
    assert jnp.all(g_log[0:2] == 0)
    assert jnp.all(g_log[-2:] == 0)
    
    # Test clamping specifically
    print("--- 1b. Testing Clamping ---")
    large_params = jax.tree_util.tree_map(lambda x: x * 1000.0, params)
    g_clamped = apply_deeponet_correction(
        f, E, E, E, B, B, B, dt, large_params, 
        log_clamp=1e-5, boundary_buffer=2, v_grid=v_grid
    )
    max_val = jnp.max(jnp.abs(g_clamped))
    print(f"Max value with log_clamp=1e-5: {max_val:.4e}")
    assert max_val <= 1.01e-5 # Allow slight epsilon
    
    # Test gradient flow
    def simple_loss(p):
        out = apply_deeponet_correction(f, E, E, E, B, B, B, dt, p, boundary_buffer=2, v_grid=v_grid)
        return jnp.sum(out**2)
        
    grads = jax.grad(simple_loss)(params)
    branch_grad = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads['branch'])))
    trunk_grad = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads['trunk'])))
    print(f"Branch Gradient Norm: {branch_grad:.4e}")
    print(f"Trunk Gradient Norm: {trunk_grad:.4e}")
    
    assert branch_grad > 0
    assert trunk_grad > 0
    print("Standalone DeepONet Test: SUCCESS\n")

def test_loss_function():
    print("--- 2. Testing Loss Function with DeepONet ---")
    nx, nv = 64, 32
    key = random.PRNGKey(0)
    
    # Force config settings
    config.ML_ACC_MODE = 'deeponet'
    config.ML_USE_CNN = False
    config.ML_USE_MLP = True
    print(f"DEBUG: config.ML_ACC_MODE = {config.ML_ACC_MODE}")
    
    # Initialize params with large std to see change
    params = init_network_params(key, nx, nv, acc_mode='deeponet', std=0.01)
    
    # Use non-uniform inputs to avoid zero-grad if some symmetry exists
    x_noise = jnp.sin(jnp.linspace(0, 4*jnp.pi, nx))
    f_coarse = jnp.ones((nx, nv, nv, nv)) * 0.5 * (1.1 + x_noise[:, None, None, None])
    f_fine = f_coarse * 1.05
    v = jnp.linspace(-5, 5, nv)
    # Fields also non-uniform
    B = (jnp.ones(nx) + 0.1*x_noise, jnp.ones(nx), jnp.ones(nx))
    E = (jnp.zeros(nx) + 0.05*x_noise, jnp.zeros(nx), jnp.zeros(nx))
    
    def debug_loss(p):
        B_t = (B[0], B[1], B[2])
        E_t = (E[0], E[1], E[2])
        return loss_fn(p, f_coarse, f_fine, dt=0.05, adv_clamp=0.02, acc_clamp=None, 
                      boundary_buffer=2, dx=0.45, v=v, B=B_t, E=E_t)

    loss = debug_loss(params)
    print(f"Initial loss: {float(loss):.8f}")
    
    # Expected log(1.05)^2 ~= 0.00238
    
    # Run gradient step
    grad_fn = jax.value_and_grad(debug_loss)
    loss_val, grads = grad_fn(params)
    
    acc_grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads['acc'])))
    print(f"Acc Gradient Norm: {acc_grad_norm:.4e}")
                             
    # Simple SGD step
    lr = 0.01 # Smaller LR for stability
    new_params = jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)
    
    new_loss = debug_loss(new_params)
                      
    print(f"Loss after one SGD step (lr={lr}): {float(new_loss):.8f}")
    print(f"Improvement: {float(loss - new_loss):.8f}")
    
    if loss > 1.0:
        print("WARNING: Loss is unexpectedly high (> 1.0)")
        
    assert new_loss < loss
    print("Loss Function Test: SUCCESS\n")

if __name__ == "__main__":
    try:
        test_deeponet_standalone()
        test_loss_function()
        print(">>> ALL DEEPONET TESTS PASSED <<<")
    except Exception as e:
        print(f"\n>>> TEST FAILED: {str(e)} <<<")
        import traceback
        traceback.print_exc()
        sys.exit(1)
