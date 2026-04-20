import jax
import jax.numpy as jnp
import optax
from jax import random, jit
from functools import partial

class MLP:
    """
    Architecture: Input (32780) -> Multi-Layer Hidden -> Output (32768)
    Inputs: [f_logs (32768), E (3), B (3), dE/dx (3), dB/dx (3)]
    Output: delta_log_f (32768)
    """
    def __init__(self, key, input_dim, hidden_dims, output_dim):
        """
        hidden_dims: list of hidden layer sizes (e.g., [256, 256])
        """
        keys = random.split(key, len(hidden_dims) + 1)
        
        def init_layer(key, d_in, d_out):
            k1, k2 = random.split(key)
            scale = jnp.sqrt(2.0 / (d_in + d_out))
            w = random.normal(k1, (d_in, d_out)) * scale
            b = jnp.zeros(d_out)
            return w, b

        dims = [input_dim] + list(hidden_dims) + [output_dim]
        self.params = []
        for i in range(len(dims) - 1):
            self.params.append(init_layer(keys[i], dims[i], dims[i+1]))

    @staticmethod
    def forward(params, x):
        """
        Deeper forward pass with Residual (Skip) Connections.
        Identifies layers with matching dimensions and applies identity skips.
        """
        h = x
        for i in range(len(params) - 1):
            w, b = params[i]
            
            # Linear + Activation
            z = jnp.dot(h, w) + b
            h = jax.nn.gelu(z)
            
            # (Residual Connection Disabled for Baseline Study)
            # if h.shape[-1] == w.shape[1]:
            #     h = z + h
            # else:
            #     h = z
        
        # Output layer (Linear)
        w_out, b_out = params[-1]
        return jnp.dot(h, w_out) + b_out

def get_n_v_from_f(f_flat, v, dv):
    """
    Calculates density and 1D velocity from a flattened distribution,
    dynamically deriving dimensions from the velocity grid.
    """
    nv = len(v)
    f = f_flat.reshape(-1, nv, nv, nv)
    n = jnp.sum(f, axis=(1, 2, 3)) * (dv**3)
    n_safe = jnp.maximum(n, 1e-6)
    
    vx_grid = v[None, :, None, None]
    vi_x = jnp.sum(f * vx_grid, axis=(1, 2, 3)) * (dv**3) / n_safe
    return n, vi_x

def get_velocity_weight_mask(v, v_scale=4.0):
    """
    Generates a weight mask W(v) = 1 + (v^2 / v_scale^2) to emphasize tails.
    """
    vx = v[:, None, None]
    vy = v[None, :, None]
    vz = v[None, None, :]
    v_sq = vx**2 + vy**2 + vz**2
    return (1.0 + v_sq / v_scale**2).reshape(-1)

@jit
def physics_loss_fn(params, x, y_target_log, v, dv, lambda_phys=1.0, v_scale=4.0):
    """
    Weighted Loss = Weighted_MSE(log_f) + lambda * (MSE(n) + MSE(v))
    """
    # 1. Distribution Matching (Log-Space) with Velocity Weighting
    pred_log_residual = MLP.forward(params, x)
    
    w_mask = get_velocity_weight_mask(v, v_scale)
    weighted_diff_sq = w_mask * (pred_log_residual - y_target_log)**2
    mse_log_weighted = jnp.mean(weighted_diff_sq)
    
    # 2. Physics Matching (Moment-Consistency)
    nv = len(v)
    f_coarse_flat = x[:, :nv**3]
    f_pred = f_coarse_flat * jnp.exp(pred_log_residual)
    f_target = f_coarse_flat * jnp.exp(y_target_log)
    
    n_pred, v_pred = get_n_v_from_f(f_pred, v, dv)
    n_target, v_target = get_n_v_from_f(f_target, v, dv)
    
    mse_phys = jnp.mean((n_pred - n_target)**2) + jnp.mean((v_pred - v_target)**2)
    
    return mse_log_weighted + lambda_phys * mse_phys

@partial(jit, static_argnums=(4,))
def update_physics(params, x, y, opt_state, optimizer, v, dv, lambda_phys=1.0, v_scale=4.0):
    """
    Update step using the weighted physics loss.
    """
    loss, grads = jax.value_and_grad(physics_loss_fn)(params, x, y, v, dv, lambda_phys, v_scale)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss
