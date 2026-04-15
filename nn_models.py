"""
nn_models.py — Pure JAX Physics-ML Correction Layers (Warm Start Edition)

Provides three correction networks:
  1. Advection corrector       — depth-wise 1D CNN over the spatial axis
  2. Acceleration corrector    — field-aware MLP over (E, B, n) features [legacy]
  3. DeepONet corrector        — Branch CNN (spatial context) × Trunk MLP
                                 (velocity basis), evaluated as a matrix multiply.
                                 Uses tanh clamp for stability.

All initialization uses small-variance (1e-4) weights (Warm Start).
"""

import jax
import jax.numpy as jnp
from jax import random, nn, jit
from functools import partial

# ==========================================
# PURE JAX NEURAL NETWORK PRIMITIVES
# ==========================================

def init_layer(key, in_dim, out_dim, std=1e-4):
    """Small-variance initialization for Warm Start."""
    k1, _ = random.split(key)
    w = random.normal(k1, (in_dim, out_dim)) * std
    b = jnp.zeros(out_dim)
    return {'w': w, 'b': b}

def init_conv1d_params(key, in_channels, out_channels, kernel_size=3, std=1e-4):
    """Small-variance initialization for Warm Start."""
    k1, _ = random.split(key)
    w = random.normal(k1, (kernel_size, in_channels, out_channels)) * std
    b = jnp.zeros(out_channels)
    return {'w': w, 'b': b}

# ==========================================
# 1. ADVECTION CORRECTOR (Depth-wise 1D CNN over X)
# ==========================================

@jit
def apply_advection_correction(f, params, log_clamp=1e-2, boundary_buffer=0):
    """
    Depth-wise CNN correction for spatial advection errors.
    Zeros out corrections within the boundary_buffer zone.
    Input:  f      (nx, nv, nv, nv)
    Output: g_log  (nx, nv, nv, nv)
    """
    nx, nv, _, _ = f.shape
    f_log = jnp.log(jnp.maximum(f, 1e-12))

    f_flat = f_log.reshape(1, nx, -1)
    nv3 = f_flat.shape[-1]

    w, b = params['w'], params['b']
    dn = jax.lax.conv_dimension_numbers(f_flat.shape, w.shape, ('NHC', 'HIO', 'NHC'))
    g_log = jax.lax.conv_general_dilated(
        f_flat, w, (1,), 'SAME',
        dimension_numbers=dn,
        feature_group_count=nv3
    )
    g_log = g_log + b[None, None, :]
    g_log = nn.tanh(g_log) * log_clamp
    g_log = g_log.reshape(f.shape)

    nx = g_log.shape[0]
    idx = jnp.arange(nx)
    mask = jnp.logical_and(idx >= boundary_buffer, idx < (nx - boundary_buffer))
    mask = mask[:, None, None, None]
    g_log = g_log * mask

    return g_log

# ==========================================
# 2. ACCELERATION CORRECTOR (Field-Aware MLP)
# ==========================================

@jit
def apply_acceleration_correction(f, E_x, E_y, E_z, B_x, B_y, B_z, dt, params, 
                                  log_clamp=1e-2, boundary_buffer=0):
    """
    Learns velocity-space log-corrections conditioned on local EM fields.
    Zeros out corrections within the boundary_buffer zone.
    """
    nx, nv, _, _ = f.shape

    # Local density (scalar per cell)
    n = jnp.sum(f, axis=(1, 2, 3), keepdims=True)
    fields = jnp.stack([E_x, E_y, E_z, B_x, B_y, B_z], axis=-1)
    features = jnp.concatenate([fields, n.reshape(nx, 1)], axis=-1)  # (nx, 7)

    def mlp_step(x, p):
        h = x @ p['w1'] + p['b1']
        h = jax.nn.relu(h)
        return h @ p['w2'] + p['b2']

    g_log = mlp_step(features, params)
    g_log = jax.nn.tanh(g_log) * log_clamp
    g_log = g_log.reshape(f.shape)

    nx = g_log.shape[0]
    idx = jnp.arange(nx)
    mask = jnp.logical_and(idx >= boundary_buffer, idx < (nx - boundary_buffer))
    mask = mask[:, None, None, None]
    g_log = g_log * mask

    return g_log

# ==========================================
# 3. PARAMETER INITIALIZATION
# ==========================================

def init_network_params(key, nx, nv, kernel_size=3, accel_hidden=32, std=1e-4,
                        acc_mode='mlp', deeponet_d=32,
                        deeponet_trunk_hidden=64, deeponet_branch_hidden=32,
                        deeponet_kernel=5):
    """
    Initializes parameter trees with small variance (Warm Start).

    acc_mode:
      'mlp'      — original 7→hidden→nv³ MLP (legacy)
      'deeponet' — Branch CNN × Trunk MLP (recommended)
    """
    k1, k2, k3 = random.split(key, 3)
    nv3 = nv**3

    # Advection: depth-wise 1D CNN over X (unchanged)
    adv_params = init_conv1d_params(k1, 1, nv3, kernel_size=kernel_size, std=std)

    # Acceleration corrector
    if acc_mode == 'deeponet':
        acc_params = init_deeponet_params(
            k2, D=deeponet_d,
            trunk_hidden=deeponet_trunk_hidden,
            branch_hidden=deeponet_branch_hidden,
            branch_kernel=deeponet_kernel,
            std=std
        )
    else:  # 'mlp' — legacy
        acc_params = {
            'w1': random.normal(k2, (7, accel_hidden)) * std,
            'b1': jnp.zeros(accel_hidden),
            'w2': random.normal(k3, (accel_hidden, nv3)) * std,
            'b2': jnp.zeros(nv3),
        }

    return {'adv': adv_params, 'acc': acc_params}

def init_flux_params(key, in_channels, out_channels, kernel_size=3, std=1e-4):
    """Initialize a small-variance conv for face-based flux corrections."""
    return init_conv1d_params(key, in_channels, out_channels, kernel_size=kernel_size, std=std)

# ==========================================
# 4. DEEPONET CORRECTOR
#    Branch (1D CNN over x) × Trunk (MLP over v)
#    Output: g(x,v) = c(x) · ψ(v)
# ==========================================

def _branch_forward(features, params):
    """3-layer 1D CNN over the spatial axis."""
    x = features[None, :, :]  # (1, nx, C_in)
    for name, activate in [('c1', True), ('c2', True), ('c3', False)]:
        p = params[name]
        dn = jax.lax.conv_dimension_numbers(
            x.shape, p['w'].shape, ('NHC', 'HIO', 'NHC')
        )
        x = jax.lax.conv_general_dilated(
            x, p['w'], (1,), 'SAME', dimension_numbers=dn
        ) + p['b'][None, None, :]
        if activate:
            x = jax.nn.gelu(x)
    return x[0]  # (nx, D)

def _trunk_forward(v_triplets, params):
    """3-layer shared MLP over velocity coordinates."""
    x = jax.nn.gelu(v_triplets @ params['w1'] + params['b1'])
    x = jax.nn.gelu(x         @ params['w2'] + params['b2'])
    x = x                     @ params['w3'] + params['b3']
    return x

def init_deeponet_params(key, D=32, trunk_hidden=64, branch_hidden=32,
                         branch_kernel=5, std=1e-4):
    k1, k2, k3, k4, k5, k6 = random.split(key, 6)
    C_in = 10  # branch input channels
    branch = {
        'c1': {'w': random.normal(k1, (branch_kernel, C_in, branch_hidden)) * std,
               'b': jnp.zeros(branch_hidden)},
        'c2': {'w': random.normal(k2, (branch_kernel, branch_hidden, branch_hidden)) * std,
               'b': jnp.zeros(branch_hidden)},
        'c3': {'w': random.normal(k3, (branch_kernel, branch_hidden, D)) * std,
               'b': jnp.zeros(D)},
    }
    trunk = {
        'w1': random.normal(k4, (3, trunk_hidden)) * std,
        'b1': jnp.zeros(trunk_hidden),
        'w2': random.normal(k5, (trunk_hidden, trunk_hidden)) * std,
        'b2': jnp.zeros(trunk_hidden),
        'w3': random.normal(k6, (trunk_hidden, D)) * std,
        'b3': jnp.zeros(D),
    }
    return {'branch': branch, 'trunk': trunk}

@jit
def apply_deeponet_correction(f, E_x, E_y, E_z, B_x, B_y, B_z, dt, params,
                               log_clamp=1e-2, boundary_buffer=0, v_grid=None):
    """
    DeepONet log-correction for the acceleration sub-step.
    Bounded by tanh(g_log) * log_clamp for numerical stability.
    """
    nx, nv = f.shape[0], f.shape[1]
    dv3 = (v_grid[1] - v_grid[0]) ** 3

    vx_1d = v_grid[:, None, None]
    vy_1d = v_grid[None, :, None]
    vz_1d = v_grid[None, None, :]

    n     = jnp.sum(f, axis=(1, 2, 3)) * dv3
    n_s   = jnp.maximum(n, 1e-6)
    Vx    = jnp.sum(f * vx_1d[None], axis=(1, 2, 3)) * dv3 / n_s
    Vy    = jnp.sum(f * vy_1d[None], axis=(1, 2, 3)) * dv3 / n_s
    Vz    = jnp.sum(f * vz_1d[None], axis=(1, 2, 3)) * dv3 / n_s

    features = jnp.stack([E_x, E_y, E_z, B_x, B_y, B_z, n_s, Vx, Vy, Vz], axis=-1)
    c = _branch_forward(features, params['branch'])

    vx_g = jnp.broadcast_to(v_grid[:, None, None], (nv, nv, nv))
    vy_g = jnp.broadcast_to(v_grid[None, :, None], (nv, nv, nv))
    vz_g = jnp.broadcast_to(v_grid[None, None, :], (nv, nv, nv))
    v_triplets = jnp.stack([vx_g.ravel(), vy_g.ravel(), vz_g.ravel()], axis=-1)
    psi = _trunk_forward(v_triplets, params['trunk'])

    g_log = (c @ psi.T).reshape(nx, nv, nv, nv)
    
    if log_clamp is not None:
        g_log = jax.nn.tanh(g_log) * log_clamp

    idx = jnp.arange(nx)
    mask = jnp.logical_and(idx >= boundary_buffer, idx < (nx - boundary_buffer))
    g_log = g_log * mask[:, None, None, None]

    return g_log

@jit
def apply_advection_flux_correction(f, params, log_clamp=1e-3, boundary_buffer=0):
    nx, nv, _, _ = f.shape
    f_log = jnp.log(jnp.maximum(f, 1e-18))
    f_flat = f_log.reshape(1, nx, -1)
    nv3 = f_flat.shape[-1]
    w, b = params['w'], params['b']
    dn = jax.lax.conv_dimension_numbers(f_flat.shape, w.shape, ('NHC', 'HIO', 'NHC'))
    g_face = jax.lax.conv_general_dilated(f_flat, w, (1,), 'SAME', dimension_numbers=dn, feature_group_count=nv3)
    g_face = g_face + b[None, None, :]
    g_face = nn.tanh(g_face) * log_clamp
    g_face = g_face.reshape(f.shape)
    idx = jnp.arange(nx)
    mask = jnp.logical_and(idx >= boundary_buffer, idx < (nx - boundary_buffer))
    g_face = g_face * mask[:, None, None, None]
    return g_face
