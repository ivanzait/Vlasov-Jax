"""
train_corrector.py — Online Training Utilities for VLSV-JAX

This module provide the Adam optimizer and training loop.
Updated to handle decoupled ADVECTION and ACCELERATION clamps.
"""

import os
import jax
import jax.numpy as jnp
from jax import jit, value_and_grad
from functools import partial
import numpy as np

from nn_models import (apply_advection_correction, apply_acceleration_correction,
                       apply_advection_flux_correction, apply_deeponet_correction)
from data_io import load_snapshot
import config


# ==========================================
# 1. ADAM OPTIMIZER (Native JAX)
# ==========================================

def init_adam_state(params):
    """Initialize first and second moment buffers for Adam."""
    m = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), params)
    v = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), params)
    return m, v

@jit
def adam_update(params, grads, m, v, t, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
    """Bias-corrected Adam parameter update."""
    # Optional global gradient clipping (by norm) to stabilize updates.
    clip = getattr(config, 'ML_GRAD_CLIP_NORM', None)
    if clip is not None and clip > 0:
        leaves = jax.tree_util.tree_leaves(grads)
        sq = 0.0
        for g in leaves:
            sq = sq + jnp.sum(g**2)
        gnorm = jnp.sqrt(sq)
        # scale = min(1, clip / (gnorm + tiny))
        tiny = 1e-12
        scale = jnp.minimum(1.0, clip / (gnorm + tiny))
        grads = jax.tree_util.tree_map(lambda x: x * scale, grads)

    t = t + 1
    m = jax.tree_util.tree_map(lambda m, g: b1 * m + (1 - b1) * g, m, grads)
    v = jax.tree_util.tree_map(lambda v, g: b2 * v + (1 - b2) * (g**2), v, grads)

    m_hat = jax.tree_util.tree_map(lambda m: m / (1 - b1**t), m)
    v_hat = jax.tree_util.tree_map(lambda v: v / (1 - b2**t), v)

    params = jax.tree_util.tree_map(
        lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps),
        params, m_hat, v_hat
    )
    return params, m, v


# ==========================================
# 2. DATA UTILITIES
# ==========================================

def coarsen_data(f_fine, nx_coarse, nv_coarse=None):
    """Block-averages fine-resolution data onto a coarse grid."""
    nx_fine = f_fine.shape[0]
    if nx_fine != nx_coarse:
        factor_x = nx_fine // nx_coarse
        f_fine = f_fine.reshape(nx_coarse, factor_x, *f_fine.shape[1:]).mean(axis=1)

    if nv_coarse is not None:
        nv_fine = f_fine.shape[1]
        if nv_fine != nv_coarse:
            factor_v = nv_fine // nv_coarse
            f_fine = f_fine.reshape(
                nx_coarse,
                nv_coarse, factor_v,
                nv_coarse, factor_v,
                nv_coarse, factor_v
            ).mean(axis=(2, 4, 6))

    return f_fine


def load_training_pair(snapshot_path, nx_coarse, nv_coarse=None):
    """Loads a fine snapshot and coarse-grains it for comparison."""
    data = load_snapshot(snapshot_path)
    f_coarsened = coarsen_data(data['f'], nx_coarse, nv_coarse)
    return f_coarsened, data['B'], data['E']


# ==========================================
# 3. TRAINING CORE
# ==========================================

def loss_fn(params, f_coarse, f_fine_coarsened, dt, adv_clamp, acc_clamp, boundary_buffer, dx, v, B=None, E=None):
    """
    Decoupled sequential log-MSE loss for advection and acceleration correctors.

    Strategy:
      1. Advection (CNN/flux-CNN): loss vs. the full target log(f_fine / f_coarse).
         f_after_adv is computed from the CNN output (flux or multiplicative).
      2. Acceleration (MLP): loss vs. the *residual* log(f_fine / f_after_adv),
         i.e. what the MLP still needs to fix after the advection corrector.
         jax.lax.stop_gradient on f_after_adv decouples the gradient flows so
         MLP gradients never backprop through CNN weights.

    When ML_ADVECT_FLUX_CORRECTION=True the flux CFL uses dt/2 to match the
    half-step sub-step in strang_step.
    """
    eps = 1e-12
    dt_half = dt / 2.0  # flux correction lives inside a dt/2 advection sub-step

    # Boundary mask -- shared between all sub-losses
    nx = f_coarse.shape[0]
    idx     = jnp.arange(nx)
    mask    = jnp.logical_and(idx >= boundary_buffer, idx < (nx - boundary_buffer))
    mask4d  = mask[:, None, None, None]
    
    # Total number of 4D points used in the loss sum
    nv = f_coarse.shape[1]
    n_interior = jnp.sum(mask4d.astype(jnp.float32)) * (nv ** 3)
    eps     = 1e-10
    
    use_cnn  = getattr(config, 'ML_USE_CNN',  True)
    use_flux = getattr(config, 'ML_ADVECT_FLUX_CORRECTION', False)
    use_mlp  = getattr(config, 'ML_USE_MLP',  True)

    total_loss  = jnp.array(0.0)
    f_after_adv = f_coarse  # identity fallback if CNN is off

    # ------------------------------------------------------------------
    # 1. Advection correction loss — CNN targets full log(f_fine/f_coarse)
    # ------------------------------------------------------------------
    if use_cnn:
        target_g_adv = jnp.log(jnp.maximum(f_fine_coarsened, eps)) \
                     - jnp.log(jnp.maximum(f_coarse, eps))

        if use_flux:
            # Face flux correction — CFL uses dt_half to match strang_step
            g_face = apply_advection_flux_correction(
                f_coarse, params['adv'], log_clamp=adv_clamp,
                boundary_buffer=boundary_buffer)
            vx_grid   = v[None, :, None, None]
            f_right   = jnp.roll(f_coarse, -1, axis=0)
            f_upwind  = jnp.where(vx_grid > 0, f_coarse, f_right)
            flux_base = vx_grid * f_upwind
            flux_corr = flux_base * jnp.exp(g_face)
            f_after_adv = f_coarse - (dt_half / dx) * (
                flux_corr - jnp.roll(flux_corr, 1, axis=0))
            g_adv = jnp.log(jnp.maximum(f_after_adv, 1e-18)) \
                  - jnp.log(jnp.maximum(f_coarse, 1e-18))
        else:
            # Multiplicative cell-wise correction
            g_adv = apply_advection_correction(
                f_coarse, params['adv'], log_clamp=adv_clamp,
                boundary_buffer=boundary_buffer)
            f_after_adv = f_coarse * jnp.exp(g_adv)

        loss_adv   = jnp.sum((g_adv - target_g_adv) ** 2 * mask4d) / n_interior
        total_loss = total_loss + loss_adv

    # ------------------------------------------------------------------
    # 2. Acceleration correction loss — MLP / DeepONet targets residual after adv
    # ------------------------------------------------------------------
    if use_mlp and B is not None and E is not None:
        # Stop gradient: acc corrector gradients must not flow back through CNN weights.
        f_adv_sg     = jax.lax.stop_gradient(f_after_adv)
        target_g_acc = jnp.log(jnp.maximum(f_fine_coarsened, eps)) \
                     - jnp.log(jnp.maximum(f_adv_sg, eps))

        B_x, B_y, B_z = B
        E_x, E_y, E_z = E

        acc_mode = getattr(config, 'ML_ACC_MODE', 'mlp')
        if acc_mode == 'deeponet':
            # v is already threaded through from maybe_train_step — used as v_grid
            g_acc = apply_deeponet_correction(
                f_adv_sg, E_x, E_y, E_z, B_x, B_y, B_z, dt, params['acc'],
                log_clamp=acc_clamp, boundary_buffer=boundary_buffer, v_grid=v)
        else:
            g_acc = apply_acceleration_correction(
                f_adv_sg, E_x, E_y, E_z, B_x, B_y, B_z, dt, params['acc'],
                log_clamp=acc_clamp, boundary_buffer=boundary_buffer)

        loss_acc   = jnp.sum((g_acc - target_g_acc) ** 2 * mask4d) / n_interior
        total_loss = total_loss + loss_acc

    # ------------------------------------------------------------------
    # 3. Optional mass-conservation penalty (applied on fully corrected state)
    # ------------------------------------------------------------------
    mass_weight = getattr(config, 'ML_MASS_LOSS_WEIGHT', 0.0)
    if mass_weight and mass_weight > 0.0:
        # Build best estimate of the fully corrected state
        if use_mlp and B is not None and E is not None:
            f_pred = jax.lax.stop_gradient(f_after_adv) * jnp.exp(g_acc)
        else:
            f_pred = f_after_adv
        mass_coarse = jnp.sum(f_coarse * mask4d)
        mass_pred   = jnp.sum(f_pred   * mask4d)
        rel_change  = (mass_pred - mass_coarse) / jnp.maximum(jnp.abs(mass_coarse), eps)
        total_loss  = total_loss + (rel_change ** 2) * mass_weight

    return total_loss


@jit
def train_step(params, m_adam, v_adam, t, f_coarse, f_fine_coarsened, dt, lr_adv, lr_acc, adv_clamp, acc_clamp, boundary_buffer, dx, v, B=None, E=None):
    """Single Adam update step with separate learning rates for adv (CNN) and acc (MLP)."""
    loss, grads = value_and_grad(loss_fn)(
        params, f_coarse, f_fine_coarsened, dt, adv_clamp, acc_clamp, boundary_buffer, dx, v, B, E
    )

    # Update advection (CNN) subtree (skip if disabled)
    if getattr(config, 'ML_USE_CNN', True):
        params_adv, m_adv, v_adv = adam_update(params['adv'], grads['adv'], m_adam['adv'], v_adam['adv'], t, lr=lr_adv)
    else:
        params_adv, m_adv, v_adv = params['adv'], m_adam['adv'], v_adam['adv']

    # Update acceleration (MLP) subtree (skip if disabled)
    if getattr(config, 'ML_USE_MLP', True):
        params_acc, m_acc, v_acc = adam_update(params['acc'], grads['acc'], m_adam['acc'], v_adam['acc'], t, lr=lr_acc)
    else:
        params_acc, m_acc, v_acc = params['acc'], m_adam['acc'], v_adam['acc']

    # Reassemble full trees
    params = {'adv': params_adv, 'acc': params_acc}
    m_out = {'adv': m_adv, 'acc': m_acc}
    v_out = {'adv': v_adv, 'acc': v_acc}

    return params, m_out, v_out, loss


# ==========================================
# 4. ONLINE TRAINING HOOK
# ==========================================

def maybe_train_step(step, f, B, E, ml_params, m_adam, v_adam, t_adam,
                     fine_dir, nx, nv, dt, nt, lr_adv, lr_acc, adv_clamp, acc_clamp, boundary_buffer,
                     dx=None, v=None):
    """Conditionally fires an online training update."""
    fine_path = os.path.join(fine_dir, f"snapshot_{step:05d}.npz")

    if not os.path.exists(fine_path):
        return ml_params, m_adam, v_adam, t_adam

    f_fine_c, _, _ = load_training_pair(fine_path, nx, nv)
    # Number of Adam updates to perform per saved snapshot (can be configured)
    updates = getattr(config, 'ML_UPDATES_PER_SAVE', 1)
    last_loss = None
    for _ in range(max(1, int(updates))):
        ml_params, m_adam, v_adam, loss = train_step(
            ml_params, m_adam, v_adam, t_adam,
            f, f_fine_c, dt, lr_adv, lr_acc, adv_clamp, acc_clamp, boundary_buffer, dx, v, B=B, E=E
        )
        t_adam += 1
        last_loss = loss
        print(f" [ML-Train] Step {step:04d} | Loss: {loss:.10f} | Updates: {t_adam}")
    # print(f" [ML-Train] Step {step:04d} | Loss: {last_loss:.10f} | Updates: {updates}")
    return ml_params, m_adam, v_adam, t_adam
