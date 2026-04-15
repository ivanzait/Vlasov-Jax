import os, sys
sys.path.insert(0, os.getcwd())
import jax, jax.numpy as jnp
from initialize_maxwell import initialize_simulation
from nn_models import init_network_params, apply_advection_flux_correction
from train_corrector import loss_fn
import config

# ensure flux-correction mode
config.ML_ADVECT_FLUX_CORRECTION = True
config.ML_USE_CNN = True

sim = initialize_simulation()
f = sim['f']
B_x,B_y,B_z = sim['B']
E_x,E_y,E_z = sim['E']
x,v,dx,dv,V_sq = sim['grids']

nx = len(x)
nv = len(v)
dt = sim['dt']

key = jax.random.PRNGKey(999)
ml_params = init_network_params(key, nx, nv, kernel_size=config.ML_ADVECTION_KERNEL, accel_hidden=config.ML_ACCEL_HIDDEN, std=1e-4)

# load a fine snapshot and coarsen
fine_path = 'data/fine_data/snapshot_00010.npz'
if not os.path.exists(fine_path):
    print('fine snapshot not found:', fine_path)
    sys.exit(1)

from train_corrector import load_training_pair
f_fine_c, B_f, E_f = load_training_pair(fine_path, nx, nv)

# Compute g_face and fluxes
g_face = apply_advection_flux_correction(f, ml_params['adv'], log_clamp=config.ML_ADVECTION_CLAMP, boundary_buffer=config.ML_BOUNDARY_BUFFER)

vx_grid = v[None, :, None, None]
f_right = jnp.roll(f, -1, axis=0)
f_face_upwind = jnp.where(vx_grid > 0, f, f_right)
flux_face_base = vx_grid * f_face_upwind
flux_face_corr = flux_face_base * jnp.exp(g_face)
f_after = f - (dt / dx) * (flux_face_corr - jnp.roll(flux_face_corr, 1, axis=0))

print('g_face stats:', float(jnp.min(g_face)), float(jnp.max(g_face)), float(jnp.mean(g_face)))
print('flux_face_base stats:', float(jnp.min(flux_face_base)), float(jnp.max(flux_face_base)))
print('flux_face_corr stats:', float(jnp.min(flux_face_corr)), float(jnp.max(flux_face_corr)))
print('f_after stats:', float(jnp.min(f_after)), float(jnp.max(f_after)), float(jnp.mean(f_after)))

# compute loss and grads
from jax import value_and_grad
loss_val, grads = value_and_grad(lambda p: loss_fn({'adv': p, 'acc': ml_params['acc']}, f, f_fine_c, dt, config.ML_ADVECTION_CLAMP, config.ML_ACCELERATION_CLAMP, config.ML_BOUNDARY_BUFFER, dx, v, B_f, E_f)) (ml_params['adv'])
print('loss:', float(loss_val))

# compute grad norms (max abs)
import numpy as np
flat_grads = jax.tree_util.tree_flatten(grads)[0]
max_grads = [float(jnp.max(jnp.abs(g))) for g in flat_grads]
print('max grad values:', max_grads[:5])

# compute proposed adam delta on first adv weight slice (approx, no m/v state)
lr = config.ML_LR_CNN
approx_updates = [lr * g for g in flat_grads]
max_updates = [float(jnp.max(jnp.abs(u))) for u in approx_updates]
print('approx max updates:', max_updates[:5])
