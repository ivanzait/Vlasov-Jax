import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import jax, jax.numpy as jnp
from initialize_maxwell import initialize_simulation
from nn_models import init_network_params, apply_advection_correction, apply_acceleration_correction
from train_corrector import init_adam_state, train_step
import config

sim = initialize_simulation()
f = sim['f']
B_x,B_y,B_z = sim['B']
E_x,E_y,E_z = sim['E']
x,v,dx,dv,V_sq = sim['grids']

nx = len(x)
nv = len(v)
dt = sim['dt']

key = jax.random.PRNGKey(0)
ml_params = init_network_params(key, nx, nv, kernel_size=config.ML_ADVECTION_KERNEL, accel_hidden=config.ML_ACCEL_HIDDEN, std=1e-4)

# compute masses
total_mass = float(jnp.sum(f) * (dx * dv**3))
print('Initial total mass:', total_mass)

# Apply adv correction
adv_clamp = config.ML_ADVECTION_CLAMP
buffer = config.ML_BOUNDARY_BUFFER
g_adv = apply_advection_correction(f, ml_params['adv'], log_clamp=adv_clamp, boundary_buffer=buffer)
mean_g_adv = jnp.mean(g_adv, axis=(1,2,3))
print('adv mean g per-x (first 5):', mean_g_adv[:5])

f_adv = f * jnp.exp(g_adv)
mass_after_adv = float(jnp.sum(f_adv) * (dx * dv**3))
print('After adv total mass:', mass_after_adv, 'delta:', mass_after_adv - total_mass)

# Apply acc correction
acc_clamp = config.ML_ACCELERATION_CLAMP
g_acc = apply_acceleration_correction(f, E_x, E_y, E_z, B_x, B_y, B_z, dt, ml_params['acc'], log_clamp=acc_clamp, boundary_buffer=buffer)
mean_g_acc = jnp.mean(g_acc, axis=(1,2,3))
print('acc mean g per-x (first 5):', mean_g_acc[:5])

f_acc = f * jnp.exp(g_acc)
mass_after_acc = float(jnp.sum(f_acc) * (dx * dv**3))
print('After acc total mass:', mass_after_acc, 'delta:', mass_after_acc - total_mass)

# Training step with f_fine_coarsened == f (should give zero loss)
m_adam, v_adam = init_adam_state(ml_params)
params_before = ml_params
params_after, m_after, v_after, loss = train_step(params_before, m_adam, v_adam, 0, f, f, dt, config.ML_LR_CNN, config.ML_LR_MLP, adv_clamp, acc_clamp, buffer)
print('Training loss (should be ~0):', float(loss))

# Check param changes
import numpy as np
flat_before = jax.tree_util.tree_flatten(params_before)[0]
flat_after = jax.tree_util.tree_flatten(params_after)[0]
changes = [np.max(np.abs(np.array(a)-np.array(b))) for a,b in zip(flat_before, flat_after)]
print('Max param change after zero-target train step:', max(changes))
