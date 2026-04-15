import os, sys
sys.path.insert(0, os.getcwd())
import jax, jax.numpy as jnp
from initialize_maxwell import initialize_simulation
from nn_models import init_network_params
from train_corrector import init_adam_state, train_step
import config

# initialize
sim = initialize_simulation()
f = sim['f']
B_x,B_y,B_z = sim['B']
E_x,E_y,E_z = sim['E']
x,v,dx,dv,V_sq = sim['grids']

nx = len(x)
nv = len(v)
dt = sim['dt']

key = jax.random.PRNGKey(123)
ml_params = init_network_params(key, nx, nv, kernel_size=config.ML_ADVECTION_KERNEL, accel_hidden=config.ML_ACCEL_HIDDEN, std=1e-4)
m_adam, v_adam = init_adam_state(ml_params)

# run K updates and print loss each time
K = int(os.environ.get('K', '5'))
print('Running', K, 'updates')
for k in range(K):
    ml_params, m_adam, v_adam, loss = train_step(ml_params, m_adam, v_adam, k, f, f, dt, config.ML_LR_CNN, config.ML_LR_MLP, config.ML_ADVECTION_CLAMP, config.ML_ACCELERATION_CLAMP, config.ML_BOUNDARY_BUFFER, B=(B_x,B_y,B_z), E=(E_x,E_y,E_z))
    print(f'Update {k+1}/{K}: loss={float(loss)}')

# show param change magnitude
import numpy as np
flat_before = jax.tree_util.tree_flatten(init_network_params(jax.random.PRNGKey(123), nx, nv, kernel_size=config.ML_ADVECTION_KERNEL, accel_hidden=config.ML_ACCEL_HIDDEN, std=1e-4))[0]
flat_after = jax.tree_util.tree_flatten(ml_params)[0]
changes = [np.max(np.abs(np.array(a)-np.array(b))) for a,b in zip(flat_before, flat_after)]
print('Max param change after all updates:', max(changes))
