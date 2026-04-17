import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
from .state import SimulationState

@partial(jit, static_argnums=(6,))
def apply_bc(f, B_y, B_z, E_x, E_y, E_z, bc_x, static_data=None):
    """
    Explicitly enforces boundary conditions on the fields and distribution function.
    Handles separate left/right spatial boundaries using 2-cell ghost layers.
    Includes Electric Field synchronization.
    """
    # 1. Left Boundary Ghost Cells ([0, 1])
    if bc_x[0] == 'copy':
        B_y = B_y.at[0:2].set(B_y[2:4])
        B_z = B_z.at[0:2].set(B_z[2:4])
        f = f.at[0:2].set(f[2:4])
        E_x = E_x.at[0:2].set(E_x[2:4])
        E_y = E_y.at[0:2].set(E_y[2:4])
        E_z = E_z.at[0:2].set(E_z[2:4])
    elif bc_x[0] == 'static' and static_data is not None:
        B_y = B_y.at[0:2].set(static_data['By_left'])
        B_z = B_z.at[0:2].set(static_data['Bz_left'])
        f = f.at[0:2].set(static_data['f_left'])
        E_x = E_x.at[0:2].set(static_data['Ex_left'])
        E_y = E_y.at[0:2].set(static_data['Ey_left'])
        E_z = E_z.at[0:2].set(static_data['Ez_left'])

    # 2. Right Boundary Ghost Cells ([-2, -1])
    if bc_x[1] == 'copy':
        B_y = B_y.at[-2:].set(B_y[-4:-2])
        B_z = B_z.at[-2:].set(B_z[-4:-2])
        f = f.at[-2:].set(f[-4:-2])
        E_x = E_x.at[-2:].set(E_x[-4:-2])
        E_y = E_y.at[-2:].set(E_y[-4:-2])
        E_z = E_z.at[-2:].set(E_z[-4:-2])
    elif bc_x[1] == 'static' and static_data is not None:
        B_y = B_y.at[-2:].set(static_data['By_right'])
        B_z = B_z.at[-2:].set(static_data['Bz_right'])
        f = f.at[-2:].set(static_data['f_right'])
        E_x = E_x.at[-2:].set(static_data['Ex_right'])
        E_y = E_y.at[-2:].set(static_data['Ey_right'])
        E_z = E_z.at[-2:].set(static_data['Ez_right'])
            
    return f, B_y, B_z, E_x, E_y, E_z


@partial(jit, static_argnums=(1,))
def synchronize_state_ghosts(state: SimulationState, bc_x, static_data=None) -> SimulationState:
    """
    Specialized ghost-cell synchronization for SimulationState Pytree.
    Updates f, B, and E fields.
    """
    f, B_y, B_z, E_x, E_y, E_z = apply_bc(
        state.f, state.B_y, state.B_z, 
        state.E_x, state.E_y, state.E_z, 
        bc_x, static_data
    )
    return SimulationState(f, state.B_x, B_y, B_z, E_x, E_y, E_z)
