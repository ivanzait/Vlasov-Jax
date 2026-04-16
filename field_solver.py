import jax
import jax.numpy as jnp
from jax import jit
from functools import partial

@partial(jit, static_argnums=(2, 3))
def get_gradient(arr, dx, bc_left='periodic', bc_right='periodic'):
    """
    Unified central difference leveraging the 2-cell ghost padding.
    The ghost cells (applied in boundary.py) allow this stencil to remain 
    accurate throughout the entire physical domain.
    """
    # Uses jnp.roll for efficiency; boundary errors at the literal array edges 
    # (indices 0 and -1) are transparent to the physical domain.
    return (jnp.roll(arr, -1, axis=0) - jnp.roll(arr, 1, axis=0)) / (2 * dx)


@jit
def get_moments(f, v, dv):
    """
    Extracts n, V, and T from the distribution function f.
    Functional implementation.
    """
    n = jnp.sum(f, axis=(1, 2, 3)) * (dv**3)
    n_safe = jnp.maximum(n, 1e-6)
    
    vx_grid = v[None, :, None, None]
    vy_grid = v[None, None, :, None]
    vz_grid = v[None, None, None, :]
    
    Vi_x = jnp.sum(f * vx_grid, axis=(1, 2, 3)) * (dv**3) / n_safe
    Vi_y = jnp.sum(f * vy_grid, axis=(1, 2, 3)) * (dv**3) / n_safe
    Vi_z = jnp.sum(f * vz_grid, axis=(1, 2, 3)) * (dv**3) / n_safe
    
    # Temperature T: P = n*T = m * \int (v - V)^2 f d3v
    v_sq_avg = jnp.sum(f * (vx_grid**2 + vy_grid**2 + vz_grid**2), axis=(1, 2, 3)) * (dv**3) / n_safe
    V_mag_sq = Vi_x**2 + Vi_y**2 + Vi_z**2
    T = (v_sq_avg - V_mag_sq) / 3.0 # Assuming isotropic T
    
    return n, Vi_x, Vi_y, Vi_z, T


@partial(jit, static_argnums=(5,))
def get_fields(f, B_x, B_y, B_z, dx, bc_x, v, dv):
    """
    Calculates E via Ohm's law and updates J under full 3D B fields.
    Functional implementation.
    """
    n_i, Vi_x, Vi_y, Vi_z, T = get_moments(f, v, dv)
    n_e = jnp.maximum(n_i, 1e-6)
    
    dBy_dx = get_gradient(B_y, dx, bc_x[0], bc_x[1])
    dBz_dx = get_gradient(B_z, dx, bc_x[0], bc_x[1])
    
    J_x = jnp.zeros_like(B_x) 
    J_y = -dBz_dx
    J_z = dBy_dx
    
    VixB_x = Vi_y * B_z - Vi_z * B_y
    VixB_y = Vi_z * B_x - Vi_x * B_z
    VixB_z = Vi_x * B_y - Vi_y * B_x
    
    JxB_x = J_y * B_z - J_z * B_y
    JxB_y = J_z * B_x - J_x * B_z
    JxB_z = J_x * B_y - J_y * B_x
    
    E_x = -VixB_x + JxB_x / n_e
    E_y = -VixB_y + JxB_y / n_e
    E_z = -VixB_z + JxB_z / n_e
    
    return E_x, E_y, E_z, J_x, J_y, J_z


@partial(jit, static_argnums=(6,))
def advance_magnetic_field(B_x, B_y, B_z, E_y, E_z, dx, bc_x, dt):
    """ Faraday's law: partial(B)/partial(t) = -curl(E) """
    dEy_dx = get_gradient(E_y, dx, bc_x[0], bc_x[1])
    dEz_dx = get_gradient(E_z, dx, bc_x[0], bc_x[1])
    
    # B_x is constant in 1D simulations
    B_y_new = B_y + dt * dEz_dx
    B_z_new = B_z - dt * dEy_dx
    
    return B_x, B_y_new, B_z_new
