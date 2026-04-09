import jax.numpy as jnp


def setup_two_stream(X, VX, VY, VZ, v_drift, vth_beam, k_pert, pert_amp):
    """
    Initial condition for the Two-Stream Instability.
    Consists of two counter-streaming Maxwellians with a spatial perturbation.
    """
    
    # Two counter-streaming Maxwellians background
    f_maxwell_1 =  jnp.exp(-((VX - v_drift)**2 + VY**2 + VZ**2) / (2 * vth_beam**2)) / ((2 * jnp.pi)**1.5 * vth_beam**3)
    f_maxwell_2 =  jnp.exp(-((VX + v_drift)**2 + VY**2 + VZ**2) / (2 * vth_beam**2)) / ((2 * jnp.pi)**1.5 * vth_beam**3)
    f_background = (f_maxwell_1 + f_maxwell_2)
    
    # Apply density perturbation across the domain
    f = f_background * (1.0 + pert_amp * jnp.cos(k_pert * X))
    return f
