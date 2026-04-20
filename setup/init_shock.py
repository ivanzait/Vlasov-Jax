import jax.numpy as jnp


# Derived from plasma_calculator.ipynb constraints
shock_dict_normalized = {
    'n_up': 0.29069767,
    'n_down': 1.0,
    'B_up': jnp.array([0.26655629, 0.0, 0.26655629]),
    'B_down': jnp.array([0.26655629, 0.0, 0.96381935]),
    'V_up': jnp.array([-1.16672594, 0.0, 1.16672594]),
    'V_down': jnp.array([-0.33941118, 0.0, -1.2261229]),
    'T_up': 0.3382,
    'T_down': 4.35646874
}



def setup_shock_hybrid(X, VX, VY, VZ, lx, x_1d, params=shock_dict_normalized):
    """
    Initial condition for a Hybrid Shock Tube.
    Flow direction: Right -> Left (Vx < 0).
    Upstream (Entrance): Right side (x=L).
    Downstream (Exit): Left side (x=0).
    """
    n_up, n_down = params['n_up'], params['n_down']
    B_up, B_down = params['B_up'], params['B_down']
    V_up, V_down = params['V_up'], params['V_down']
    T_up, T_down = params['T_up'], params['T_down']

    # 1. Coordinate & smoothing setup
    # S=0 at x=0 (Left), S=1 at x=L (Right)
    width = 2.0 
    shock_pos = 2 * lx / 4
    S = 0.5 * (1.0 + jnp.tanh((x_1d - shock_pos) / width))
    S_3d = 0.5 * (1.0 + jnp.tanh((X - shock_pos) / width)) 

    # 2. Field & Density Profiles (S=1 -> Upstream)
    n_1d = n_down + (n_up - n_down) * S
    B_x_1d = jnp.ones_like(x_1d) * B_up[0]
    B_y_1d = B_down[1] + (B_up[1] - B_down[1]) * S
    B_z_1d = B_down[2] + (B_up[2] - B_down[2]) * S
    
    # Equilibrium Pressure
    P_up = n_up * T_up + jnp.sum(B_up**2) / 2.0
    P_down = n_down * T_down + jnp.sum(B_down**2) / 2.0
    P_tot = jnp.maximum(P_up, P_down)
    
    # 3. 3D Moment Fields
    n_3d = n_down + (n_up - n_down) * S_3d
    V_x_3d = V_down[0] + (V_up[0] - V_down[0]) * S_3d
    V_y_3d = V_down[1] + (V_up[1] - V_down[1]) * S_3d
    V_z_3d = V_down[2] + (V_up[2] - V_down[2]) * S_3d
    
    # Local B field for pressure balance
    By_3d = B_down[1] + (B_up[1] - B_down[1]) * S_3d
    Bz_3d = B_down[2] + (B_up[2] - B_down[2]) * S_3d
    B_sq_3d = B_up[0]**2 + By_3d**2 + Bz_3d**2
    
    # Thermodynamics
    T_3d = (P_tot - B_sq_3d / 2.0) / n_3d
    T_3d = jnp.maximum(T_3d, 0.01)
    vth = jnp.sqrt(T_3d)

    # 4. Phase Space Distribution
    v_norm_sq = ((VX - V_x_3d)**2 + (VY - V_y_3d)**2 + (VZ - V_z_3d)**2)
    f = n_3d * jnp.exp(-v_norm_sq / (2 * vth**2)) / ((2 * jnp.pi * vth**2)**1.5)
    
    return f, B_x_1d, B_y_1d, B_z_1d