import jax.numpy as jnp


# Derived from plasma_calculator.ipynb constraints
shock_dict_normalized = {
    'n_up': 1.0,
    'n_down': 3.4375,
    'B_up': jnp.array([0.70710678, 0.0, 0.70710678]),
    'B_down': jnp.array([0.70710678, 0.0, 2.55677028]),
    'V_up': jnp.array([-5.0365157, 0.0, 5.0365157]),
    'V_down': jnp.array([-1.4651682, 0.0, -5.2929201]),
    'T_up': 0.3461,
    'T_down': 4.4555 # Derived from Momentum Conservation
}




def setup_shock_hybrid(X, VX, VY, VZ, lx, x_1d, params=shock_dict_normalized):
    """
    Initial condition for a Hybrid Shock Tube maintaining mechanical equilibrium.
    Satisfies n(x)T(x) + B(x)^2/2 = Constant.
    """
    n_up, n_down = params['n_up'], params['n_down']
    B_up, B_down = params['B_up'], params['B_down']
    V_up, V_down = params['V_up'], params['V_down']
    T_up, T_down = params['T_up'], params['T_down']

    # 1. Coordinate & Smoothing setup
    width = 1.0 
    step1 = 0.5 * (1.0 + jnp.tanh((x_1d -  3 * lx / 4) / width))
    S = step1
    S_3d = 0.5 * (1.0 + jnp.tanh((X - 3 * lx / 4) / width)) 
    #step2 = 0.5 * (1.0 - jnp.tanh((x_1d - 3 * lx / 4) / width))
    #S = step1 * step2 
    #S_3d = 0.5 * (1.0 + jnp.tanh((X - lx / 4) / width)) * 0.5 * (1.0 - jnp.tanh((X - 3 * lx / 4) / width))

    # 2. Field & Density Profiles
    n_1d = n_up + (n_down - n_up) * S
    B_x_1d = jnp.ones_like(x_1d) * B_up[0]
    B_y_1d = B_up[1] + (B_down[1] - B_up[1]) * S
    B_z_1d = B_up[2] + (B_down[2] - B_up[2]) * S
    
    # Define Total Pressure based on the maximum required to keep both sides positive
    P_up = n_up * T_up + jnp.sum(B_up**2) / 2.0
    P_down = n_down * T_down + jnp.sum(B_down**2) / 2.0
    P_tot = jnp.maximum(P_up, P_down)
    
    # 3. 3D Moment Fields
    n_3d = n_up + (n_down - n_up) * S_3d
    V_x_3d = V_up[0] + (V_down[0] - V_up[0]) * S_3d
    V_y_3d = V_up[1] + (V_down[1] - V_up[1]) * S_3d
    V_z_3d = V_up[2] + (V_down[2] - V_up[2]) * S_3d
    
    # Compute local B^2 on 3D grid
    By_3d = B_up[1] + (B_down[1] - B_up[1]) * S_3d
    Bz_3d = B_up[2] + (B_down[2] - B_up[2]) * S_3d
    B_sq_3d = B_up[0]**2 + By_3d**2 + Bz_3d**2
    
    # Calculate local Temperature T(x) = (P_tot - P_mag) / n
    T_3d = (P_tot - B_sq_3d / 2.0) / n_3d
    T_3d = jnp.maximum(T_3d, 0.01) # Robust floor
    
    vth = jnp.sqrt(T_3d)

    # 4. Phase Space Distribution f(x, v)
    v_norm_sq = ((VX - V_x_3d)**2 + (VY - V_y_3d)**2 + (VZ - V_z_3d)**2)
    f = n_3d * jnp.exp(-v_norm_sq / (2 * vth**2)) / ((2 * jnp.pi * vth**2)**1.5)
    
    return f, B_x_1d, B_y_1d, B_z_1d