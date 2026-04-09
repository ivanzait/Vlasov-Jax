import jax.numpy as jnp

# ==========================================
# ELECTROSTATIC (ELECTRON) NORMALIZATION
# ==========================================
# Normalization for Poisson solver (Vlasov-Poisson):
# distance ~ Debye length (lambda_D)
# velocity ~ Thermal velocity (v_th)
# time ~ Inverse plasma frequency (1 / omega_pe)

q_e = -1.0        # Normalized electron charge
m_e = 1.0         # Normalized electron mass
epsilon_0 = 1.0   # Normalized permittivity 

def get_pe(n_e):
    return jnp.sqrt(n_e * q_e**2 / (epsilon_0 * m_e))

def get_vt(T_e):
    return jnp.sqrt(T_e / m_e)

def get_debye(T_e, n_e):
    return jnp.sqrt(epsilon_0 * T_e / (n_e * q_e**2)) 
