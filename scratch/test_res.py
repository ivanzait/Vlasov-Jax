import jax.numpy as jnp

def test_gaussian_res(vth, dv, nv):
    v = jnp.linspace(-nv*dv/2, nv*dv/2, nv)
    # Gaussian 1D for simplicity
    f = jnp.exp(-v**2 / (2 * vth**2)) / (jnp.sqrt(2 * jnp.pi) * vth)
    n_sum = jnp.sum(f) * dv
    error = jnp.abs(n_sum - 1.0)
    return error

vth_up = jnp.sqrt(0.3461) # ~0.588
print(f"vth_up: {vth_up}")

for dv in [0.9, 0.45, 0.225]:
    err = test_gaussian_res(vth_up, dv, 128)
    print(f"dv={dv}, Error={err:.10f}")
