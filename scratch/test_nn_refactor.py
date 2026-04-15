import jax
import jax.numpy as jnp
from nn_models import init_network_params, apply_advection_correction

def test_shapes():
    nx, nv = 32, 16 
    key = jax.random.PRNGKey(0)
    
    print(f"Initializing params for nx={nx}, nv={nv}...")
    params = init_network_params(key, nx, nv)
    
    # Check param shapes
    w_shape = params['adv']['w'].shape
    print(f"Advection weight shape: {w_shape}")
    expected_w_shape = (3, 1, nv**3)
    assert w_shape == expected_w_shape, f"Wrong shape: {w_shape} != {expected_w_shape}"
    
    # Check forward pass
    f = jax.random.normal(key, (nx, nv, nv, nv))
    v = jnp.linspace(-5, 5, nv)
    dt = 0.1
    
    print("Running forward pass...")
    delta_f = apply_advection_correction(f, v, dt, params['adv'])
    
    print(f"Initial f shape: {f.shape}")
    print(f"Delta f shape: {delta_f.shape}")
    assert f.shape == delta_f.shape, "Shape mismatch in forward pass"
    print("Success! Memory-efficient architecture is functional.")

if __name__ == "__main__":
    test_shapes()
