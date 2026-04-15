import jax
import jax.numpy as jnp
import os
import shutil
from data_io import save_snapshot, load_snapshot, save_model_weights, load_model_weights
from nn_models import init_network_params

def test_snapshot_io():
    print("Testing Snapshot I/O...")
    temp_dir = "test_data"
    f = jnp.ones((10, 5, 5, 5))
    B = (jnp.zeros(10), jnp.ones(10), jnp.zeros(10))
    E = (jnp.ones(10), jnp.zeros(10), jnp.zeros(10))
    params = init_network_params(jax.random.PRNGKey(0), 10, 5)
    
    save_snapshot(1, f, B, E, params, save_dir=temp_dir)
    
    loaded = load_snapshot(os.path.join(temp_dir, "snapshot_00001.npz"))
    
    assert jnp.allclose(f, loaded['f'])
    assert jnp.allclose(B[1], loaded['B'][1])
    print("Snapshot I/O: SUCCESS")
    
    # Cleanup
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

def test_weight_io():
    print("Testing Weight I/O...")
    params = init_network_params(jax.random.PRNGKey(0), 10, 5)
    save_model_weights(params, "test_weights.npz")
    
    # Load back
    loaded_params = load_model_weights("test_weights.npz", params)
    
    # Check values
    flat_orig, _ = jax.tree_util.tree_flatten(params)
    flat_load, _ = jax.tree_util.tree_flatten(loaded_params)
    
    for o, l in zip(flat_orig, flat_load):
        assert jnp.allclose(o, l)
        
    print("Weight I/O: SUCCESS")
    os.remove("test_weights.npz")

if __name__ == "__main__":
    test_snapshot_io()
    test_weight_io()
