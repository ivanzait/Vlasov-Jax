import jax
import jax.numpy as jnp
import numpy as np
import os

def downsample_velocity(f_fine):
    """
    Downsamples f from (..., 64, 64, 64) to (..., 32, 32, 32) 
    using 2x2x2 box averaging.
    """
    shape = f_fine.shape[:-3]
    f_reshaped = f_fine.reshape(*shape, 32, 2, 32, 2, 32, 2)
    return jnp.mean(f_reshaped, axis=(-5, -3, -1))

def get_gradients(arr, dx):
    """
    Central difference gradient for 1D fields.
    """
    return (jnp.roll(arr, -1, axis=-1) - jnp.roll(arr, 1, axis=-1)) / (2 * dx)

def load_simulation_data(data_dir, steps):
    """
    Loads f, E, B, and grid info from multiple .npz files.
    Returns a dictionary of stacked arrays and metadata.
    Raises KeyError if any self-describing metadata is missing.
    """
    f_list, e_list, b_list = [], [], []
    
    # 1. Probe first file for metadata
    first_path = os.path.join(data_dir, f"step_{steps[0]:04d}.npz")
    with np.load(first_path) as data:
        dx = float(data['dx'])
        dv = float(data['dv'])
        dt = float(data['dt'])
        v = jnp.array(data['v'])
        x_grid = jnp.array(data['x'])

    # 2. Sequential loading
    for step in steps:
        path = os.path.join(data_dir, f"step_{step:04d}.npz")
        if not os.path.exists(path):
            continue
        with np.load(path) as data:
            f_list.append(jnp.array(data['f']))
            e_list.append(jnp.stack([data['E_x'], data['E_y'], data['E_z']], axis=-1))
            b_list.append(jnp.stack([data['B_x'], data['B_y'], data['B_z']], axis=-1))
            
    return {
        'f': jnp.stack(f_list),
        'E': jnp.stack(e_list),
        'B': jnp.stack(b_list),
        'metadata': {
            'dx': dx, 'dv': dv, 'dt': dt, 'v': v, 'x': x_grid
        }
    }

class EnrichedDataset:
    def __init__(self, fine_dir='data/fine', coarse_dir='data/coarse', steps=None, epsilon=1e-12):
        if steps is None:
            files = [f for f in os.listdir(coarse_dir) if f.startswith('step_') and f.endswith('.npz')]
            steps = sorted([int(f[5:9]) for f in files])
        
        print(f"Loading {len(steps)} snapshots for enriched training...")
        
        # 1. Load Data (Strict Dictionary Contract)
        data_c = load_simulation_data(coarse_dir, steps)
        data_f = load_simulation_data(fine_dir, steps)
        
        f_coarse = data_c['f']
        e_coarse = data_c['E']
        b_coarse = data_c['B']
        dx = data_c['metadata']['dx']
        
        self.v = data_c['metadata']['v']
        self.dv = data_c['metadata']['dv']
        
        # 2. Downsample Fine
        f_fine_down = downsample_velocity(data_f['f'])
        
        # 3. Calculate Gradients
        de_dx = jnp.stack([get_gradients(e_coarse[..., i], dx) for i in range(3)], axis=-1)
        db_dx = jnp.stack([get_gradients(b_coarse[..., i], dx) for i in range(3)], axis=-1)
        
        # 4. Target Residuals
        self.epsilon = epsilon
        self.labels = jnp.log(f_fine_down + epsilon) - jnp.log(f_coarse + epsilon)
        
        # 5. Build Input Vectors
        n_steps, nx = f_coarse.shape[:2]
        self.n_samples = n_steps * nx
        
        f_flat = f_coarse.reshape(self.n_samples, -1)
        e_flat = e_coarse.reshape(self.n_samples, 3)
        b_flat = b_coarse.reshape(self.n_samples, 3)
        de_flat = de_dx.reshape(self.n_samples, 3)
        db_flat = db_dx.reshape(self.n_samples, 3)
        
        self.inputs = jnp.concatenate([f_flat, e_flat, b_flat, de_flat, db_flat], axis=1)
        self.labels = self.labels.reshape(self.n_samples, -1)
        
        print(f"Dataset ready. Input dimension: {self.inputs.shape[1]}")

    def get_split(self, key, ratios=(0.6, 0.2, 0.2)):
        indices = jax.random.permutation(key, self.n_samples)
        train_end = int(self.n_samples * ratios[0])
        val_end = train_end + int(self.n_samples * ratios[1])
        
        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]
        
        return (self.inputs[train_idx], self.labels[train_idx]), \
               (self.inputs[val_idx], self.labels[val_idx]), \
               (self.inputs[test_idx], self.labels[test_idx])
