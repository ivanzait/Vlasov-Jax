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
    Loads f, E, B, and dx from multiple .npz files.
    """
    f_list, e_list, b_list = [], [], []
    dx = 0.5 # Default fallback
    
    for step in steps:
        path = os.path.join(data_dir, f"step_{step:04d}.npz")
        if not os.path.exists(path):
            continue
        with np.load(path) as data:
            f_list.append(jnp.array(data['f']))
            e_list.append(jnp.stack([data['E_x'], data['E_y'], data['E_z']], axis=-1))
            b_list.append(jnp.stack([data['B_x'], data['B_y'], data['B_z']], axis=-1))
            dx = float(data['dx'])
            
    return jnp.stack(f_list), jnp.stack(e_list), jnp.stack(b_list), dx

class EnrichedDataset:
    def __init__(self, fine_dir='data/fine', coarse_dir='data/coarse', steps=None, epsilon=1e-12):
        if steps is None:
            files = [f for f in os.listdir(coarse_dir) if f.startswith('step_') and f.endswith('.npz')]
            steps = sorted([int(f[5:9]) for f in files])
        
        print(f"Loading {len(steps)} snapshots for enriched training...")
        
        # 1. Load Data
        f_coarse, e_coarse, b_coarse, dx = load_simulation_data(coarse_dir, steps)
        f_fine, _, _, _ = load_simulation_data(fine_dir, steps)
        
        # 2. Downsample Fine
        f_fine_down = downsample_velocity(f_fine)
        
        # 3. Calculate Gradients (Gradients are calculated per component)
        # e_coarse shape: (n_steps, NX, 3)
        de_dx = jnp.stack([get_gradients(e_coarse[..., i], dx) for i in range(3)], axis=-1)
        db_dx = jnp.stack([get_gradients(b_coarse[..., i], dx) for i in range(3)], axis=-1)
        
        # 4. Target Residuals
        self.epsilon = epsilon
        self.labels = jnp.log(f_fine_down + epsilon) - jnp.log(f_coarse + epsilon)
        
        # 5. Build Input Vectors: [f_flattened (32768), E (3), B (3), dE_dx (3), dB_dx (3)]
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
        """
        Randomly splits into train, val, and test sets.
        """
        indices = jax.random.permutation(key, self.n_samples)
        
        train_end = int(self.n_samples * ratios[0])
        val_end = train_end + int(self.n_samples * ratios[1])
        
        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]
        
        return (self.inputs[train_idx], self.labels[train_idx]), \
               (self.inputs[val_idx], self.labels[val_idx]), \
               (self.inputs[test_idx], self.labels[test_idx])
