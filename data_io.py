import os
import jax
import jax.numpy as jnp
import numpy as np

def save_snapshot(step, f, B, E, grid_info=None, params=None, save_dir="data_snapshots"):
    """
    Saves a full simulation snapshot to a compressed .npz file.
    grid_info: dict or tuple containing (nx, nv, dx, dv, lx, lv)
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    path = os.path.join(save_dir, f"snapshot_{step:05d}.npz")
    
    # Pack array data
    data = {
        'f': np.array(f),
        'Bx': np.array(B[0]),
        'By': np.array(B[1]),
        'Bz': np.array(B[2]),
        'Ex': np.array(E[0]),
        'Ey': np.array(E[1]),
        'Ez': np.array(E[2]),
        'step': np.array(step)
    }
    
    # Pack grid metadata if provided
    if grid_info is not None:
        if isinstance(grid_info, dict):
            for k, v in grid_info.items():
                data[f'grid/{k}'] = np.array(v)
        elif isinstance(grid_info, (list, tuple)) and len(grid_info) >= 6:
            # Fallback for tuple ordering: (nx, nv, dx, dv, lx, lv)
            keys = ['nx', 'nv', 'dx', 'dv', 'lx', 'lv']
            for k, v in zip(keys, grid_info):
                data[f'grid/{k}'] = np.array(v)
    
    # Flatten ML params if provided
    if params is not None:
        flat_params, _ = jax.tree_util.tree_flatten_with_path(params)
        for p_path, val in flat_params:
            key = "params/" + "/".join([str(p.name) if hasattr(p, 'name') else str(p) for p in p_path])
            data[key] = np.array(val)
            
    np.savez_compressed(path, **data)
    print(f"Snapshot saved: {path}")

def load_snapshot(path):
    """
    Loads a simulation snapshot and returns a dictionary of JAX arrays and metadata.
    """
    with np.load(path, allow_pickle=True) as data:
        res = {
            'f': jnp.array(data['f']),
            'B': (jnp.array(data['Bx']), jnp.array(data['By']), jnp.array(data['Bz'])),
            'E': (jnp.array(data['Ex']), jnp.array(data['Ey']), jnp.array(data['Ez'])),
            'step': int(data['step'])
        }
        
        # Reconstruct grid metadata
        grid_keys = [k for k in data.files if k.startswith('grid/')]
        metadata = {}
        for k in grid_keys:
            metadata[k.replace('grid/', '')] = float(data[k])
        res['metadata'] = metadata
        
        # Reconstruct params if they exist
        param_keys = [k for k in data.files if k.startswith('params/')]
        if param_keys:
            params_flat = {}
            for k in param_keys:
                params_flat[k.replace('params/', '')] = jnp.array(data[k])
            res['params_flat'] = params_flat
            
    return res

def save_model_weights(params, path):
    """ Saves only the ML parameters. """
    flat_params, _ = jax.tree_util.tree_flatten_with_path(params)
    data = {}
    for p_path, val in flat_params:
        key = "/".join([str(p.name) if hasattr(p, 'name') else str(p) for p in p_path])
        data[key] = np.array(val)
    np.savez_compressed(path, **data)
    print(f"Weights saved: {path}")

def load_model_weights(path, template_params):
    """ 
    Loads weights from path and injects them into template_params (Pytree).
    Ensures the structure matches.
    """
    with np.load(path) as data:
        # We need the structure to reconstruct the tree
        import re
        flat_template, tree_def = jax.tree_util.tree_flatten_with_path(template_params)

        def _clean_component(comp):
            # Handle DictKey objects (from flatten_with_path on dicts)
            if hasattr(comp, 'key'):
                s = str(comp.key)
            else:
                s = str(comp)
            # Remove surrounding brackets/quotes like ['acc'] -> acc
            s = re.sub(r'^[\[\]\'\"]+|[\[\]\'\"]+$', '', s)
            # Keep only reasonable chars
            s = re.sub(r'[^0-9A-Za-z_]+', '', s)
            return s

        new_values = []
        for p_path, _ in flat_template:
            # Build raw key taking DictKey.key into account
            parts = []
            for p in p_path:
                if hasattr(p, 'key'):
                    parts.append(str(p.key))
                elif hasattr(p, 'name'):
                    parts.append(str(p.name))
                else:
                    parts.append(str(p))
            raw_key = "/".join(parts)
            # Try raw key first (legacy behavior)
            if raw_key in data:
                new_values.append(jnp.array(data[raw_key]))
                continue

            # Try cleaned key (normalizing bracketed representations)
            clean_key = "/".join([_clean_component(p) for p in p_path])
            if clean_key in data:
                new_values.append(jnp.array(data[clean_key]))
                continue

            # Try alternative existing-format keys where components are like "['acc']" joined with '/'
            alt_key = "/".join([f"[{repr(str(p))}]" for p in p_path])
            if alt_key in data:
                new_values.append(jnp.array(data[alt_key]))
                continue

            # As a last resort, try matching by normalized stored keys
            matched = None
            for k in data.files:
                # produce a normalized form of stored key, e.g. "['acc']/['b1']" -> "acc/b1"
                norm = re.sub(r"[^0-9A-Za-z_/]+", "", k)
                if norm == clean_key or norm.endswith('/' + clean_key) or norm.endswith(clean_key):
                    matched = k
                    break
            if matched is not None:
                new_values.append(jnp.array(data[matched]))
                continue

            raise KeyError(f"Weight key '{raw_key}' not found in checkpoint.")

        return jax.tree_util.tree_unflatten(tree_def, new_values)
