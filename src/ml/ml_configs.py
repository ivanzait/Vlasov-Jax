"""
Centralized Configuration for ML Experiments.
Defines feature sets, dimensions, and training presets.
"""

# Mapping of features to their spatial/vector dimensions
FEATURES_DIM = {
    'f': 32768,      # Canoncial 32^3 Velocity Grid
    'E': 3,          # Electric Field (Ex, Ey, Ez)
    'B': 3,          # Magnetic Field (Bx, By, Bz)
    'grad_E': 3,     # E-Field spatial gradient (dEx/dx, dEy/dx, dEz/dx)
    'grad_B': 3      # B-Field spatial gradient (dBx/dx, dBy/dx, dBz/dx)
}

# --- Experiment Presets ---

BASELINE_CONFIG = {
    'f': True,
    'E': True,
    'B': True,
    'grad_E': True,
    'grad_B': True
}

NO_GRAD_CONFIG = {
    'f': True,
    'E': True,
    'B': True,
    'grad_E': False,
    'grad_B': False
}

def get_input_dim(config):
    """Calculates the total input dimension for a given feature config."""
    return sum(FEATURES_DIM[k] for k, v in config.items() if v)

def get_config(name):
    if name == 'baseline':
        return BASELINE_CONFIG
    elif name == 'no_grad':
        return NO_GRAD_CONFIG
    else:
        raise ValueError(f"Unknown config: {name}")
