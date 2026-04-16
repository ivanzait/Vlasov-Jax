# Simulation Configuration for VLSV-JAX
# All parameters are normalized to ion scales (d_i, Omega_ci, v_A)

# --- Grid Parameters ---
# Spatial Grid (X)
NX = 64
DX = 0.5 

# Velocity Grid (Vx, Vy, Vz)
NV = 64
DV = 0.5

# --- Simulation Timing ---
NT = 100
DT = 0.02

# --- Physical Constants ---
QI = 1.0    # Normalized ion charge
MI = 1.0    # Normalized ion mass
MU0 = 1.0   # Normalized vacuum permeability

# --- Boundary Conditions ---
# Options: 'periodic', 'copy', 'static'
BC_X = ('copy', 'static')
BC_V = 'copy'

# --- Visualization ---
PLOT_DIR = "plots/plots_fine"
PLOT_EVERY = 10

# --- Data Persistence ---
DATA_DIR = "data/fine"
SAVE_STRIDE = 2
