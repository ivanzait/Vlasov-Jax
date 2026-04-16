# Simulation Configuration for VLSV-JAX
# All parameters are normalized to ion scales (d_i, Omega_ci, v_A)

# --- Grid Parameters ---
# Spatial Grid (X)
NX = 64
DX = 0.5 

# Velocity Grid (Vx, Vy, Vz)
NV = 32
DV = 0.6

# --- Simulation Timing ---
NT = 70
DT = 0.05

# --- Physical Constants ---
QI = 1.0    # Normalized ion charge
MI = 1.0    # Normalized ion mass
MU0 = 1.0   # Normalized vacuum permeability

# --- Boundary Conditions ---
# Options: 'periodic', 'copy', 'static'
BC_X = ('static', 'copy')
BC_V = 'copy'

# --- Visualization ---
PLOT_DIR = "plots/plots_test"
PLOT_EVERY = 10

# --- Data Persistence ---
DATA_DIR = "data/test"
SAVE_STRIDE = 10
