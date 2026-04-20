# Super-Coarse Simulation Configuration (NV=16)
# Used for the Phase 9 Generalization Study

# --- Grid Parameters ---
NX = 64
DX = 0.5 

# Velocity Grid (16^3)
# We preserve the velocity span [-16, 16] by doubling DV
NV = 16
DV = 2.0

# --- Simulation Timing ---
NT = 100
DT = 0.02

# --- Physical Constants ---
QI = 1.0
MI = 1.0
MU0 = 1.0

# --- Boundary Conditions ---
BC_X = ('copy', 'static')
BC_V = 'copy'

# --- Visualization ---
# Using the modernized publication styling
PLOT_DIR = "plots/plots_coarse_nv16"
PLOT_EVERY = 10

# --- Data Persistence ---
DATA_DIR = "data/coarse_nv16"
SAVE_STRIDE = 2
