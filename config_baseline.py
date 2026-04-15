# VLSV-JAX Fine-Resolution Configuration (Ground Truth)
# NV=64, DV=0.9 — well-resolved velocity space (dv/vth ~ 0.43)
# Used to generate training targets in data/fine_data/
import os

# ==========================================
# 1. RESOLUTION & GRID
# ==========================================
NX = 64
NV = 64
NT = 30

DX = 0.7
DV = 0.9   # lv = NV * DV / 2 = 28.8 V_A
DT = 0.05

# Boundary Conditions — static inflow (upstream), copy outflow (downstream)
BC_X = ('static', 'copy')
BC_V = 'copy'

# ==========================================
# 2. MACHINE LEARNING (disabled for data generation)
# ==========================================
USE_ML = False
ML_MODE = 'inference'
MODEL_PATH = "data/ml_data/model_weights_final.npz"
ML_LOG_CLAMP = 5e-2    # Max log-residual magnitude per sub-step (tanh clamp)

# ==========================================
# 3. I/O & DIAGNOSTICS
# ==========================================
SAVE_EVERY = 5
SAVE_DIR = "data/fine_data"    # <-- ground truth snapshots
ML_DATA_DIR = "data/ml_data"
PLOT_DIR = "plots_fine"

# Ensure crucial directories exist
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(ML_DATA_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# ==========================================
# 4. PHYSICAL CONSTANTS (Normalized)
# ==========================================
# Normalized to B_0=1.0, n_0=1.0, m_i=1.0, q_i=1.0
B_BACKGROUND = 1.0
MU_0 = 1.0
MI = 1.0
QI = 1.0
