# config_infer.py — Specialized for Final Validation Run
import os
# ==========================================
# 1. RESOLUTION & GRID (Coarse Target)
# ==========================================
NX = 64
NV = 32
NT = 30
DX = 0.45
DV = 1.8
DT = 0.05
LV = 28.8

BC_X = ('static', 'copy')
BC_V = 'copy'

# ==========================================
# 2. MACHINE LEARNING (Physics-ML)
# ==========================================
USE_ML = True
ML_MODE = 'inference'
MODEL_PATH = "data/ml_data/model_weights_final.npz"

# Hyperparameters (Advection CNN)
ML_ADVECTION_CLAMP    = 2e-2
ML_ADVECTION_KERNEL   = 3
ML_BOUNDARY_BUFFER    = 2
ML_LR_CNN             = 1e-3
ML_USE_CNN            = True
ML_ADVECT_FLUX_CORRECTION = False

# Acceleration Corrector: DeepONet (Branch CNN x Trunk MLP)
ML_USE_MLP            = True
ML_ACC_MODE           = 'deeponet'   # 'mlp' | 'deeponet'
ML_DEEPONET_D         = 32           # latent dimension D
ML_DEEPONET_TRUNK_HIDDEN  = 64       # trunk MLP hidden dim
ML_DEEPONET_BRANCH_HIDDEN = 32       # branch CNN channels
ML_DEEPONET_KERNEL    = 5            # branch CNN kernel size
ML_LR_MLP             = 1e-3         # Adam LR for acc subtree
ML_GRAD_CLIP_NORM     = 1.0          # global gradient clipping (replaces tanh clamp)
ML_UPDATES_PER_SAVE   = 3

# Legacy (unused in deeponet mode, kept for MLP fallback)
ML_ACCEL_HIDDEN       = 16
ML_ACCELERATION_CLAMP = 1e-2

# ==========================================
# 3. I/O & DIAGNOSTICS
# ==========================================
SAVE_EVERY = 5
SAVE_DIR = "data/corrected_data"
ML_DATA_DIR = "data/ml_data"
ML_WEIGHTS_PATH = os.path.join(ML_DATA_DIR, "model_weights_final.npz")
PLOT_DIR = "plots/verification"

# ==========================================
# 4. PHYSICAL CONSTANTS
# ==========================================
B_BACKGROUND = 1.0
NI_UPSTREAM  = 1.0
MI = 1.0
QI = 1.0
MU_0 = 1.0
GAMMA = 5.0/3.0
kB    = 1.0
