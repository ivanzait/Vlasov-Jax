#!/bin/bash

# VLSV-JAX Pipeline Automation Script
# Orchestrates: Online Training -> Fixed-Weight Inference -> Dashboard Generation

# Exit on any error
set -e

echo "===================================================="
echo "🚀 Starting VLSV-JAX Physics-ML Pipeline"
echo "===================================================="

# Ensure directories exist
mkdir -p data/temp_training data/corrected_data data/ml_data
mkdir -p plots/temp_training plots/corrected plots/verification

# --------------------------------------------------
# STEP 1: Online Training
# --------------------------------------------------
echo ""
echo "🔥 STEP 1: Online Training (Weights Evolving)"
echo "----------------------------------------------------"
cp config_train.py config.py
python run_maxwell.py

# --------------------------------------------------
# STEP 2: Fixed-Weight Inference
# --------------------------------------------------
echo ""
echo "💎 STEP 2: Fixed-Weight Inference (Clean Validation)"
echo "----------------------------------------------------"
cp config_infer.py config.py
python run_maxwell.py

# --------------------------------------------------
# STEP 3: Verification Dashboard
# --------------------------------------------------
echo ""
echo "📊 STEP 3: Generating Verification Dashboards"
echo "----------------------------------------------------"
python verify_training.py

echo ""
echo "===================================================="
echo "✅ Pipeline Complete!"
echo "Dashboards generated in: plots/verification/"
echo "Final weights saved to: data/ml_data/model_weights_final.npz"
echo "===================================================="
