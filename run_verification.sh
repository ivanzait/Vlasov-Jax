#!/bin/bash
# VLSV-JAX ML Quantification & Plotting
echo "--- Running ML Performance Scorecard and Verification Dashboard ---"
python3 -m src.ml.ml_quantification
python3 -m src.ml.ml_plots
