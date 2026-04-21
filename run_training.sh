#!/bin/bash
# VLSV-JAX ML Offline Training
echo "--- Starting Enriched MLP Training ---"
./.venv/bin/python3 -m src.ml.train_offline --config baseline
#./.venv/bin/python3 -m src.ml.train_offline --config no_grad
