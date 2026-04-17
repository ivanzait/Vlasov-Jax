#!/bin/bash
# VLSV-JAX Simulation Runner
# Usage: ./run_simulation.sh [config_name]

CONFIG=config_coarse

echo "--- Starting VLSV-JAX Hybrid Simulation: $CONFIG ---"
python3 -m src.solver.simulator --config $CONFIG
