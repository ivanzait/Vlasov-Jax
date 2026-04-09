from initialize_maxwell import initialize_simulation
import jax.numpy as jnp

print("--- TESTING DV DEPENDENCE ---")
for dv_val in [0.9, 0.45]:
    # We need to calculate nv for a fixed lv or similar, or just change nv and dv together
    # initialize_simulation doesn't take dv directly, it calculates it from nv and lv (incorrectly in initialize_maxwell.py)
    # Wait, initialize_maxwell.py has hardcoded dx and dv inside the function now?
    # Let me check the file content again.
    pass

# Better approach: modify initialize_maxwell.py temporarily or call it with different params if I can.
import os

# Let's read the current initialize_maxwell.py to see how it sets dx/dv
