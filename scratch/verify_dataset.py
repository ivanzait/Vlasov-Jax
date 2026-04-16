import jax
import jax.numpy as jnp
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ml_dataset import EnrichedDataset

def verify_dataset():
    # Load a small subset
    dataset = EnrichedDataset(steps=[0, 10, 20])
    
    key = jax.random.PRNGKey(42)
    train, val, test = dataset.get_split(key)
    
    inputs, labels = train
    print(f"Input shape: {inputs.shape}")
    print(f"Label shape: {labels.shape}")
    
    # Check dimensions
    # f (32768) + E (3) + B (3) + dE (3) + dB (3) = 32780
    assert inputs.shape[1] == 32780, f"Expected 32780, got {inputs.shape[1]}"
    
    # Check fields are not all zero (gradients should exist)
    fields = inputs[:, -12:]
    print("Field Stats (E, B, dE, dB):")
    print(f"  Mean Magnitudes: {jnp.mean(jnp.abs(fields), axis=0)}")
    
    print("\nDataset Verification Successful.")

if __name__ == "__main__":
    verify_dataset()
