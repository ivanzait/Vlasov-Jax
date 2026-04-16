import jax
import jax.numpy as jnp
import optax
import time
import os
import pickle

from ml_dataset import EnrichedDataset
from ml_models import MLP, update_physics, physics_loss_fn

def main():
    # 1. Configuration (Phase 4 Final Refinement)
    # Architecture: 32780 -> 256 -> 256 -> 128 -> 32768
    HIDDEN_DIMS = [256, 256, 128] 
    INPUT_DIM = 32780
    OUTPUT_DIM = 32768
    
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4
    EPOCHS = 500
    LAMBDA_PHYS = 5.0  # Increased from 1.0 to prioritize density/velocity matching
    V_SCALE = 4.0
    
    v = jnp.linspace(-16, 16, 32)
    dv = v[1] - v[0]
    
    # 2. Setup Dataset
    dataset = EnrichedDataset(fine_dir='data/fine', coarse_dir='data/coarse')
    key = jax.random.PRNGKey(42)
    train_set, val_set, test_set = dataset.get_split(key, ratios=(0.6, 0.2, 0.2))
    
    train_inputs, train_labels = train_set
    val_inputs, val_labels = val_set
    
    # 3. Initialize Model & Optimizer
    key, model_key = jax.random.split(key)
    model = MLP(model_key, input_dim=INPUT_DIM, hidden_dims=HIDDEN_DIMS, output_dim=OUTPUT_DIM)
    params = model.params
    
    optimizer = optax.adam(LEARNING_RATE)
    opt_state = optimizer.init(params)
    
    # 4. Training Loop
    print(f"--- Starting Final Refinement Training (3-layer Hidden) ---")
    print(f"Architecture: {INPUT_DIM} -> {HIDDEN_DIMS} -> {OUTPUT_DIM}")
    print(f"Lambda Physics: {LAMBDA_PHYS} (Stronger Moment Constrainment)")
    
    train_key = jax.random.PRNGKey(0)
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        
        train_key, batch_key = jax.random.split(train_key)
        indices = jax.random.choice(batch_key, train_inputs.shape[0], shape=(BATCH_SIZE,))
        bx, by = train_inputs[indices], train_labels[indices]
        
        # Physics-aware weighted update
        params, opt_state, train_loss = update_physics(
            params, bx, by, opt_state, optimizer, v, dv, 
            lambda_phys=LAMBDA_PHYS, v_scale=V_SCALE
        )
        
        if epoch % 50 == 0:
            val_loss = physics_loss_fn(params, val_inputs[:16], val_labels[:16], v, dv, LAMBDA_PHYS, V_SCALE)
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch:03d}: Total Loss = {train_loss:.6f} | Val Loss = {val_loss:.6f} | Time = {epoch_time:.2f}s")
            
    # 5. Save Final Artifacts
    os.makedirs('ml_weights', exist_ok=True)
    weight_path = 'ml_weights/mlp_final_phys.pkl'
    with open(weight_path, 'wb') as f:
        pickle.dump(params, f)
    
    test_path = 'ml_weights/test_data_split.pkl'
    with open(test_path, 'wb') as f:
        pickle.dump({'inputs': test_set[0], 'labels': test_set[1], 'v': v, 'dv': dv}, f)
        
    print(f"\nTraining complete. Weights saved to {weight_path}")

if __name__ == "__main__":
    main()
