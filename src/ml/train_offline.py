import jax
import jax.numpy as jnp
import optax
import time
import os
import pickle

from .ml_dataset import EnrichedDataset, load_simulation_data
from .ml_models import MLP, update_physics, physics_loss_fn

def main():
    # 1. Configuration (Phase 4 Final Refinement)
    # Architecture: 32780 -> 256 -> 256 -> 128 -> 32768
    HIDDEN_DIMS = [256, 256, 128] 
    INPUT_DIM = 32780
    OUTPUT_DIM = 32768
    
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4  # Higher LR for simpler 3-layer network
    EPOCHS = 2000
    LAMBDA_PHYS = 5.0 
    V_SCALE = 4.0
    
    # 2. Setup Dataset (Native 32^3 Benchmark)
    dataset = EnrichedDataset(
        fine_dir='data/fine', 
        coarse_dirs=['data/coarse']
    )
    
    # Grid info for moment calculation (Canonical 32^3 Reference)
    # We must use the 32^3 grid because the MLP always operates at this resolution
    data_ref = load_simulation_data('data/coarse', [0])
    v_phys = data_ref['metadata']['v']
    dv_phys = data_ref['metadata']['dv']
    
    key = jax.random.PRNGKey(42)
    train_set, val_set, test_set = dataset.get_split(key, ratios=(0.6, 0.2, 0.2))
    
    train_inputs, train_labels = train_set
    val_inputs, val_labels = val_set
    
    # 3. Initialize Model & Optimizer
    key, model_key = jax.random.split(key)
    model = MLP(model_key, input_dim=INPUT_DIM, hidden_dims=HIDDEN_DIMS, output_dim=OUTPUT_DIM)
    params = model.params
    
    # Use scheduler to prevent loss bounce at the end of training
    lr_schedule = optax.exponential_decay(
        init_value=LEARNING_RATE, 
        transition_steps=600, 
        decay_rate=0.85
    )
    optimizer = optax.adam(lr_schedule)
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
            params, bx, by, opt_state, optimizer, v_phys, dv_phys, 
            lambda_phys=LAMBDA_PHYS, v_scale=V_SCALE
        )
        
        if epoch % 50 == 0:
            val_loss = physics_loss_fn(params, val_inputs[:16], val_labels[:16], v_phys, dv_phys, LAMBDA_PHYS, V_SCALE)
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch:03d}: Total Loss = {train_loss:.6f} | Val Loss = {val_loss:.6f} | Time = {epoch_time:.2f}s")
            
    # 5. Save Final Artifacts
    os.makedirs('data/ml_weights', exist_ok=True)
    weight_path = 'data/ml_weights/mlp_final_phys.pkl'
    with open(weight_path, 'wb') as f:
        pickle.dump(params, f)
    
    test_path = 'data/ml_weights/test_data_split.pkl'
    with open(test_path, 'wb') as f:
        pickle.dump({'inputs': test_set[0], 'labels': test_set[1], 'v': v_phys, 'dv': dv_phys}, f)
        
    print(f"\nTraining complete. Weights saved to {weight_path}")

if __name__ == "__main__":
    main()
