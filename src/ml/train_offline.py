import jax
import jax.numpy as jnp
import optax
import time
import os
import pickle
import argparse

from .ml_dataset import EnrichedDataset, load_simulation_data
from .ml_models import MLP, update_physics, physics_loss_fn
from .ml_configs import get_config, get_input_dim

def main():
    parser = argparse.ArgumentParser(description="Train VLSV-JAX Neural Corrector.")
    parser.add_argument("--config", type=str, default="baseline", help="Experiment config name (baseline, no_grad)")
    parser.add_argument("--epochs", type=int, default=2000, help="Number of training epochs")
    args = parser.parse_args()
    
    # 1. Load Configuration
    exp_config = get_config(args.config)
    
    HIDDEN_DIMS = [256, 256, 128] 
    INPUT_DIM = get_input_dim(exp_config)
    OUTPUT_DIM = 32768
    
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4  
    EPOCHS = args.epochs
    LAMBDA_PHYS = 5.0 
    V_SCALE = 4.0
    
    # 2. Setup Dataset (Dictionary-based feature activation)
    dataset = EnrichedDataset(
        fine_dir='data/fine', 
        coarse_dirs=['data/coarse'],
        feature_config=exp_config
    )
    
    # Grid info for moment calculation (Canonical 32^3 Reference)
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
    weight_path = f'data/ml_weights/mlp_{args.config}.pkl'
    with open(weight_path, 'wb') as f:
        pickle.dump(params, f)
    
    test_path = f'data/ml_weights/test_data_{args.config}.pkl'
    with open(test_path, 'wb') as f:
        pickle.dump({'inputs': test_set[0], 'labels': test_set[1], 'v': v_phys, 'dv': dv_phys}, f)
        
    print(f"\nTraining complete. Weights saved to {weight_path}")

if __name__ == "__main__":
    main()
