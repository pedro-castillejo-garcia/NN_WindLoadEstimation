import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_squared_error
from torch.utils.data import TensorDataset, DataLoader
import sys
from features import load_data

# Get the absolute path of the project's root directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))  # Move up one level

# Add project root to sys.path
sys.path.append(project_root)

from models.Transformer import TransformerModel

def main():
    print("[INFO] Script started. Initializing variables...")

    # Load test data
    batch_params = {
        "gap": 10,
        "total_len": 100,
        "batch_size": 32  
    }
    print(f"[INFO] Batch parameters: {batch_params}")

    hyperparameters = {
        "dropout": 0.3,
        "d_model": 64,
        "nhead": 4,
        "num_layers": 2,
        "dim_feedforward": 256,
        "layer_norm_eps": 1e-5,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
    }
    print(f"[INFO] Hyperparameters: {hyperparameters}")

    print("[INFO] Loading test data...")
    train_loader, _, test_data_x, test_data_y, scaler_x, scaler_y = load_data(batch_params)
    print(f"[INFO] Test data loaded: X shape: {test_data_x.shape}, Y shape: {test_data_y.shape}")

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Path to saved model
    checkpoints_dir = os.path.join(project_root, "checkpoints")
    model_name = "transformer_2025-03-09_15-18-32.pth"
    model_path = os.path.join(checkpoints_dir, model_name)
    print(f"[INFO] Loading model from: {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[ERROR] Model file {model_name} not found in {checkpoints_dir}")

    # Construct model architecture exactly matching trained model
    model = TransformerModel(
        input_dim=train_loader.dataset[0][0].shape[-1],
        output_dim=train_loader.dataset[0][1].shape[-1],
        seq_len=batch_params['total_len'] // batch_params['gap'],
        d_model=hyperparameters['d_model'],
        nhead=hyperparameters['nhead'],
        num_layers=hyperparameters['num_layers'],
        dim_feedforward=hyperparameters['dim_feedforward'],
        dropout=hyperparameters['dropout'],
        layer_norm_eps=hyperparameters['layer_norm_eps']
    )

    print("[INFO] Model architecture initialized.")

    # Load weights and move to device
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("[INFO] Model loaded successfully.")

    # Convert test data to tensors
    X_test_tensor = torch.tensor(test_data_x, dtype=torch.float32)
    Y_test_tensor = torch.tensor(test_data_y, dtype=torch.float32)

    # Build a DataLoader for test data in small mini‐batches
    test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
    # Pick a batch_size that fits your GPU; 512 is just a starting guess
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)  

    print("[INFO] Generating predictions in mini-batches...")

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(Y_batch.numpy())

    # Concatenate all mini-batch results
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)

    print(f"[INFO] Combined predictions shape: {y_pred.shape}")

    # Compute MSE in original scale
    inversed_pred = scaler_y.inverse_transform(y_pred)
    inversed_true = scaler_y.inverse_transform(y_true)
    mse = mean_squared_error(inversed_true, inversed_pred)
    print(f"[INFO] Computed MSE: {mse}")

    # Save MSE results
    logs_dir = os.path.join(project_root, "logs", "test_logs")
    os.makedirs(logs_dir, exist_ok=True)
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    mse_log_path = os.path.join(logs_dir, f"test_logs_{current_datetime}.csv")

    mse_df = pd.DataFrame({
        "Metric": ["MSE"] + list(hyperparameters.keys()),
        "Value": [mse] + list(hyperparameters.values())
    })
    mse_df.to_csv(mse_log_path, index=False)
    print(f"[INFO] Test MSE and hyperparameters logged at {mse_log_path}")

    # Create a plot of a slice of predictions vs. ground truth
    plots_dir = os.path.join(project_root, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    slice_size = min(100, len(y_true))  # only plot up to 100 points
    t_slice = np.arange(slice_size)
    pred_slice = y_pred[:slice_size]
    true_slice = y_true[:slice_size]

    plt.figure(figsize=(12, 6))
    for j in range(true_slice.shape[1]):
        plt.plot(t_slice, true_slice[:, j], label=f"Ground Truth Mz{j+1}", linestyle="dotted")
        plt.plot(t_slice, pred_slice[:, j], label=f"Prediction Mz{j+1}")

    plt.xlabel("Time Steps")
    plt.ylabel("Torque Values")
    plt.title("Predictions vs Ground Truth (Sample Slice)")
    plt.legend()
    plt.grid()

    plot_path = os.path.join(plots_dir, f"test_plot_{current_datetime}.png")
    plt.savefig(plot_path, dpi=300)
    print(f"[INFO] Test plot saved at {plot_path}")
    print("[INFO] Evaluation script completed successfully.")

if __name__ == "__main__":
    main()
