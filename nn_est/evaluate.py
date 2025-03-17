import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_squared_error
from torch.utils.data import TensorDataset, DataLoader
import sys
from features import prepare_dataloaders

# Get the absolute path of the project's root directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))  # Move up one level

# Add project root to sys.path
sys.path.append(project_root)

from models.Transformer import TransformerModel
from models.XGBoost import XGBoostModel
from models.FFNN import FFNNModel

def evaluate_transformer(batch_params, hyperparameters, model_name):
    print("[INFO] Evaluating Transformer model...")

    # Load test data
    train_loader, val_loader, test_loader, _, scaler_x, scaler_y = prepare_dataloaders(batch_params)
    
    test_data_x = test_loader.dataset.tensors[0].numpy()
    test_data_y = test_loader.dataset.tensors[1].numpy()

    print(f"[INFO] Test data loaded: X shape: {test_data_x.shape}, Y shape: {test_data_y.shape}")

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Path to saved model
    checkpoints_dir = os.path.join(project_root, "checkpoints")
    model_path = os.path.join(checkpoints_dir, model_name)
    print(f"[INFO] Loading model from: {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[ERROR] Model file {model_name} not found in {checkpoints_dir}")

    # Construct model architecture exactly matching trained model
    model = TransformerModel(
        input_dim=train_loader.dataset.tensors[0].shape[-1],  
        output_dim=train_loader.dataset.tensors[1].shape[-1],  
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
    mse_log_path = os.path.join(logs_dir, f"transformer_logs_{current_datetime}.csv")

    mse_df = pd.DataFrame({
        "Metric": ["MSE"] + list(hyperparameters.keys()),
        "Value": [mse] + list(hyperparameters.values())
    })
    mse_df.to_csv(mse_log_path, index=False)
    print(f"[INFO] Test MSE and hyperparameters logged at {mse_log_path}")

    # Call the separate plot function
    plot_results(y_true, y_pred, scaler_y, model_name, "Transformer")

    return mse

def evaluate_xgboost(batch_params, hyperparameters, model_name):
    print("[INFO] Evaluating XGBoost model...")

    # Load test data
    _, _, _, xgb_data, scaler_x, scaler_y = prepare_dataloaders(batch_params)

    # Load trained XGBoost model
    xgb_model = XGBoostModel(
        n_estimators=hyperparameters["n_estimators"],
        max_depth=hyperparameters["max_depth"],
        learning_rate=hyperparameters["learning_rate"],
        objective="reg:squarederror"
    )

    # Load model weights if saved
    checkpoints_dir = os.path.join(project_root, "checkpoints")
    model_path = os.path.join(checkpoints_dir, model_name)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[ERROR] XGBoost model file {model_name} not found in {checkpoints_dir}")

    xgb_model.load_model(model_path)

    # Generate predictions
    y_pred = xgb_model.predict(xgb_data["X_test"])

    # Ensure correct shape
    y_pred = y_pred.reshape(-1, 3)

    # Compute MSE
    inversed_pred = scaler_y.inverse_transform(y_pred)
    inversed_true = scaler_y.inverse_transform(xgb_data["y_test"])
    
    mse = mean_squared_error(inversed_true, inversed_pred)

    print(f"[INFO] XGBoost Model MSE: {mse}")

    # Save MSE results
    logs_dir = os.path.join(project_root, "logs", "test_logs")
    os.makedirs(logs_dir, exist_ok=True)
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    mse_log_path = os.path.join(logs_dir, f"xg_boost_logs_{current_datetime}.csv")

    mse_df = pd.DataFrame({
        "Metric": ["MSE"] + list(hyperparameters.keys()),
        "Value": [mse] + list(hyperparameters.values())
    })
    mse_df.to_csv(mse_log_path, index=False)
    print(f"[INFO] Test MSE and hyperparameters logged at {mse_log_path}")

    # Plot results
    plot_results(xgb_data["y_test"], y_pred, scaler_y, model_name, "XGBoost")

    return mse

def evaluate_ffnn(batch_params, hyperparameters, model_name):
    print("Evaluating FFNN")

    _, _, test_loader, _, scaler_x, scaler_y = prepare_dataloaders(batch_params)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    ffnn_model = FFNNModel(
        input_dim=test_loader.dataset[0][0].shape[-1],
        output_dim=test_loader.dataset[0][1].shape[-1],
        seq_len=batch_params['total_len'] // batch_params['gap']
    )
    
    # Load model weights if saved
    checkpoints_dir = os.path.join(project_root, "checkpoints")
    model_path = os.path.join(checkpoints_dir, model_name)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[ERROR] FFNN model file {model_name} not found in {checkpoints_dir}")

    ffnn_model.load_state_dict(torch.load(model_path, map_location=device))  # Correct way to load model weights
    ffnn_model.to(device)
    ffnn_model.eval()
    
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            predictions = ffnn_model(X_batch).cpu().numpy()
            all_preds.append(predictions)
            all_targets.append(y_batch.numpy())

    y_pred = scaler_y.inverse_transform(np.concatenate(all_preds, axis=0))
    y_true = scaler_y.inverse_transform(np.concatenate(all_targets, axis=0))
    
    mse = ((y_pred - y_true) ** 2).mean()
    print(f"FFNN Evaluation MSE: {mse:.4f}")
    
    # Ensure plot_results function is correctly defined
    plot_results(y_true, y_pred, scaler_y, model_name, "FFNN")
    
def plot_results(y_true, y_pred, scaler_y, model_name, model_type):
    """Function to generate and save the plot of predictions vs ground truth with correct torque values."""
    print("[INFO] Generating plot for predictions vs ground truth...")
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # Get main project root
    plots_dir = os.path.join(project_root, "plots")  # Ensure plots go into the main project root
    os.makedirs(plots_dir, exist_ok=True)

    # Inverse transform to restore original scale of torque values
    y_true_original = scaler_y.inverse_transform(y_true)
    y_pred_original = scaler_y.inverse_transform(y_pred)

    sample_size = min(1600, len(y_true_original))  # Limit to first 1600 samples (10 seconds)
    time_labels = np.linspace(0, 10, num=sample_size)  # Generate time labels from 0 to 10 sec

    pred_sample = y_pred_original[:sample_size]
    true_sample = y_true_original[:sample_size]

    plt.figure(figsize=(12, 6))
    for j in range(true_sample.shape[1]):
        plt.plot(time_labels, true_sample[:, j], label=f"{model_type} Ground Truth Mz{j+1}", linestyle="dotted")
        plt.plot(time_labels, pred_sample[:, j], label=f"{model_type} Prediction Mz{j+1}")

    plt.xlabel("Time (seconds)")
    plt.ylabel("Torque Values")
    plt.title(f"{model_type} Predictions vs Ground Truth (First 10 Seconds)")
    plt.legend()
    plt.grid()

    # Custom filename for each model type
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plot_filename = f"{model_type.lower()}_plot_{current_datetime}.png"
    plot_path = os.path.join(plots_dir, plot_filename)
    
    plt.savefig(plot_path, dpi=300)
    print(f"[INFO] {model_type} plot saved at {plot_path}")

# Example usage
if __name__ == "__main__":
    batch_params = {
        "gap": 10,
        "total_len": 100,
        "batch_size": 32
    }

    hyperparameters = {
        "dropout": 0.3,
        "d_model": 64,
        "nhead": 4,
        "num_layers": 2,
        "dim_feedforward": 256,
        "layer_norm_eps": 1e-5,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "n_estimators": 300,
        "max_depth": 8,
        "subsample": 0.8,       
        "colsample_bytree": 0.8,     
        "gamma": 0.1,  
    }
    
    # Set model names
    transformer_model_name = "transformer_latest.pth"
    xgboost_model_name = "xgboost_latest.json"
    ffnn_model_name = "ffnn_latest.pth"

    # Choose which model to evaluate
    evaluate_transformer_flag = True
    evaluate_xgboost_flag = True
    evaluate_ffnn_flag = True

    if evaluate_transformer_flag:
        evaluate_transformer(batch_params, hyperparameters, transformer_model_name)

    if evaluate_xgboost_flag:
        evaluate_xgboost(batch_params, hyperparameters, xgboost_model_name)
        
    if evaluate_ffnn_flag:
        evaluate_ffnn(batch_params, hyperparameters, ffnn_model_name)
        
