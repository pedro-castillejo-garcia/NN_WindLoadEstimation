import os
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_squared_error
from torch.utils.data import TensorDataset, DataLoader

from features_new_test_data import prepare_dataloaders_new_test_data
from hyperparameters import batch_parameters, hyperparameters

# Get the absolute path of the project's root directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))  # Move up one level

# Add project root to sys.path
sys.path.append(project_root)

# Add the models you want to evaluate with the new data here
from models.FFNN import FFNNModel

def evaluate_ffnn_new_test_data(batch_parameters, hyperparameters, model_name):
    print("Evaluating FFNN")

    test_loader, scaler_x, scaler_y, source_tensor = prepare_dataloaders_new_test_data(batch_parameters)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    ffnn_model = FFNNModel(
        input_dim=test_loader.dataset[0][0].shape[-1],
        output_dim=test_loader.dataset[0][1].shape[-1],
        seq_len=batch_parameters['total_len'] // batch_parameters['gap']
    )
    
    # Load model weights if saved
    checkpoints_dir = os.path.join(project_root, "checkpoints")
    model_path = os.path.join(checkpoints_dir, model_name)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[ERROR] FFNN model file {model_name} not found in {checkpoints_dir}")

    ffnn_model.load_state_dict(torch.load(model_path, map_location=device))  # Correct way to load model weights
    ffnn_model.to(device)
    ffnn_model.eval()
    
    all_preds, all_targets, all_times, all_sources = [], [], [], []

    with torch.no_grad():
        for idx, (X_batch, y_batch, t_batch, source_idx) in enumerate(test_loader):
            X_batch = X_batch.to(device)
            predictions = ffnn_model(X_batch).cpu().numpy()
            all_preds.append(predictions)
            all_targets.append(y_batch.numpy())
            all_times.append(t_batch.numpy())
            all_sources.append(source_idx.numpy()) 

    # Concatenate all mini-batch results
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    time_values = np.concatenate(all_times, axis=0)[:, 0, 0]
    source_indices = np.concatenate(all_sources, axis=0)

    print(f"[INFO] Combined predictions shape: {y_pred.shape}")

    # Compute MSE in original scale
    inversed_pred = scaler_y.inverse_transform(y_pred)
    inversed_true = scaler_y.inverse_transform(y_true)
    
    # Save the results DataFrame as a CSV file
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
    # Create full DataFrame
    results_df = pd.DataFrame({
        "Time": time_values,
        "Actual_Mz1": scaler_y.inverse_transform(y_true)[:, 0],
        "Predicted_Mz1": scaler_y.inverse_transform(y_pred)[:, 0],
        "Actual_Mz2": scaler_y.inverse_transform(y_true)[:, 1],
        "Predicted_Mz2": scaler_y.inverse_transform(y_pred)[:, 1],
        "Actual_Mz3": scaler_y.inverse_transform(y_true)[:, 2],
        "Predicted_Mz3": scaler_y.inverse_transform(y_pred)[:, 2],
        "File": [source_tensor[i] for i in source_indices]
    })
    
    results_dir = os.path.join(project_root, "results_new_test_data")
    os.makedirs(results_dir, exist_ok=True)

    for file_name, group in results_df.groupby("File"):
        clean_name = os.path.splitext(file_name)[0]
        save_path = os.path.join(results_dir, f"{clean_name}_new_test_data_results.csv")
        group.drop(columns="File").to_csv(save_path, index=False)
        print(f"[INFO] Saved: {save_path}")

    mse = mean_squared_error(inversed_true, inversed_pred)
    print(f"FFNN Evaluation MSE: {mse:.4f}")
    
    # Save MSE results
    logs_dir = os.path.join(project_root, "logs", "test_new_data_logs")
    os.makedirs(logs_dir, exist_ok=True)
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    mse_log_path = os.path.join(logs_dir, f"ffnn_test_new_data_logs_{current_datetime}.csv")

    mse_df = pd.DataFrame({
        "Metric": ["MSE"] + list(hyperparameters.keys()) + list(batch_parameters.keys()),
        "Value": [mse] + list(hyperparameters.values()) + list(batch_parameters.values())
    })
    mse_df.to_csv(mse_log_path, index=False)
    print(f"[INFO] Test MSE and hyperparameters logged at {mse_log_path}")
    
    # Ensure plot_results function is correctly defined
    plot_results(y_true, y_pred, scaler_y, model_name, "FFNN", mse)

def plot_results(y_true, y_pred, scaler_y, model_name, model_type, mse):
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
    plot_filename = f"{model_type.lower()}_plot_new_data_mse_{mse:.2f}_{current_datetime}.png"
    plot_path = os.path.join(plots_dir, plot_filename)
    
    plt.savefig(plot_path, dpi=300)
    print(f"[INFO] {model_type} plot saved at {plot_path}")    
    
if __name__ == "__main__":
        
    # Set model names
    ffnn_model_name = "ffnn_latest.pth"

    # DO THIS FOR EVERY MODEL YOU WANT TO EVALUATE    
    evaluate_ffnn_flag = True
        
    if evaluate_ffnn_flag:
        evaluate_ffnn_new_test_data(batch_parameters, hyperparameters, ffnn_model_name)
    

            