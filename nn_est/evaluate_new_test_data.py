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
from models.Transformer import TransformerModel
from models.TCN import TCNModel
from models.CNNLSTM import CNNLSTMModel
from models.LSTM import LSTMModel

def evaluate_ffnn_new_test_data(batch_parameters, hyperparameters, model_name, max_files=None):
    print("[INFO] Evaluating FFNN")

    test_loader, scaler_x, scaler_y, source_tensor = prepare_dataloaders_new_test_data(batch_parameters, max_files=max_files)
    
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
        save_path = os.path.join(results_dir, f"{clean_name}_new_test_data_ffnn_results.csv")
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


def evaluate_transformer_new_test_data(batch_parameters, hyperparameters, model_name, max_files=None):
    print("[INFO] Evaluating Transformer model on new test data...")

    test_loader, scaler_x, scaler_y, source_tensor = prepare_dataloaders_new_test_data(batch_parameters, max_files=max_files)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    transformer_model = TransformerModel(
        input_dim=test_loader.dataset.tensors[0].shape[-1],
        output_dim=test_loader.dataset.tensors[1].shape[-1],
        seq_len=batch_parameters['total_len'] // batch_parameters['gap'],
        d_model=hyperparameters['d_model'],
        nhead=hyperparameters['nhead'],
        num_layers=hyperparameters['num_layers'],
        dim_feedforward=hyperparameters['dim_feedforward'],
        dropout=hyperparameters['dropout'],
        layer_norm_eps=hyperparameters['layer_norm_eps']
    )

    model_path = os.path.join(project_root, "checkpoints", model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[ERROR] Model file {model_name} not found at {model_path}")

    transformer_model.load_state_dict(torch.load(model_path, map_location=device))
    transformer_model.to(device)
    transformer_model.eval()
    print("[INFO] Model loaded successfully.")

    # Create results directory
    results_dir = os.path.join(project_root, "results_new_test_data")
    os.makedirs(results_dir, exist_ok=True)

    # For logging MSE per file
    per_file_mse = {}
    file_data_map = {}

    print("[INFO] Collecting predictions and grouping by file...")

    with torch.no_grad():
        for i, (X_batch, y_batch, t_batch, source_idx_batch) in enumerate(test_loader):
            X_batch = X_batch.to(device)
            preds = transformer_model(X_batch).cpu().numpy()
            y_true = y_batch.numpy()
            t_vals = t_batch.numpy()[:, 0, 0]
            file_names = [source_tensor[idx] for idx in source_idx_batch.numpy()]

            for j in range(len(preds)):
                fname = file_names[j]
                if fname not in file_data_map:
                    file_data_map[fname] = {
                        "Time": [],
                        "Actual_Mz1": [],
                        "Predicted_Mz1": [],
                        "Actual_Mz2": [],
                        "Predicted_Mz2": [],
                        "Actual_Mz3": [],
                        "Predicted_Mz3": []
                    }

                # Inverse transform predictions and targets
                pred_inv = scaler_y.inverse_transform(preds[j].reshape(1, -1))[0]
                true_inv = scaler_y.inverse_transform(y_true[j].reshape(1, -1))[0]

                file_data_map[fname]["Time"].append(t_vals[j])
                file_data_map[fname]["Actual_Mz1"].append(true_inv[0])
                file_data_map[fname]["Predicted_Mz1"].append(pred_inv[0])
                file_data_map[fname]["Actual_Mz2"].append(true_inv[1])
                file_data_map[fname]["Predicted_Mz2"].append(pred_inv[1])
                file_data_map[fname]["Actual_Mz3"].append(true_inv[2])
                file_data_map[fname]["Predicted_Mz3"].append(pred_inv[2])

    print("[INFO] Saving results per file...")

    for fname, data in file_data_map.items():
        results_df = pd.DataFrame(data)
        mse = mean_squared_error(
            results_df[["Actual_Mz1", "Actual_Mz2", "Actual_Mz3"]],
            results_df[["Predicted_Mz1", "Predicted_Mz2", "Predicted_Mz3"]],
        )
        per_file_mse[fname] = mse

        name_base = os.path.splitext(fname)[0]
        output_path = os.path.join(results_dir, f"{name_base}_new_test_data_transformers_results.csv")
        results_df.to_csv(output_path, index=False)
        print(f"[INFO] Saved: {output_path}")

    # Save MSE summary
    log_dir = os.path.join(project_root, "logs", "test_new_data_logs")
    os.makedirs(log_dir, exist_ok=True)
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    mse_log_path = os.path.join(log_dir, f"transformer_test_new_data_logs_{current_datetime}.csv")

    pd.DataFrame(list(per_file_mse.items()), columns=["File", "MSE"]).to_csv(mse_log_path, index=False)
    print(f"[INFO] Per-file MSE summary saved to: {mse_log_path}")

    # Option: plot first file's results
    first_file = list(file_data_map.keys())[0]
    plot_df = pd.DataFrame(file_data_map[first_file])
    mse = per_file_mse[first_file]

    plot_results(
        y_true=scaler_y.transform(plot_df[["Actual_Mz1", "Actual_Mz2", "Actual_Mz3"]]),
        y_pred=scaler_y.transform(plot_df[["Predicted_Mz1", "Predicted_Mz2", "Predicted_Mz3"]]),
        scaler_y=scaler_y,
        model_name=model_name,
        model_type="Transformer",
        mse=mse
    )

def evaluate_tcn_new_test_data(batch_parameters, hyperparameters, model_name, max_files=None):
    print("[INFO] Evaluating TCN model on new test data...")

    test_loader, scaler_x, scaler_y, source_tensor = prepare_dataloaders_new_test_data(batch_parameters, max_files=max_files)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    model = TCNModel(
        input_dim=test_loader.dataset[0][0].shape[-1],
        output_dim=test_loader.dataset[0][1].shape[-1],
        seq_len=batch_parameters['total_len'] // batch_parameters['gap'],
        num_channels=hyperparameters['num_channels'],
        kernel_size=hyperparameters['kernel_size'],
        dropout=hyperparameters['dropout'],
        causal=hyperparameters['causal'],
        use_skip_connections=hyperparameters['use_skip_connections'],
        use_norm=hyperparameters['use_norm'],
        activation=hyperparameters['activation']
    )

    model_path = os.path.join(project_root, "checkpoints", model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[ERROR] Model file {model_name} not found at {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("[INFO] Model loaded successfully.")

    all_preds, all_targets, all_times, all_sources = [], [], [], []

    with torch.no_grad():
        for X_batch, y_batch, t_batch, source_idx in test_loader:
            X_batch = X_batch.permute(0, 2, 1).to(device)  # Permute needed for TCN!
            preds = model(X_batch).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(y_batch.numpy())
            all_times.append(t_batch.numpy())
            all_sources.append(source_idx.numpy())

    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    time_values = np.concatenate(all_times, axis=0)[:, 0, 0]
    source_indices = np.concatenate(all_sources, axis=0)

    inversed_pred = scaler_y.inverse_transform(y_pred)
    inversed_true = scaler_y.inverse_transform(y_true)

    # Build full DataFrame
    results_df = pd.DataFrame({
        "Time": time_values,
        "Actual_Mz1": inversed_true[:, 0],
        "Predicted_Mz1": inversed_pred[:, 0],
        "Actual_Mz2": inversed_true[:, 1],
        "Predicted_Mz2": inversed_pred[:, 1],
        "Actual_Mz3": inversed_true[:, 2],
        "Predicted_Mz3": inversed_pred[:, 2],
        "File": [source_tensor[i] for i in source_indices]
    })

    results_dir = os.path.join(project_root, "results_new_test_data")
    os.makedirs(results_dir, exist_ok=True)

    for file_name, group in results_df.groupby("File"):
        clean_name = os.path.splitext(file_name)[0]
        save_path = os.path.join(results_dir, f"{clean_name}_new_test_data_tcn_results.csv")
        group.drop(columns="File").to_csv(save_path, index=False)
        print(f"[INFO] Saved: {save_path}")

    mse = mean_squared_error(inversed_true, inversed_pred)
    print(f"[INFO] TCN Evaluation MSE: {mse:.4f}")

    logs_dir = os.path.join(project_root, "logs", "test_new_data_logs")
    os.makedirs(logs_dir, exist_ok=True)
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    mse_log_path = os.path.join(logs_dir, f"tcn_test_new_data_logs_{current_datetime}.csv")

    mse_df = pd.DataFrame({
        "Metric": ["MSE"] + list(hyperparameters.keys()) + list(batch_parameters.keys()),
        "Value": [mse] + list(hyperparameters.values()) + list(batch_parameters.values())
    })
    mse_df.to_csv(mse_log_path, index=False)
    print(f"[INFO] Test MSE and hyperparameters logged at {mse_log_path}")

    # Plot
    plot_results(y_true, y_pred, scaler_y, model_name, "TCN", mse)


def evaluate_cnnlstm_new_test_data(batch_parameters, hyperparameters, model_name, max_files=None):
    print("[INFO] Evaluating CNN-LSTM model on new test data...")

    test_loader, scaler_x, scaler_y, source_tensor = prepare_dataloaders_new_test_data(batch_parameters, max_files=max_files)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    model = CNNLSTMModel(
        input_dim=test_loader.dataset[0][0].shape[-1],
        output_dim=test_loader.dataset[0][1].shape[-1],
        seq_len=batch_parameters['total_len'] // batch_parameters['gap'],
        cnn_filters=hyperparameters['cnn_filters'],
        lstm_hidden=hyperparameters['lstm_hidden'],
        dropout=hyperparameters['dropout'],
        dense_units=hyperparameters['dense_units']
    )

    model_path = os.path.join(project_root, "checkpoints", model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[ERROR] Model file {model_name} not found at {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("[INFO] Model loaded successfully.")

    all_preds, all_targets, all_times, all_sources = [], [], [], []

    with torch.no_grad():
        for X_batch, y_batch, t_batch, source_idx in test_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(y_batch.numpy())
            all_times.append(t_batch.numpy())
            all_sources.append(source_idx.numpy())

    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    time_values = np.concatenate(all_times, axis=0)[:, 0, 0]
    source_indices = np.concatenate(all_sources, axis=0)

    inversed_pred = scaler_y.inverse_transform(y_pred)
    inversed_true = scaler_y.inverse_transform(y_true)

    results_df = pd.DataFrame({
        "Time": time_values,
        "Actual_Mz1": inversed_true[:, 0],
        "Predicted_Mz1": inversed_pred[:, 0],
        "Actual_Mz2": inversed_true[:, 1],
        "Predicted_Mz2": inversed_pred[:, 1],
        "Actual_Mz3": inversed_true[:, 2],
        "Predicted_Mz3": inversed_pred[:, 2],
        "File": [source_tensor[i] for i in source_indices]
    })

    results_dir = os.path.join(project_root, "results_new_test_data")
    os.makedirs(results_dir, exist_ok=True)

    for file_name, group in results_df.groupby("File"):
        clean_name = os.path.splitext(file_name)[0]
        save_path = os.path.join(results_dir, f"{clean_name}_new_test_data_cnnlstm_results.csv")
        group.drop(columns="File").to_csv(save_path, index=False)
        print(f"[INFO] Saved: {save_path}")

    mse = mean_squared_error(inversed_true, inversed_pred)
    print(f"[INFO] CNN-LSTM Evaluation MSE: {mse:.4f}")

    logs_dir = os.path.join(project_root, "logs", "test_new_data_logs")
    os.makedirs(logs_dir, exist_ok=True)
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    mse_log_path = os.path.join(logs_dir, f"cnn_lstm_test_new_data_logs_{current_datetime}.csv")

    mse_df = pd.DataFrame({
        "Metric": ["MSE"] + list(hyperparameters.keys()) + list(batch_parameters.keys()),
        "Value": [mse] + list(hyperparameters.values()) + list(batch_parameters.values())
    })
    mse_df.to_csv(mse_log_path, index=False)
    print(f"[INFO] Test MSE and hyperparameters logged at {mse_log_path}")

    # Plot
    plot_results(y_true, y_pred, scaler_y, model_name, "CNN-LSTM", mse)


def evaluate_lstm_new_test_data(batch_parameters, hyperparameters, model_name, max_files=None):
    print("[INFO] Evaluating LSTM model on new test data...")

    test_loader, scaler_x, scaler_y, source_tensor = prepare_dataloaders_new_test_data(batch_parameters, max_files=max_files)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Build model
    model = LSTMModel(
        input_dim=test_loader.dataset[0][0].shape[-1],
        output_dim=test_loader.dataset[0][1].shape[-1],
        seq_len=batch_parameters['total_len'] // batch_parameters['gap'],
        lstm_hidden=hyperparameters['lstm_hidden'],
        num_layers=hyperparameters['num_layers_lstm'],
        dropout=hyperparameters['dropout'],
        dense_units=hyperparameters['dense_units']
    )

    # Load weights
    model_path = os.path.join(project_root, "checkpoints", model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[ERROR] Model file {model_name} not found at {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("[INFO] Model loaded successfully.")

    # Predict
    all_preds, all_targets, all_times, all_sources = [], [], [], []

    with torch.no_grad():
        for X_batch, y_batch, t_batch, source_idx in test_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(y_batch.numpy())
            all_times.append(t_batch.numpy())
            all_sources.append(source_idx.numpy()) 

    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    time_values = np.concatenate(all_times, axis=0)[:, 0, 0]
    source_indices = np.concatenate(all_sources, axis=0)

    print(f"[INFO] Combined predictions shape: {y_pred.shape}")

    inversed_pred = scaler_y.inverse_transform(y_pred)
    inversed_true = scaler_y.inverse_transform(y_true)

    # Build results dataframe
    results_df = pd.DataFrame({
        "Time": time_values,
        "Actual_Mz1": inversed_true[:, 0],
        "Predicted_Mz1": inversed_pred[:, 0],
        "Actual_Mz2": inversed_true[:, 1],
        "Predicted_Mz2": inversed_pred[:, 1],
        "Actual_Mz3": inversed_true[:, 2],
        "Predicted_Mz3": inversed_pred[:, 2],
        "File": [source_tensor[i] for i in source_indices]
    })

    results_dir = os.path.join(project_root, "results_new_test_data")
    os.makedirs(results_dir, exist_ok=True)

    for file_name, group in results_df.groupby("File"):
        clean_name = os.path.splitext(file_name)[0]
        save_path = os.path.join(results_dir, f"{clean_name}_new_test_data_lstm_results.csv")
        group.drop(columns="File").to_csv(save_path, index=False)
        print(f"[INFO] Saved: {save_path}")

    mse = mean_squared_error(inversed_true, inversed_pred)
    print(f"[INFO] LSTM Evaluation MSE: {mse:.4f}")

    # Save MSE summary
    logs_dir = os.path.join(project_root, "logs", "test_new_data_logs")
    os.makedirs(logs_dir, exist_ok=True)
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    mse_log_path = os.path.join(logs_dir, f"lstm_test_new_data_logs_{current_datetime}.csv")

    mse_df = pd.DataFrame({
        "Metric": ["MSE"] + list(hyperparameters.keys()) + list(batch_parameters.keys()),
        "Value": [mse] + list(hyperparameters.values()) + list(batch_parameters.values())
    })
    mse_df.to_csv(mse_log_path, index=False)
    print(f"[INFO] Test MSE and hyperparameters logged at {mse_log_path}")

    # Plot results
    plot_results(y_true, y_pred, scaler_y, model_name, "LSTM", mse)

    
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
    transformer_model_name = "transformer_latest.pth"
    tcn_model_name = "TCN_1_latest.pth"
    cnn_lstm_model_name = "CNN-LSTM_1_latest.pth"
    lstm_model_name = "LSTM_1_latest.pth"

    # DO THIS FOR EVERY MODEL YOU WANT TO EVALUATE    
    evaluate_ffnn_flag = False
    evaluate_transformer_flag = True
    evaluate_tcn_flag = False
    evaluate_cnnlstm_flag = False
    evaluate_lstm_flag = True
    
    # Change the max values accordingly to how many of the new test data csv files you want to evaluate    
    if evaluate_ffnn_flag:
        evaluate_ffnn_new_test_data(batch_parameters, hyperparameters, ffnn_model_name, max_files=2)
        
    if evaluate_transformer_flag:
        evaluate_transformer_new_test_data(batch_parameters, hyperparameters, transformer_model_name, max_files=25)
    
    if evaluate_tcn_flag:
        evaluate_tcn_new_test_data(batch_parameters, hyperparameters, tcn_model_name, max_files=25)

    if evaluate_cnnlstm_flag:
        evaluate_cnnlstm_new_test_data(batch_parameters, hyperparameters, cnn_lstm_model_name, max_files=25)

    if evaluate_lstm_flag:
        evaluate_lstm_new_test_data(batch_parameters, hyperparameters, lstm_model_name, max_files=25)

            