import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from models.RBF_Keras import RBFLayer  # hvis du har brug for custom_objects

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
#from models.XGBoost import XGBoostModel
from models.RadialBasisFunctionModel import RBFN_model


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
    plot_results(y_true, y_pred, scaler_y, project_root, model_name="transformer_latest.pth", model_type="Transformer")

    return mse
"""
def evaluate_xgboost(batch_params, hyperparameters, model_name):
    print("[INFO] Evaluating XGBoost model...")

    # Load test data
    _, _, _, xgb_data, scaler_x, scaler_y = prepare_dataloaders(batch_params)

    # Initialize XGBoost model
    xgb_model = XGBoostModel(
        n_estimators=hyperparameters.get("n_estimators", 200),
        max_depth=hyperparameters.get("max_depth", 6),
        learning_rate=hyperparameters.get("learning_rate", 0.05)
    )

    # Load model weights if saved
    checkpoints_dir = os.path.join(project_root, "checkpoints")
    model_path = os.path.join(checkpoints_dir, model_name)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[ERROR] XGBoost model file {model_name} not found in {checkpoints_dir}")

    xgb_model.model.load_model(model_path)
    print("[INFO] XGBoost model loaded successfully.")

    # Generate predictions
    y_pred = xgb_model.predict(xgb_data["X_test"])

    # Compute MSE with inverse transformation
    inversed_pred = scaler_y.inverse_transform(y_pred)
    inversed_true = scaler_y.inverse_transform(xgb_data["y_test"])
    mse = mean_squared_error(inversed_true, inversed_pred)

    print(f"[INFO] Computed MSE: {mse}")

    # Save MSE results
    logs_dir = os.path.join(project_root, "logs", "test_logs")
    os.makedirs(logs_dir, exist_ok=True)
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    mse_log_path = os.path.join(logs_dir, f"xg_boost_logs_{current_datetime}.csv")

    #FIX THISSSSSSSSSSSSSSS
    mse_df = pd.DataFrame({
        "Metric": ["MSE"] + list(hyperparameters.keys()),
        "Value": [mse] + list(hyperparameters.values())
    })
    mse_df.to_csv(mse_log_path, index=False)
    print(f"[INFO] Test MSE and hyperparameters logged at {mse_log_path}")
    print(f"[INFO] XGBoost Model MSE: {mse}")

    # Call the separate plot function
    plot_results(inversed_true, inversed_pred, scaler_y, project_root, model_name="xgboost_latest.json", model_type="XGBoost")

    return mse
"""

def evaluate_rbf(batch_params, hyperparameters, model_name="rbfn_latest.npz"):
    print("[INFO] Evaluating RBFN model...")

    # load test data
    # xgb data holds x_test, y_test in flattend form exactly whar rbfn wants
    _, _, _, xgb_data, scaler_x, scaler_y = prepare_dataloaders(batch_params)

    X_test = xgb_data["X_test"]
    y_test = xgb_data["y_test"]
    print(f"[INFO] Test data shape: X={X_test.shape}, y={y_test.shape}")

    input_dim  = X_test.shape[1]
    output_dim = y_test.shape[1]
    num_hidden_neurons = hyperparameters.get("num_hidden_neurons", 50)
    learning_rate      = hyperparameters.get("learning_rate", 0.01)


    RBFN = RBFN_model(input_dim, num_hidden_neurons, output_dim, learning_rate=learning_rate)

    # loading saved parameters (centroids, betas, weights) from .npz

    checkpoints_dir = os.path.join(project_root, "checkpoints")
    model_path = os.path.join(checkpoints_dir, model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[ERROR] RBFN model file {model_name} not found in {checkpoints_dir}")

    
    print(f"[INFO] Loading RBFN parameters from {model_path}")
    npz_file = np.load(model_path)
    RBFN.centroids = npz_file["centroids"]
    RBFN.betas     = npz_file["betas"]
    RBFN.weights   = npz_file["weights"]
    print("[INFO] RBFN parameters loaded successfully.")

    print("[INFO] Generating RBFN predictions...")
    y_pred = RBFN.predict(X_test)

    # calculate MSE in original scale
    inversed_true = scaler_y.inverse_transform(y_test)
    inversed_pred = scaler_y.inverse_transform(y_pred)
    mse = mean_squared_error(inversed_true, inversed_pred)
    print(f"[INFO] Computed RBFN MSE on test: {mse}")

    # log  MSE & hyperparams
    logs_dir = os.path.join(project_root, "logs", "test_logs")
    os.makedirs(logs_dir, exist_ok=True)
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    mse_log_path = os.path.join(logs_dir, f"rbfn_logs_{current_datetime}.csv")

    mse_df = pd.DataFrame({
        "Metric": ["MSE"] + list(hyperparameters.keys()),
        "Value": [mse] + list(hyperparameters.values())
    })

    mse_df.to_csv(mse_log_path, index=False)
    print(f"[INFO] Test MSE and hyperparameters logged at {mse_log_path}")

    plot_results_rbf(y_test, y_pred, scaler_y, project_root, model_name=model_name, model_type="RBFN")
    
    return mse


def evaluate_rbf_keras(batch_params, hyperparameters, model_name="rbfn_keras_latest.keras"):
    print("[INFO] Evaluating Keras-RBFN model...")

    # load test data
    _, _, _, xgb_data, scaler_x, scaler_y = prepare_dataloaders(batch_params)
    X_test = xgb_data["X_test"]
    y_test = xgb_data["y_test"]
    print(f"[INFO] Test data shape: X={X_test.shape}, y={y_test.shape}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    checkpoints_dir = os.path.join(project_root, "checkpoints")
    model_path = os.path.join(checkpoints_dir, model_name)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[ERROR] Keras-RBFN model file {model_name} not found in {checkpoints_dir}")

    # For at loade custom-lag:
    # custom_objects = {"RBFLayer": RBFLayer} – ellers kender TF ikke din custom-lag
    loaded_model = tf.keras.models.load_model(model_path, custom_objects={"RBFLayer": RBFLayer})
    print("[INFO] Keras-RBFN model loaded successfully.")

    # predict
    y_pred = loaded_model.predict(X_test)

    # evaluate
    inversed_true = scaler_y.inverse_transform(y_test)
    inversed_pred = scaler_y.inverse_transform(y_pred)
    mse = mean_squared_error(inversed_true, inversed_pred)
    print(f"[INFO] Computed Keras-RBFN MSE on test: {mse}")

    # log & plot
    logs_dir = os.path.join(project_root, "logs", "test_logs")
    os.makedirs(logs_dir, exist_ok=True)
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    mse_log_path = os.path.join(logs_dir, f"rbfn_keras_logs_{current_datetime}.csv")

    mse_df = pd.DataFrame({
        "Metric": ["MSE"] + list(hyperparameters.keys()),
        "Value": [mse] + list(hyperparameters.values())
    })
    mse_df.to_csv(mse_log_path, index=False)
    print(f"[INFO] Test MSE and hyperparameters logged at {mse_log_path}")

    # Du kan genbruge dit plot_results_rbf eller lign.:
    plot_results_rbf(y_test, y_pred, scaler_y, project_root,
                     model_name=model_name, model_type="Keras-RBFN")
    
    return mse




# a plot function for rbf features/nature
def plot_results_rbf(y_true, y_pred, scaler_y, project_root, model_name, model_type):
    print("[INFO] Generating plot for RBFN predictions vs ground truth...")

    plots_dir = os.path.join(project_root, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Inverse transform
    y_true_orig = scaler_y.inverse_transform(y_true)
    y_pred_orig = scaler_y.inverse_transform(y_pred)

    sample_size = min(1600, len(y_true_orig))
    time_labels = np.linspace(0, 10, num=sample_size)  # Samme logik som i de andre

    pred_sample = y_pred_orig[:sample_size]
    true_sample = y_true_orig[:sample_size]

    plt.figure(figsize=(12, 6))
    for j in range(true_sample.shape[1]):
        plt.plot(time_labels, true_sample[:, j], 
                 label=f"{model_type} Ground Truth Mz{j+1}", linestyle="dotted")
        plt.plot(time_labels, pred_sample[:, j], 
                 label=f"{model_type} Prediction Mz{j+1}")

    plt.xlabel("Time (seconds)")
    plt.ylabel("Torque Values")
    plt.title(f"{model_type} Predictions vs Ground Truth (First 10 Seconds)")
    plt.legend()
    plt.grid()

    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plot_filename = f"{model_type.lower()}_plot_{current_datetime}.png"
    plot_path = os.path.join(plots_dir, plot_filename)
    
    plt.savefig(plot_path, dpi=300)
    print(f"[INFO] {model_type} plot saved at {plot_path}")


def plot_results(y_true, y_pred, scaler_y, project_root, model_name, model_type):
    """Function to generate and save the plot of predictions vs ground truth with correct torque values."""
    print("[INFO] Generating plot for predictions vs ground truth...")

    plots_dir = os.path.join(project_root, "plots")
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
        #"learning_rate": 1e-4,
        "weight_decay": 1e-4,

        # RBF-hyperparametre
        "num_hidden_neurons": 100,
        "learning_rate": 0.01,

    }   
    
    # Set model names
    transformer_model_name = "transformer_latest.pth"
    xgboost_model_name = "xgboost_latest.json"
    rbfn_model_name = "rbfn_latest.npz"

    #decide which model
    evaluate_transformer_flag = False
    evaluate_xgboost_flag = False
    evaluate_rbfn_flag = True  # Sæt True for at teste RBFN

    if evaluate_transformer_flag:
        evaluate_transformer(batch_params, hyperparameters, transformer_model_name)

    if evaluate_xgboost_flag:
        evaluate_xgboost(batch_params, hyperparameters, xgboost_model_name)

    if evaluate_rbfn_flag:
        #evaluate_rbf(batch_params, hyperparameters, rbfn_model_name)
        evaluate_rbf_keras(batch_params, hyperparameters, "rbfn_keras_latest.keras")