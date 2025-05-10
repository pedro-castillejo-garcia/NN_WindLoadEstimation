import os
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_squared_error
from torch.utils.data import TensorDataset, DataLoader


from features import prepare_dataloaders
from features import prepare_flat_dataloaders
from hyperparameters import batch_parameters, hyperparameters

# Get the absolute path of the project's root directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))  # Move up one level

# Add project root to sys.path
sys.path.append(project_root)

from models.Transformer import TransformerModel
from models.XGBoost import XGBoostModel
from models.FFNN import FFNNModel
from models.OneLayerNN import OneLayerNN
from models.TCN import TCNModel
from models.CNNLSTM import CNNLSTMModel
from models.LSTM import LSTMModel
from models.CNN import CNNModel
from models.RBF_PyTorch import RBFNet


def evaluate_transformer(batch_parameters, hyperparameters, model_name):
    print("[INFO] Evaluating Transformer model...")

    # Load test data
    train_loader, val_loader, test_loader, _, scaler_x, scaler_y = prepare_dataloaders(batch_parameters)
    
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
        seq_len=batch_parameters['total_len'] // batch_parameters['gap'],
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
    mse_log_path = os.path.join(logs_dir, f"transformer_test_logs_{current_datetime}.csv")

    mse_df = pd.DataFrame({
        "Metric": ["MSE"] + list(hyperparameters.keys()) + list(batch_parameters.keys()),
        "Value": [mse] + list(hyperparameters.values()) + list(batch_parameters.values())
    })
    mse_df.to_csv(mse_log_path, index=False)
    print(f"[INFO] Test MSE and hyperparameters logged at {mse_log_path}")

    # Call the separate plot function
    plot_results(y_true, y_pred, scaler_y, model_name, "Transformer", mse)

    return mse

def evaluate_xgboost(batch_parameters, hyperparameters, model_name):
    print("[INFO] Evaluating XGBoost model...")

    # Load test data
    _, _, _, xgb_data, scaler_x, scaler_y = prepare_dataloaders(batch_parameters)

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
    mse_log_path = os.path.join(logs_dir, f"xg_boost_test_logs_{current_datetime}.csv")

    mse_df = pd.DataFrame({
        "Metric": ["MSE"] + list(hyperparameters.keys()) + list(batch_parameters.keys()),
        "Value": [mse] + list(hyperparameters.values()) + list(batch_parameters.values())
    })
    mse_df.to_csv(mse_log_path, index=False)
    print(f"[INFO] Test MSE and hyperparameters logged at {mse_log_path}")

    # Plot results
    plot_results(xgb_data["y_test"], y_pred, scaler_y, model_name, "XGBoost", mse)

    return mse

def evaluate_ffnn(batch_parameters, hyperparameters, model_name):
    print("Evaluating FFNN")

    _, _, test_loader, _, scaler_x, scaler_y = prepare_dataloaders(batch_parameters)
    
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
    
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            predictions = ffnn_model(X_batch).cpu().numpy()
            all_preds.append(predictions)
            all_targets.append(y_batch.numpy())

    # Concatenate all mini-batch results
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)

    print(f"[INFO] Combined predictions shape: {y_pred.shape}")

    # Compute MSE in original scale
    inversed_pred = scaler_y.inverse_transform(y_pred)
    inversed_true = scaler_y.inverse_transform(y_true)
    
    mse = mean_squared_error(inversed_true, inversed_pred)
    print(f"FFNN Evaluation MSE: {mse:.4f}")
    
    # Save MSE results
    logs_dir = os.path.join(project_root, "logs", "test_logs")
    os.makedirs(logs_dir, exist_ok=True)
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    mse_log_path = os.path.join(logs_dir, f"ffnn_test_logs_{current_datetime}.csv")

    mse_df = pd.DataFrame({
        "Metric": ["MSE"] + list(hyperparameters.keys()) + list(batch_parameters.keys()),
        "Value": [mse] + list(hyperparameters.values()) + list(batch_parameters.values())
    })
    mse_df.to_csv(mse_log_path, index=False)
    print(f"[INFO] Test MSE and hyperparameters logged at {mse_log_path}")
    
    # Ensure plot_results function is correctly defined
    plot_results(y_true, y_pred, scaler_y, model_name, "FFNN", mse)
  
def evaluate_one_layer_nn(batch_parameters, model_name):
    print("Evaluating One-Layer NN")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader, _, scaler_x, scaler_y = prepare_dataloaders(batch_parameters)
    
    one_layer_nn_model = OneLayerNN(
        input_dim=test_loader.dataset[0][0].shape[-1],
        output_dim=test_loader.dataset[0][1].shape[-1],
        seq_len=batch_parameters['total_len'] // batch_parameters['gap']
    )
    
    # Load model weights if saved
    checkpoints_dir = os.path.join(project_root, "checkpoints")
    model_path = os.path.join(checkpoints_dir, model_name)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[ERROR] One-Layer NN model file {model_name} not found in {checkpoints_dir}")
    
    one_layer_nn_model.load_state_dict(torch.load(model_path, map_location=device))
    one_layer_nn_model.to(device)
    one_layer_nn_model.eval()
    
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            predictions = one_layer_nn_model(X_batch).cpu().numpy()
            all_preds.append(predictions)
            all_targets.append(y_batch.numpy())
    
    # Concatenate all mini-batch results
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)

    print(f"[INFO] Combined predictions shape: {y_pred.shape}")

    # Compute MSE in original scale
    inversed_pred = scaler_y.inverse_transform(y_pred)
    inversed_true = scaler_y.inverse_transform(y_true)
    
    mse = mean_squared_error(inversed_true, inversed_pred)
    print(f"One-Layer NN Evaluation MSE: {mse:.4f}")
    
    # Save MSE results
    logs_dir = os.path.join(project_root, "logs", "test_logs")
    os.makedirs(logs_dir, exist_ok=True)
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    mse_log_path = os.path.join(logs_dir, f"one_layer_nn_test_logs_{current_datetime}.csv")

    mse_df = pd.DataFrame({
        "Metric": ["MSE"] + list(hyperparameters.keys()) + list(batch_parameters.keys()),
        "Value": [mse] + list(hyperparameters.values()) + list(batch_parameters.values())
    })
    mse_df.to_csv(mse_log_path, index=False)
    print(f"[INFO] Test MSE and hyperparameters logged at {mse_log_path}")
    
    # Ensure plot_results function is correctly defined
    plot_results(y_true, y_pred, scaler_y, model_name, "One_Layer_NN", mse)


def evaluate_tcn(batch_params, hyperparameters, model_name):
    print("[INFO] Evaluating TCN model...")

    _, _, test_loader, _, scaler_x, scaler_y = prepare_dataloaders(batch_params)

    test_data_x = test_loader.dataset.tensors[0].numpy()
    test_data_y = test_loader.dataset.tensors[1].numpy()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Construct model architecture exactly matching trained model
    model = TCNModel(
        input_dim=test_loader.dataset.tensors[0].shape[-1],
        output_dim=test_loader.dataset.tensors[1].shape[-1],
        seq_len=batch_parameters['total_len'] // batch_parameters['gap'],
        num_channels=hyperparameters.get("num_channels", [32, 64, 64]),
        kernel_size=hyperparameters.get("kernel_size", 5),
        dropout=hyperparameters.get("dropout", 0.2),
        causal=hyperparameters.get("causal", True),
        use_skip_connections=hyperparameters.get("use_skip_connections", False),
        use_norm=hyperparameters.get("use_norm", "weight_norm"),
        activation=hyperparameters.get("activation", "relu")
    )

    # Path to saved model
    checkpoints_dir = os.path.join(project_root, "checkpoints")
    model_path = os.path.join(checkpoints_dir, model_name)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[ERROR] Model file {model_name} not found in {checkpoints_dir}")

    # Load weights and move to device
    model.load_state_dict(torch.load(os.path.join(project_root, "checkpoints", model_name)))
    model.to(device)
    model.eval()

    all_preds, all_targets = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.permute(0, 2, 1).to(device)
            preds = model(X_batch).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(y_batch.numpy())

    # Concatenate all mini-batch results
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)

    # Compute MSE in original scale
    inversed_pred = scaler_y.inverse_transform(y_pred)
    inversed_true = scaler_y.inverse_transform(y_true)
    mse = mean_squared_error(inversed_true, inversed_pred)

    # Save MSE results
    logs_dir = os.path.join(project_root, "logs", "test_logs")
    os.makedirs(logs_dir, exist_ok=True)
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    mse_log_path = os.path.join(logs_dir, f"TCN_test_logs_{current_datetime}.csv")

    mse_df = pd.DataFrame({
        "Metric": ["MSE"] + list(hyperparameters.keys()) + list(batch_parameters.keys()),
        "Value": [mse] + list(hyperparameters.values()) + list(batch_parameters.values())
    })
    mse_df.to_csv(mse_log_path, index=False)
    print(f"[INFO] Test MSE and hyperparameters logged at {mse_log_path}")

    # Call the separate plot function
    plot_results(y_true, y_pred, scaler_y, model_name, "TCN", mse)

    return mse


def evaluate_cnnlstm(batch_params, hyperparameters, model_name):
    print("[INFO] Evaluating CNN-LSTM model...")

    _, _, test_loader, _, scaler_x, scaler_y = prepare_dataloaders(batch_params)

    test_data_x = test_loader.dataset.tensors[0].numpy()
    test_data_y = test_loader.dataset.tensors[1].numpy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNNLSTMModel(
        input_dim=test_loader.dataset.tensors[0].shape[-1],
        output_dim=test_loader.dataset.tensors[1].shape[-1],
        seq_len=batch_params['total_len'] // batch_params['gap'],
        cnn_filters=hyperparameters.get("cnn_filters", 32),
        lstm_hidden=hyperparameters.get("lstm_hidden", 64),
        dropout=hyperparameters.get("dropout", 0.1),
        dense_units=hyperparameters.get("dense_units", 256)
    )

    # Path to saved model
    checkpoints_dir = os.path.join(project_root, "checkpoints")
    model_path = os.path.join(checkpoints_dir, model_name)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[ERROR] Model file {model_name} not found in {checkpoints_dir}")

    model.load_state_dict(torch.load(os.path.join(project_root, "checkpoints", model_name)))
    model.to(device)
    model.eval()

    all_preds, all_targets = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            predictions = model(X_batch).cpu().numpy()
            all_preds.append(predictions)
            all_targets.append(y_batch.numpy())

    # Concatenate all mini-batch results
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)

    print(f"[INFO] Combined predictions shape: {y_pred.shape}")

    # Compute MSE in original scale
    inversed_pred = scaler_y.inverse_transform(y_pred)
    inversed_true = scaler_y.inverse_transform(y_true)
    mse = mean_squared_error(inversed_true, inversed_pred)
    
    # Save MSE results
    logs_dir = os.path.join(project_root, "logs", "test_logs")
    os.makedirs(logs_dir, exist_ok=True)
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    mse_log_path = os.path.join(logs_dir, f"cnn-lstm_test_logs_{current_datetime}.csv")

    mse_df = pd.DataFrame({
        "Metric": ["MSE"] + list(hyperparameters.keys()) + list(batch_parameters.keys()),
        "Value": [mse] + list(hyperparameters.values()) + list(batch_parameters.values())
    })
    mse_df.to_csv(mse_log_path, index=False)
    print(f"[INFO] Test MSE and hyperparameters logged at {mse_log_path}")
    
    # Ensure plot_results function is correctly defined
    plot_results(y_true, y_pred, scaler_y, model_name, "CNN-LSTM", mse)


def evaluate_lstm(batch_parameters, hyperparameters, model_name):
    print("[INFO] Evaluating LSTM model...")

    # Load test data
    _, _, test_loader, _, scaler_x, scaler_y = prepare_dataloaders(batch_parameters)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Build model
    model = LSTMModel(
        input_dim=test_loader.dataset.tensors[0].shape[-1],
        output_dim=test_loader.dataset.tensors[1].shape[-1],
        seq_len=batch_parameters['total_len'] // batch_parameters['gap'],
        lstm_hidden=hyperparameters['lstm_hidden'],
        num_layers=hyperparameters['num_layers_lstm'],
        dropout=hyperparameters['dropout'],
        dense_units=hyperparameters['dense_units']
    )

    # Load trained weights
    checkpoints_dir = os.path.join(project_root, "checkpoints")
    model_path = os.path.join(checkpoints_dir, model_name)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[ERROR] Model file {model_name} not found in {checkpoints_dir}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("[INFO] Model loaded successfully.")

    # Predict
    all_preds, all_targets = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(y_batch.numpy())

    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)

    print(f"[INFO] Combined predictions shape: {y_pred.shape}")

    # Compute MSE in original scale
    inversed_pred = scaler_y.inverse_transform(y_pred)
    inversed_true = scaler_y.inverse_transform(y_true)
    mse = mean_squared_error(inversed_true, inversed_pred)

    print(f"[INFO] LSTM Evaluation MSE: {mse:.4f}")

    # Save MSE and hyperparameters
    logs_dir = os.path.join(project_root, "logs", "test_logs")
    os.makedirs(logs_dir, exist_ok=True)
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    mse_log_path = os.path.join(logs_dir, f"lstm_test_logs_{current_datetime}.csv")

    mse_df = pd.DataFrame({
        "Metric": ["MSE"] + list(hyperparameters.keys()) + list(batch_parameters.keys()),
        "Value": [mse] + list(hyperparameters.values()) + list(batch_parameters.values())
    })
    mse_df.to_csv(mse_log_path, index=False)
    print(f"[INFO] Test MSE and hyperparameters logged at {mse_log_path}")

    # Plot results
    plot_results(y_true, y_pred, scaler_y, model_name, "LSTM", mse)

    return mse

def evaluate_cnn(batch_params,
                 hyperparameters,
                 model_name="cnn_latest.pth"):
    # 1) data
    _, _, test_loader, _, scaler_x, scaler_y = prepare_dataloaders(batch_params)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2) model-genopbygning
    in_channels = test_loader.dataset.tensors[0].shape[-1]
    seq_len     = batch_params["total_len"] // batch_params["gap"]
    out_dim     = test_loader.dataset.tensors[1].shape[-1]

    model = CNNModel(in_channels=in_channels,
                     seq_length=seq_len,
                     num_outputs=out_dim).to(device)

    # 3) vægte
    ckpt = os.path.join(project_root, "checkpoints", model_name)
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    # 4) inference
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            preds.append(model(xb.to(device)).cpu().numpy())
            trues.append(yb.numpy())

    y_pred = np.concatenate(preds)
    y_true = np.concatenate(trues)

    # 5) MSE i original skala
    inv_pred = scaler_y.inverse_transform(y_pred)
    inv_true = scaler_y.inverse_transform(y_true)
    mse = mean_squared_error(inv_true, inv_pred)
    print(f"[CNN] Test MSE = {mse:.4f}")

    # 6) gem log som dine andre modeller
    logs_dir = os.path.join(project_root, "logs", "test_logs")
    os.makedirs(logs_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(logs_dir, f"cnn_logs_{ts}.csv")

    mse_df = pd.DataFrame({
        "Metric": ["MSE"] + list(hyperparameters.keys()) + list(batch_params.keys()),
        "Value":  [mse]   + list(hyperparameters.values()) + list(batch_params.values())
    })
    mse_df.to_csv(log_path, index=False)
    print(f"[INFO] Test MSE og hyperparametre gemt → {log_path}")

    # (valgfrit) plot
    plot_cnn_results(y_true, y_pred, scaler_y)
    return mse

def evaluate_rbf_pytorch(batch_params,
                         hyperparameters,
                         model_name="rbfpytorch_latest.pth"):
    """
    Evaluerer et trænet RBFNet (PyTorch) på test-split:
      • flader sekvenser ud   (batch, seq_len, feat) → (batch, seq_len*feat)
      • beregner MSE i originals­kala
      • logger MSE + alle hyper/batch-parametre til CSV
    """
    # ---------- data -------------------------------------------------------
    _, _, test_loader, _, scaler_x, scaler_y = prepare_dataloaders(batch_params)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    in_channels = test_loader.dataset.tensors[0].shape[-1]                # 12
    seq_len     = batch_params['total_len'] // batch_params['gap']        # fx 100
    flat_dim    = in_channels * seq_len                                   # 1200
    out_dim     = test_loader.dataset.tensors[1].shape[-1]                # 3

    # ---------- model ------------------------------------------------------
    model = RBFNet(
        input_dim          = flat_dim,
        num_hidden_neurons = hyperparameters['num_hidden_neurons'],
        output_dim         = out_dim,
        betas              = hyperparameters.get('beta_init', 0.5),
        initial_centroids  = None          # indlæses fra checkpoint
    ).to(device)

    ckpt = os.path.join(project_root, "checkpoints", model_name)
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    # ---------- inference --------------------------------------------------
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb_flat = xb.view(xb.size(0), -1).to(device)   # (B, seq*feat)
            preds.append(model(xb_flat).cpu().numpy())
            trues.append(yb.numpy())

    y_pred = np.concatenate(preds)
    y_true = np.concatenate(trues)

    # ---------- MSE i originals­kala --------------------------------------
    inv_pred = scaler_y.inverse_transform(y_pred)
    inv_true = scaler_y.inverse_transform(y_true)
    mse = mean_squared_error(inv_true, inv_pred)
    print(f"[RBF] Test MSE = {mse:.4f}")

    # ---------- log --------------------------------------------------------
    logs_dir = os.path.join(project_root, "logs", "test_logs")
    os.makedirs(logs_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(logs_dir, f"rbfpytorch_logs_{ts}.csv")

    mse_df = pd.DataFrame({
        "Metric": ["MSE"] + list(hyperparameters.keys()) + list(batch_params.keys()),
        "Value":  [mse]   + list(hyperparameters.values()) + list(batch_params.values())
    })
    mse_df.to_csv(log_path, index=False)
    print(f"[INFO] Test MSE og hyperparametre gemt → {log_path}")

    # (valgfrit) plot – genbrug din plot_results_rbf-funktion hvis du ønsker
    plot_results_rbf(y_true, y_pred, scaler_y, project_root,
                      model_name=model_name, model_type="RBF-PyTorch")

    return mse

def evaluate_rbf_pytorch_static(batch_params,
                                hyperparameters,
                                model_name="rbfpytorch_static.pth"):
    """
    Evaluerer den *statiske* (non-sekventielle) PyTorch-RBF:
      • loader data via prepare_flat_dataloaders
      • ingen gap/total_len – input_dim = antal features (12)
      • beregner MSE, logger og printer resultat
    """
    # ------------- data --------------------------------------------------
    train_loader, val_loader, xgb_data = prepare_flat_dataloaders(batch_params)
    X_test = xgb_data["X_val"]     # flat loader har kun train/val – brug val som "test"
    y_test = xgb_data["y_val"]

    scaler_y = None
    if "scaler_y" in xgb_data:     # hvis du gemmer scaler i dict'en
        scaler_y = xgb_data["scaler_y"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------- model -------------------------------------------------
    in_dim  = X_test.shape[1]                         # 12
    out_dim = y_test.shape[1]                         # 3
    model   = RBFNet(
        input_dim          = in_dim,
        num_hidden_neurons = hyperparameters["num_hidden_neurons"],
        output_dim         = out_dim,
        betas              = hyperparameters.get("beta_init", 0.5),
        initial_centroids  = None
    ).to(device)

    ckpt = os.path.join(project_root, "checkpoints", model_name)
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    # ------------- inference --------------------------------------------
    with torch.no_grad():
        preds = model(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy()

    # ------------- MSE ---------------------------------------------------
    if scaler_y is not None:                      
        inv_pred = scaler_y.inverse_transform(preds)
        inv_true = scaler_y.inverse_transform(y_test)
    else:
        inv_pred, inv_true = preds, y_test

    mse = mean_squared_error(inv_true, inv_pred)
    print(f"[RBF-static] Test MSE = {mse:.4f}")

    if scaler_y is not None:                  
        plot_results_rbf(
            y_true=y_test,                      
            y_pred=preds,
            scaler_y=scaler_y,
            project_root=project_root,
            model_name=model_name,
            model_type="RBF-Static"            
        )
    else:                                       
        plt.figure(figsize=(12, 6))
        for j in range(inv_true.shape[1]):
            plt.plot(inv_true[:1600, j], linestyle="dotted",
                     label=f"GT Mz{j+1}")
            plt.plot(inv_pred[:1600, j],
                     label=f"Pred Mz{j+1}")
        plt.title("RBF-Static · Predictions vs Ground Truth")
        plt.xlabel("Sample");  plt.ylabel("Torque")
        plt.legend();  plt.grid()
        plots_dir = os.path.join(project_root, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        fname = f"rbf_static_plot_{datetime.now():%Y-%m-%d_%H-%M-%S}.png"
        plt.savefig(os.path.join(plots_dir, fname), dpi=300)
        print(f"[INFO] RBF-static plot saved at plots/{fname}")



    # ------------- log ---------------------------------------------------
    logs_dir = os.path.join(project_root, "logs", "test_logs")
    os.makedirs(logs_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(logs_dir, f"rbfpytorch_static_logs_{ts}.csv")

    pd.DataFrame({
        "Metric": ["MSE"] + list(hyperparameters.keys()) + list(batch_params.keys()),
        "Value":  [mse]   + list(hyperparameters.values()) + list(batch_params.values())
    }).to_csv(log_path, index=False)
    print(f"[INFO] Test MSE og hyperparametre gemt → {log_path}")

    return mse

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
    plot_filename = f"{model_type.lower()}_plot_mse_{mse:.2f}_{current_datetime}.png"
    plot_path = os.path.join(plots_dir, plot_filename)
    
    plt.savefig(plot_path, dpi=300)
    print(f"[INFO] {model_type} plot saved at {plot_path}")

def plot_cnn_results(y_true, y_pred, scaler_y):
    """
    Plotter CNN forudsigelser vs. den sande værdi.
    Antager at y_true og y_pred er skalerede, og bruger scaler_y til at genskabe de oprindelige værdier.
    """
    # Inverse transform for at genskabe de oprindelige værdier
    y_true_orig = scaler_y.inverse_transform(y_true)
    y_pred_orig = scaler_y.inverse_transform(y_pred)
    
    # Vælg et passende antal samples til plottet (f.eks. de første 1600 samples)
    sample_size = min(1600, len(y_true_orig))
    time_labels = np.linspace(0, 10, num=sample_size)
    
    # Udvælg samples
    true_sample = y_true_orig[:sample_size]
    pred_sample = y_pred_orig[:sample_size]
    
    plt.figure(figsize=(12, 6))
    # Antag, at der er flere outputs (f.eks. tre torque-værdier)
    for j in range(true_sample.shape[1]):
        plt.plot(time_labels, true_sample[:, j],
                 label=f"CNN Ground Truth Mz{j+1}", linestyle="dotted")
        plt.plot(time_labels, pred_sample[:, j],
                 label=f"CNN Prediction Mz{j+1}")
    
    plt.xlabel("Time (seconds)")
    plt.ylabel("Torque Values")
    plt.title("CNN Predictions vs Ground Truth (First 10 Seconds)")
    plt.legend()
    plt.grid()
    
    # Gem plottet
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plots_dir = os.path.join(project_root, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plot_filename = f"cnn_plot_{current_datetime}.png"
    plot_path = os.path.join(plots_dir, plot_filename)
    plt.savefig(plot_path, dpi=300)
    plt.show()  # Vis plottet interaktivt
    print(f"[INFO] CNN plot saved at {plot_path}")


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


if __name__ == "__main__":
    SELECTED_GAP = 10
   
    run_params   = batch_parameters | {"gap": SELECTED_GAP}
    
    # Set model names
    transformer_model_name = "transformer_latest.pth"
    xgboost_model_name = "xgboost_latest.json"
    ffnn_model_name = "ffnn_latest.pth"
    one_layer_nn_model_name = "one_layer_nn_latest.pth"
    tcn_model_name = "tcn_latest.pth"
    cnn_lstm_model_name = "cnn-_lstm_latest.pth"
    lstm_model_name = "lstm_latest.pth"

    cnn_model_name = f"cnn_gap{SELECTED_GAP}.pth"
    rbf_model_name = f"rbfpytorch_gap{SELECTED_GAP}.pth"

    # DO THIS FOR EVERY MODEL YOU WANT TO EVALUATE
    
    evaluate_transformer_flag = False
    evaluate_xgboost_flag = False
    evaluate_ffnn_flag = False
    evaluate_one_layer_nn_flag = True
    evaluate_tcn_flag = False
    evaluate_cnnlstm_flag = False
    evaluate_lstm_flag = False
    evaluate_cnn_flag = False
    evaluate_rbf_pytorch_flag = False
    evaluate_rbf_pytorch_static_flag = False  

    if evaluate_transformer_flag:
        evaluate_transformer(batch_parameters, hyperparameters, transformer_model_name)

    if evaluate_xgboost_flag:
        evaluate_xgboost(batch_parameters, hyperparameters, xgboost_model_name)
        
    if evaluate_ffnn_flag:
        evaluate_ffnn(batch_parameters, hyperparameters, ffnn_model_name)
    
    if evaluate_one_layer_nn_flag:
        evaluate_one_layer_nn(batch_parameters, one_layer_nn_model_name)

    if evaluate_tcn_flag:
        evaluate_tcn(batch_parameters, hyperparameters, tcn_model_name)

    if evaluate_cnnlstm_flag:
        evaluate_cnnlstm(batch_parameters, hyperparameters, cnn_lstm_model_name)

    if evaluate_lstm_flag:
        evaluate_lstm(batch_parameters, hyperparameters, lstm_model_name)

    if evaluate_cnn_flag:
        evaluate_cnn(run_params, hyperparameters, cnn_model_name)

    if evaluate_rbf_pytorch_flag:
        evaluate_rbf_pytorch(run_params,
                             hyperparameters,
                             rbf_model_name)
    
    if evaluate_rbf_pytorch_static_flag:
        evaluate_rbf_pytorch_static(batch_parameters,   
                                    hyperparameters,
                                    model_name="rbfpytorch_static.pth")