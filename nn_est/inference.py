import os
import sys
import time
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from features import create_sequences
from hyperparameters import batch_parameters, hyperparameters

# Get project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.append(project_root)

# Import models
from models.Transformer import TransformerModel
from models.FFNN import FFNNModel
from models.OneLayerNN import OneLayerNN
from models.XGBoost import XGBoostModel
from models.CNN import CNNModel
from models.RBF_PyTorch import RBFNet

# Features and targets
features = ["Mx1", "Mx2", "Mx3", "My1", "My2", "My3", "Theta", "Vwx", "beta1", "beta2", "beta3", "omega_r"]
targets = ["Mz1", "Mz2", "Mz3"]

def load_inference_data():
    """Load and scale the inference dataset."""
    data_path = os.path.join(project_root, "data", "raw", "inference_test.csv")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"[ERROR] inference_test.csv not found at {data_path}")
    
    df = pd.read_csv(data_path)

    if "t" not in df.columns:
        raise ValueError("[ERROR] 't' column not found in inference_test.csv")

    X_raw = df[features].values
    y_raw = df[targets].values
    t_values = df["t"].values

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_scaled = scaler_x.fit_transform(X_raw)
    y_scaled = scaler_y.fit_transform(y_raw)

    return X_scaled, y_scaled, t_values, scaler_x, scaler_y

def save_results(results, model_type):
    results_df = pd.DataFrame(results)
    results_dir = os.path.join(project_root, "results_inference")
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(results_dir, f"inference_results_{model_type}_{timestamp}.csv")
    results_df.to_csv(save_path, index=False)
    print(f"[INFO] Saved inference results to {save_path}")

def infer_cnn(X_seq, y_seq, t_seq, model_name, scaler_y, device_type):
    print("[INFO] Running CNN Inference...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2) Genskab din CNN præcis som i træningskoden
    in_channels = X_seq.shape[2]       # antal features
    seq_len     = X_seq.shape[1]       # sekvenslængde
    out_dim     = y_seq.shape[1]       # antal target‑dim

    model = CNNModel(
        in_channels=in_channels,
        seq_length=seq_len,
        num_outputs=out_dim
    )

    # 3) Load vægte
    model_path = os.path.join(project_root, "checkpoints", model_name)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    results = []
    X_tensor = torch.tensor(X_seq, dtype=torch.float32)

    with torch.no_grad():
        for i in range(X_tensor.shape[0]):
            single_X = X_tensor[i:i+1].to(device)
            start = time.time()
            pred = model(single_X)
            end   = time.time()

            inf_ms = (end - start) * 1000
            pred_inv = scaler_y.inverse_transform(pred.cpu().numpy())[0]
            true_inv = scaler_y.inverse_transform(y_seq[i].reshape(1,-1))[0]

            results.append({
                "t": t_seq[i],
                "Actual_Mz1": true_inv[0],  "Predicted_Mz1": pred_inv[0],
                "Actual_Mz2": true_inv[1],  "Predicted_Mz2": pred_inv[1],
                "Actual_Mz3": true_inv[2],  "Predicted_Mz3": pred_inv[2],
                "Inference_Time_ms": inf_ms
            })

    save_results(results, "cnn")

def infer_rbf(X_seq, y_seq, t_seq,
              model_name, scaler_y, device_type="gpu"):
    """
    Inference med den trænte PyTorch-RBFNet.
    X_seq:  (N, L, C)  – samme sekvensstrukturer som til CNN/Transformer
    y_seq:  (N, 3)
    t_seq:  (N,)       – tidspunkter for den **sidste** sample i hvert vindue
    """
    print("[INFO] Running RBF Inference ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    N, L, C   = X_seq.shape
    flat_dim  = L * C                 # input-dimension efter flatten
    out_dim   = y_seq.shape[1]
    n_hidden  = hyperparameters.get("num_hidden_neurons", 100)
    beta_init = hyperparameters.get("beta_init", 0.5)

    # 1) Genskab modellen (centroids indlæses fra checkpoint)
    model = RBFNet(input_dim=flat_dim,
                   num_hidden_neurons=n_hidden,
                   output_dim=out_dim,
                   betas=beta_init,
                   initial_centroids=None)

    ckpt_path = os.path.join(project_root, "checkpoints", model_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"[ERROR] RBF checkpoint ikke fundet: {ckpt_path}")

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device).eval()
    print(f"[INFO] Loaded weights from {ckpt_path}")

    # 2) Forbered data (flatten)
    X_flat   = X_seq.reshape(N, -1)                     # (N, L*C)
    X_tensor = torch.tensor(X_flat, dtype=torch.float32)
    results  = []

    with torch.no_grad():
        for i in range(N):
            single_X = X_tensor[i:i+1].to(device)
            start    = time.time()
            pred     = model(single_X)
            inf_ms   = (time.time() - start) * 1000

            pred_inv = scaler_y.inverse_transform(pred.cpu().numpy())[0]
            true_inv = scaler_y.inverse_transform(y_seq[i].reshape(1, -1))[0]

            results.append({
                "t": t_seq[i],
                "Actual_Mz1": true_inv[0],  "Predicted_Mz1": pred_inv[0],
                "Actual_Mz2": true_inv[1],  "Predicted_Mz2": pred_inv[1],
                "Actual_Mz3": true_inv[2],  "Predicted_Mz3": pred_inv[2],
                "Inference_Time_ms": inf_ms
            })

    save_results(results, "rbf")
    print(f"[INFO] Inference finished – {len(results)} windows gemt.")
    
def infer_rbf_static(X_raw, y_raw, t_vals,
                     model_name, scaler_y,
                     device_type="gpu"):
    """
    •  X_raw : (N, C)   – skalerede features pr. time-step
    •  y_raw : (N, 3)   – skalerede mål
    •  t_vals: (N,)     – tidskolonnen “t”
    """
    print("[INFO] Running RBF-Static Inference …")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    in_dim   = X_raw.shape[1]                     # 12
    out_dim  = y_raw.shape[1]                     # 3
    n_hidden = hyperparameters.get("num_hidden_neurons", 100)
    beta0    = hyperparameters.get("beta_init",   0.5)

    # 1) genskab model-arkitekturen
    model = RBFNet(input_dim          = in_dim,
                   num_hidden_neurons = n_hidden,
                   output_dim         = out_dim,
                   betas              = beta0,
                   initial_centroids  = None)

    ckpt = os.path.join(project_root, "checkpoints", model_name)
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"[ERROR] Checkpoint ikke fundet: {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device).eval()
    print(f"[INFO] Weights loaded from {ckpt}")

    X_tensor = torch.tensor(X_raw, dtype=torch.float32)
    results  = []

    with torch.no_grad():
        for i in range(X_tensor.shape[0]):
            xb   = X_tensor[i:i+1].to(device)
            tic  = time.time()
            pred = model(xb)
            toc  = time.time()

            pred_inv = scaler_y.inverse_transform(pred.cpu().numpy())[0]
            true_inv = scaler_y.inverse_transform(y_raw[i].reshape(1, -1))[0]

            results.append({
                "t": t_vals[i],
                "Actual_Mz1": true_inv[0],  "Predicted_Mz1": pred_inv[0],
                "Actual_Mz2": true_inv[1],  "Predicted_Mz2": pred_inv[1],
                "Actual_Mz3": true_inv[2],  "Predicted_Mz3": pred_inv[2],
                "Inference_Time_ms": (toc - tic) * 1000
            })

    save_results(results, "rbf_static")
    print(f"[INFO] Static-RBF inference complete – {len(results)} rækker gemt.")
    


    device_type = "gpu"  

    # Flags
    infer_cnn_flag = False
    infer_rbf_flag = False     
    infer_rbf_static_flag = True      

    cnn_model_name = "cnn_gap10.pth"
    rbf_model_name = "rbfpytorch_gap10.pth"

    # Load and prepare data
    X_scaled, y_scaled, t_values, scaler_x, scaler_y = load_inference_data()
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, batch_parameters['gap'], batch_parameters['total_len'])
    t_seq = t_values[batch_parameters['total_len'] - 1:]

    # Run inference
    if infer_cnn_flag:
        infer_cnn(X_seq, y_seq, t_seq, cnn_model_name, scaler_y, device_type)

    if infer_rbf_flag:
        infer_rbf(X_seq, y_seq, t_seq,
                  rbf_model_name, scaler_y, device_type)
        
    if infer_rbf_static_flag:
        infer_rbf_static(
            X_scaled, y_scaled, t_values,
            model_name="rbfpytorch_static.pth",   
            scaler_y=scaler_y,
            device_type=device_type
        )

def infer_transformer(X_seq, y_seq, t_seq, model_name, scaler_y, device_type):
    print("[INFO] Running Transformer Inference...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TransformerModel(
        input_dim=len(features),
        output_dim=3,
        seq_len=batch_parameters['total_len'] // batch_parameters['gap'],
        d_model=hyperparameters['d_model'],
        nhead=hyperparameters['nhead'],
        num_layers=hyperparameters['num_layers'],
        dim_feedforward=hyperparameters['dim_feedforward'],
        dropout=hyperparameters['dropout'],
        layer_norm_eps=hyperparameters['layer_norm_eps']
    )

    model_path = os.path.join(project_root, "checkpoints", model_name)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    results = []
    X_seq_tensor = torch.tensor(X_seq, dtype=torch.float32)

    with torch.no_grad():
        for i in range(X_seq_tensor.shape[0]):
            single_X = X_seq_tensor[i:i+1].to(device)
            start_time = time.time()
            pred = model(single_X)
            end_time = time.time()

            inference_duration = (end_time - start_time) * 1000  # ms
            pred_inv = scaler_y.inverse_transform(pred.cpu().numpy())[0]
            true_inv = scaler_y.inverse_transform(y_seq[i].reshape(1, -1))[0]

            results.append({
                "t": t_seq[i],
                "Actual_Mz1": true_inv[0],
                "Predicted_Mz1": pred_inv[0],
                "Actual_Mz2": true_inv[1],
                "Predicted_Mz2": pred_inv[1],
                "Actual_Mz3": true_inv[2],
                "Predicted_Mz3": pred_inv[2],
                "Inference_Time_ms": inference_duration
            })

    save_results(results, "transformer")

def infer_ffnn(X_seq, y_seq, t_seq, model_name, scaler_y, device_type):
    print("[INFO] Running FFNN Inference...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X_seq.shape[2]

    model = FFNNModel(
        input_dim=input_dim,
        output_dim=3,
        seq_len=batch_parameters['total_len'] // batch_parameters['gap'],
        dropout=hyperparameters['dropout']
    )

    model_path = os.path.join(project_root, "checkpoints", model_name)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    results = []
    X_flat = X_seq.reshape(X_seq.shape[0], -1)
    X_tensor = torch.tensor(X_flat, dtype=torch.float32)

    with torch.no_grad():
        for i in range(X_tensor.shape[0]):
            single_X = X_tensor[i:i+1].to(device)
            start_time = time.time()
            pred = model(single_X)
            end_time = time.time()

            inference_duration = (end_time - start_time) * 1000
            pred_inv = scaler_y.inverse_transform(pred.cpu().numpy())[0]
            true_inv = scaler_y.inverse_transform(y_seq[i].reshape(1, -1))[0]

            results.append({
                "t": t_seq[i],
                "Actual_Mz1": true_inv[0],
                "Predicted_Mz1": pred_inv[0],
                "Actual_Mz2": true_inv[1],
                "Predicted_Mz2": pred_inv[1],
                "Actual_Mz3": true_inv[2],
                "Predicted_Mz3": pred_inv[2],
                "Inference_Time_ms": inference_duration
            })

    save_results(results, "ffnn")

def infer_one_layer_nn(X_seq, y_seq, t_seq, model_name, scaler_y, device_type):
    print("[INFO] Running OneLayerNN Inference...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = OneLayerNN(
        input_dim=len(features),
        output_dim=3,
        seq_len=batch_parameters['total_len'] // batch_parameters['gap']
    )

    model_path = os.path.join(project_root, "checkpoints", model_name)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    results = []
    X_seq_tensor = torch.tensor(X_seq, dtype=torch.float32)

    with torch.no_grad():
        for i in range(X_seq_tensor.shape[0]):
            single_X = X_seq_tensor[i:i+1].to(device)
            start_time = time.time()
            pred = model(single_X)
            end_time = time.time()

            inference_duration = (end_time - start_time) * 1000
            pred_inv = scaler_y.inverse_transform(pred.cpu().numpy())[0]
            true_inv = scaler_y.inverse_transform(y_seq[i].reshape(1, -1))[0]

            results.append({
                "t": t_seq[i],
                "Actual_Mz1": true_inv[0],
                "Predicted_Mz1": pred_inv[0],
                "Actual_Mz2": true_inv[1],
                "Predicted_Mz2": pred_inv[1],
                "Actual_Mz3": true_inv[2],
                "Predicted_Mz3": pred_inv[2],
                "Inference_Time_ms": inference_duration
            })

    save_results(results, "one_layer_nn")

def infer_xgboost(X_seq, y_seq, t_seq, model_name, scaler_y, device_type):
    print("[INFO] Running XGBoost Inference...")

    model = XGBoostModel(
        n_estimators=hyperparameters["n_estimators"],
        max_depth=hyperparameters["max_depth"],
        learning_rate=hyperparameters["learning_rate"]
    )

    model_path = os.path.join(project_root, "checkpoints", model_name)
    model.load_model(model_path)

    X_flat = X_seq.reshape(X_seq.shape[0], -1)

    results = []
    for i in range(X_flat.shape[0]):
        single_X = X_flat[i:i+1]
        start_time = time.time()
        pred = model.predict(single_X)
        end_time = time.time()

        inference_duration = (end_time - start_time) * 1000
        pred_inv = scaler_y.inverse_transform(pred.reshape(1, -1))[0]
        true_inv = scaler_y.inverse_transform(y_seq[i].reshape(1, -1))[0]

        results.append({
            "t": t_seq[i],
            "Actual_Mz1": true_inv[0],
            "Predicted_Mz1": pred_inv[0],
            "Actual_Mz2": true_inv[1],
            "Predicted_Mz2": pred_inv[1],
            "Actual_Mz3": true_inv[2],
            "Predicted_Mz3": pred_inv[2],
            "Inference_Time_ms": inference_duration
        })

    save_results(results, "xgboost")

if __name__ == "__main__":
    device_type = "gpu"  # Options: "cuda" or "cpu"

    # Flags
    infer_transformer_flag = False
    infer_ffnn_flag = False
    infer_one_layer_nn_flag = False
    infer_xgboost_flag = True
    infer_cnn_flag = False
    infer_rbf_flag = False     
    infer_rbf_static_flag = False      

    cnn_model_name = "cnn_gap10.pth"
    rbf_model_name = "rbfpytorch_gap10.pth"
    transformer_model_name = "transformer_latest.pth"
    ffnn_model_name = "ffnn_sequenced_2025-04-28_19-57-03_latest.pth"
    one_layer_nn_model_name = "one_layer_nn_latest.pth"
    xgboost_model_name = "xgboost_latest.json"

    # Load and prepare data
    X_scaled, y_scaled, t_values, scaler_x, scaler_y = load_inference_data()
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, batch_parameters['gap'], batch_parameters['total_len'])
    t_seq = t_values[batch_parameters['total_len'] - 1:]

    # Run inference
    if infer_transformer_flag:
        infer_transformer(X_seq, y_seq, t_seq, transformer_model_name, scaler_y, device_type)

    if infer_ffnn_flag:
        infer_ffnn(X_seq, y_seq, t_seq, ffnn_model_name, scaler_y, device_type)

    if infer_one_layer_nn_flag:
        infer_one_layer_nn(X_seq, y_seq, t_seq, one_layer_nn_model_name, scaler_y, device_type)

    if infer_xgboost_flag:
        infer_xgboost(X_seq, y_seq, t_seq, xgboost_model_name, scaler_y, device_type)

    if infer_cnn_flag:
        infer_cnn(X_seq, y_seq, t_seq, cnn_model_name, scaler_y, device_type)

    if infer_rbf_flag:
        infer_rbf(X_seq, y_seq, t_seq,
                  rbf_model_name, scaler_y, device_type)
        
    if infer_rbf_static_flag:
        infer_rbf_static(
            X_scaled, y_scaled, t_values,
            model_name="rbfpytorch_static.pth",   
            scaler_y=scaler_y,
            device_type=device_type
        )
