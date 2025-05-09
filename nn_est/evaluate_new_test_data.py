import os
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_squared_error
from torch.utils.data import TensorDataset, DataLoader
from features import create_sequences


from features_new_test_data import prepare_dataloaders_new_test_data
from hyperparameters import batch_parameters, hyperparameters

# Get the absolute path of the project's root directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))  # Move up one level

# Add project root to sys.path
sys.path.append(project_root)

# Add the models you want to evaluate with the new data here

from models.CNN import CNNModel
from models.RBF_PyTorch import RBFNet, initialize_centroids




def evaluate_cnn_new_test_data(batch_params, hyperparams, model_name, max_files=None):
    print("[INFO] Evaluating CNN model on new test data...")

    # 1) Load rå data + scalers
    all_test_data, test_x, test_y, test_t, scaler_x, scaler_y = (
        prepare_dataloaders_new_test_data(batch_params, max_files)
    )

    # 2) Vælg device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # 3) Parametre til model
    in_channels = test_x.shape[1]                                # 12 features
    seq_len     = batch_params["total_len"] // batch_params["gap"]
    out_dim     = test_y.shape[1]                                # 3 targets

    # 4) Byg modellen
    cnn_model = CNNModel(
        in_channels=in_channels,
        seq_length=seq_len,
        num_outputs=out_dim
    )

    # 5) Load checkpoint
    chkpt = os.path.join(os.path.dirname(__file__), "..", "checkpoints", model_name)
    if not os.path.exists(chkpt):
        raise FileNotFoundError(f"[ERROR] Model file {model_name} not found at {chkpt}")
    cnn_model.load_state_dict(torch.load(chkpt, map_location=device))
    cnn_model.to(device).eval()
    print("[INFO] Model loaded successfully.")

    # 6) Per‑fil inference + gem resultater
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results_new_test_data")
    os.makedirs(results_dir, exist_ok=True)

    per_file_mse = {}
    unique_files = np.unique(all_test_data["source_file"].values)

    for src in unique_files:
        mask   = all_test_data["source_file"].values == src
        X_raw  = test_x[mask]
        Y_raw  = test_y[mask]
        T_raw  = test_t[mask]

        # 6a) Window into sequences
        X_seq, Y_seq = create_sequences(
            X_raw, Y_raw,
            batch_params["gap"], batch_params["total_len"]
        )
        dummy = np.zeros(len(T_raw), dtype=np.float32)
        T_seq, _ = create_sequences(
            T_raw.reshape(-1,1), dummy,
            batch_params["gap"], batch_params["total_len"]
        )

        # 6b) DataLoader til denne fil
        ds     = TensorDataset(torch.tensor(X_seq),
                               torch.tensor(Y_seq),
                               torch.tensor(T_seq))
        loader = DataLoader(ds, batch_size=batch_params["batch_size"], shuffle=False)

        # 6c) Inference-loop
        all_preds, all_trues = [], []
        with torch.no_grad():
            for xb, yb, _ in loader:
                pred = cnn_model(xb.to(device)).cpu().numpy()
                all_preds.append(pred)
                all_trues.append(yb.numpy())

        # 6d) Saml og inverse‑scale
        y_pred = np.concatenate(all_preds, axis=0)
        y_true = np.concatenate(all_trues, axis=0)
        inv_pred = scaler_y.inverse_transform(y_pred)
        inv_true = scaler_y.inverse_transform(y_true)

        mse = mean_squared_error(inv_true, inv_pred)
        per_file_mse[src] = mse
        print(f"[CNN] {src}  MSE = {mse:.4f}")

        # 6e) Gem per‑fil CSV
        df = pd.DataFrame({
            "Time":           T_seq[:,0,0],
            "Actual_Mz1":     inv_true[:,0],
            "Predicted_Mz1":  inv_pred[:,0],
            "Actual_Mz2":     inv_true[:,1],
            "Predicted_Mz2":  inv_pred[:,1],
            "Actual_Mz3":     inv_true[:,2],
            "Predicted_Mz3":  inv_pred[:,2],
            "File":           src
        })
        base    = os.path.splitext(src)[0]
        outname = f"{base}_cnn_gap{batch_params['gap']}_mse_{mse:.4f}_results.csv"
        df.to_csv(os.path.join(results_dir, outname), index=False)
        print(f"[CNN] Saved → {outname}")

    # 7) Gem samlet MSE‑log
    logs_dir = os.path.join(os.path.dirname(__file__), "..", "logs", "test_new_data_logs")
    os.makedirs(logs_dir, exist_ok=True)

    log_df = pd.DataFrame(
        list(per_file_mse.items()),
        columns=["File", "MSE"]
    )
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(logs_dir, f"cnn_test_new_data_logs_{timestamp}.csv")
    log_df.to_csv(log_path, index=False)
    print(f"[CNN] Saved MSE‑log → {log_path}")

def evaluate_rbf_new_test_data(batch_params, hyperparams, model_name, max_files=None):
    print("[INFO] Evaluating RBFNet on new test data...")

    # 1) Load rå data + scalers
    all_test_data, test_x, test_y, test_t, scaler_x, scaler_y = (
        prepare_dataloaders_new_test_data(batch_params, max_files)
    )

    # 2) Vælg device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # 3) Initialiser model ‑ centroids og betas hentes fra hyperparameters
    #    Her antager vi xgb_data‐dict ikke, så vi kører KMeans centrering på hele test‐X
    X_all = torch.from_numpy(test_x.reshape(-1, test_x.shape[-1])).to(device)
    centroids = initialize_centroids(X_all, hyperparams["num_hidden_neurons"]).to(device)
    # Inden du loader checkpoint:
   # 3) Parametre
    n_features = test_x.shape[-1]                            # fx 12
    seq_len    = batch_params["total_len"] // batch_params["gap"]
    flat_dim   = seq_len * n_features                        # fx 20 * 12 = 240

    # 4) Opbyg model UTEN initial_centroids
    rbf_model = RBFNet(
        input_dim          = flat_dim,
        num_hidden_neurons = hyperparams["num_hidden_neurons"],
        output_dim         = test_y.shape[-1],
        betas              = hyperparams.get("beta_init", 0.5),
        initial_centroids  = None    # drop — load_state_dict sætter centroids
    ).to(device)

    # 4) Load trænings‑checkpoint
    chkpt = os.path.join(os.path.dirname(__file__), "..", "checkpoints", model_name)
    if not os.path.exists(chkpt):
        raise FileNotFoundError(f"[ERROR] RBF model file {model_name} not found at {chkpt}")
    rbf_model.load_state_dict(torch.load(chkpt, map_location=device))
    rbf_model.eval()
    print("[INFO] RBF model loaded successfully.")

    # 5) For hver fil: window, infer, inverse‐scale, save CSV + samle MSE log
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results_new_test_data")
    os.makedirs(results_dir, exist_ok=True)

    per_file_mse = {}
    unique_files = np.unique(all_test_data["source_file"].values)

    for src in unique_files:
        mask   = all_test_data["source_file"].values == src
        X_raw  = test_x[mask]
        Y_raw  = test_y[mask]
        T_raw  = test_t[mask]

        X_seq, Y_seq = create_sequences(
            X_raw, Y_raw,
            batch_params["gap"], batch_params["total_len"]
        )
        dummy = np.zeros(len(T_raw), dtype=np.float32)
        T_seq, _ = create_sequences(
            T_raw.reshape(-1,1), dummy,
            batch_params["gap"], batch_params["total_len"]
        )

        ds     = TensorDataset(torch.tensor(X_seq), torch.tensor(Y_seq), torch.tensor(T_seq))
        loader = DataLoader(ds, batch_size=batch_params["batch_size"], shuffle=False)

        all_preds, all_trues = [], []
        with torch.no_grad():
            for xb, yb, _ in loader:
                xb = xb.to(device)
                preds = rbf_model(xb.reshape(xb.size(0), -1)).cpu().numpy()
                all_preds.append(preds)
                all_trues.append(yb.numpy())

        y_pred = np.concatenate(all_preds, axis=0)
        y_true = np.concatenate(all_trues, axis=0)
        inv_pred = scaler_y.inverse_transform(y_pred)
        inv_true = scaler_y.inverse_transform(y_true)

        mse = mean_squared_error(inv_true, inv_pred)
        per_file_mse[src] = mse
        print(f"[RBF] {src}  MSE = {mse:.4f}")

        df = pd.DataFrame({
            "Time":           T_seq[:,0,0],
            "Actual_Mz1":     inv_true[:,0],
            "Predicted_Mz1":  inv_pred[:,0],
            "Actual_Mz2":     inv_true[:,1],
            "Predicted_Mz2":  inv_pred[:,1],
            "Actual_Mz3":     inv_true[:,2],
            "Predicted_Mz3":  inv_pred[:,2],
            "File":           src
        })
        base    = os.path.splitext(src)[0]
        outname = f"{base}_rbf_gap{batch_params['gap']}_mse_{mse:.4f}_results.csv"
        df.to_csv(os.path.join(results_dir, outname), index=False)
        print(f"[RBF] Saved → {outname}")

    # 6) Samlet MSE‑log
    logs_dir = os.path.join(os.path.dirname(__file__), "..", "logs", "test_new_data_logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_df = pd.DataFrame(list(per_file_mse.items()), columns=["File", "MSE"])
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(logs_dir, f"rbf_test_new_data_logs_{timestamp}.csv")
    log_df.to_csv(log_path, index=False)
    print(f"[RBF] Saved MSE‑log → {log_path}")

def evaluate_rbf_static_new_test_data(batch_params,
                                      hyperparams,
                                      model_name,
                                      max_files=None):
    """
    • Læser de nye test-csv-filer (samme helper som før).
    • Én prøve = ét tids-step  →  ingen create_sequences(…)
    • Gemmer et CSV + MSE-log pr. fil.
    """
    print("[INFO] Evaluating STATIC RBF model on new test data …")

    # ---------- 1) Rå data + scalers ----------
    (all_test_df,
     test_x, test_y, test_t,
     scaler_x, scaler_y) = prepare_dataloaders_new_test_data(
        batch_params, max_files=max_files, return_raw=True  # <- sørg for at return_raw=True
    )

    # ---------- 2) Device ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # ---------- 3) Model ----------
    n_feat = test_x.shape[1]          # 12
    n_out  = test_y.shape[1]          # 3

    rbf = RBFNet(
        input_dim          = n_feat,
        num_hidden_neurons = hyperparams["num_hidden_neurons"],
        output_dim         = n_out,
        betas              = hyperparams.get("beta_init", 0.5),
        initial_centroids  = None            # centroids kommer fra checkpoint
    ).to(device)

    ckpt_path = os.path.join(project_root, "checkpoints", model_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"[ERROR] {ckpt_path} findes ikke")
    rbf.load_state_dict(torch.load(ckpt_path, map_location=device))
    rbf.eval()
    print("[INFO] RBF checkpoint indlæst.")

    # ---------- 4) Result-mapper ----------
    results_dir = os.path.join(project_root, "results_new_test_data")
    os.makedirs(results_dir, exist_ok=True)

    per_file_mse = {}
    for src in np.unique(all_test_df["source_file"]):
        mask      = all_test_df["source_file"].values == src
        X_file    = torch.tensor(test_x[mask], dtype=torch.float32).to(device)
        y_file    = test_y[mask]          # numpy
        t_file    = test_t[mask]          # numpy

        with torch.no_grad():
            y_pred = rbf(X_file).cpu().numpy()

        inv_pred = scaler_y.inverse_transform(y_pred)
        inv_true = scaler_y.inverse_transform(y_file)

        mse = mean_squared_error(inv_true, inv_pred)
        per_file_mse[src] = mse
        print(f"[RBF-static] {src}  MSE={mse:.4f}")

        t_file = test_t[mask].ravel()   # <- tilføj .ravel()

        # ---------- gem CSV ----------
        out_df = pd.DataFrame({
            "Time"         : t_file,              # allerede 1-D
            "Actual_Mz1"   : inv_true[:, 0],
            "Pred_Mz1"     : inv_pred[:, 0],
            "Actual_Mz2"   : inv_true[:, 1],
            "Pred_Mz2"     : inv_pred[:, 1],
            "Actual_Mz3"   : inv_true[:, 2],
            "Pred_Mz3"     : inv_pred[:, 2]
        })
        base = os.path.splitext(src)[0]
        csv_name = f"{base}_rbf_static_results_mse_{mse:.4f}.csv"
        out_df.to_csv(os.path.join(results_dir, csv_name), index=False)
        print(f"[RBF-static]   → gemt {csv_name}")

    # ---------- 5) samlet log ----------
    logs_dir = os.path.join(project_root, "logs", "test_new_data_logs")
    os.makedirs(logs_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    pd.DataFrame(list(per_file_mse.items()), columns=["File", "MSE"]) \
        .to_csv(os.path.join(logs_dir,
                 f"rbf_static_test_logs_{ts}.csv"),
                index=False)
    print(f"[RBF-static] MSE-log gemt.")
    
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
    
SELECTED_GAP = 10         

if __name__ == "__main__":

    run_params   = batch_parameters | {"gap": SELECTED_GAP}

    # Set model names
    cnn_lstm_model_name = "cnn-_lstm_latest.pth"
    lstm_model_name = "lstm_latest.pth"

    # DO THIS FOR EVERY MODEL YOU WANT TO EVALUATE    
    evaluate_ffnn_flag = False
    evaluate_transformer_flag = False
    evaluate_tcn_flag = False
    evaluate_cnnlstm_flag = False
    evaluate_lstm_flag = False
    evaluate_cnn_flag = True             
    evaluate_rbf_flag = False
    evaluate_rbf_static_flag = False

    if evaluate_cnn_flag:
        model_name = f"cnn_gap{SELECTED_GAP}.pth"
        evaluate_cnn_new_test_data(
            run_params,
            hyperparameters,
            model_name,
            max_files=None
        )
    if evaluate_rbf_flag:
        model_name = f"rbfpytorch_gap{SELECTED_GAP}.pth"
        evaluate_rbf_new_test_data(run_params, hyperparameters, model_name, max_files=None)

    if evaluate_rbf_static_flag:
        model_name = f"rbfpytorch_static.pth"          
        evaluate_rbf_static_new_test_data(
            run_params,
            hyperparameters,
            model_name,
            max_files=None        
        )

    
    """
    if evaluate_ffnn_flag:
        evaluate_ffnn_new_test_data(batch_parameters, hyperparameters, ffnn_model_name, max_files=2)
        
    if evaluate_transformer_flag:
        evaluate_transformer_new_test_data(batch_parameters, hyperparameters, transformer_model_name, max_files=25)
    
    if evaluate_tcn_flag:
        evaluate_tcn_new_test_data(batch_parameters, hyperparameters, tcn_model_name)

    if evaluate_cnnlstm_flag:
        evaluate_cnnlstm_new_test_data(batch_parameters, hyperparameters, cnn_lstm_model_name)

    if evaluate_lstm_flag:
        evaluate_lstm_new_test_data(batch_parameters, hyperparameters, lstm_model_name)
    """
            
