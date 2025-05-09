import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from datetime import datetime
from sklearn.metrics import mean_squared_error
from torch.utils.data import TensorDataset, DataLoader
import sys
from features import prepare_dataloaders
from hyperparameters import batch_parameters, hyperparameters
from features import prepare_flat_dataloaders 

# Get the absolute path of the project's root directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))  # Move up one level

# Add project root to sys.path
sys.path.append(project_root)


from models.CNN import CNNModel
from models.RBF_PyTorch import RBFNet

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

def plot_results(y_true, y_pred, scaler_y, model_name, model_type):
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
    SELECTED_GAP = 10
   
    run_params   = batch_parameters | {"gap": SELECTED_GAP}

    # Set model names
    cnn_model_name = f"cnn_gap{SELECTED_GAP}.pth"
    rbf_model_name = f"rbfpytorch_gap{SELECTED_GAP}.pth"


    #decide which model
    evaluate_cnn_flag = True
    evaluate_rbf_pytorch_flag = False
    evaluate_rbf_pytorch_static_flag = False  

    if evaluate_cnn_flag:
        evaluate_cnn(run_params, hyperparameters, cnn_model_name)

    if evaluate_rbf_pytorch_flag:
        evaluate_rbf_pytorch(run_params,
                             hyperparameters,
                             rbf_model_name)
    
    if evaluate_rbf_pytorch_static_flag:
        evaluate_rbf_pytorch_static(batch_parameters,   # gap er ligegyldig her
                                    hyperparameters,
                                    model_name="rbfpytorch_static.pth")

