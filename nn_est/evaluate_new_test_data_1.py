#!/usr/bin/env python3
import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error

from features_new_test_data import load_data_new_test_data
from features import create_sequences
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

# pick device once
DEVICE = torch.device("mps") if torch.backends.mps.is_available() \
         else torch.device("cuda") if torch.cuda.is_available() \
         else torch.device("cpu")
print(f"[INFO] Using device: {DEVICE}")

def evaluate_per_file(model, ckpt_name, tag,
                      batch_params,
                      scaler_x, scaler_y,
                      all_test_data, test_x, test_y, test_t):
    """
    For each unique CSV in all_test_data['source_file']:
      - window into sequences
      - run model
      - inverse-transform
      - compute & print per-file MSE
      - save one CSV with MSE in its filename
    """
    # load weights
    model_path = os.path.join(project_root, "checkpoints", ckpt_name)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE).eval()

    results_dir = os.path.join(project_root, "results_new_test_data")
    os.makedirs(results_dir, exist_ok=True)

    # get each file’s rows
    sources = all_test_data["source_file"].values
    unique_files = np.unique(sources)

    for src in unique_files:
        mask = (sources == src)
        X_raw = test_x[mask]
        Y_raw = test_y[mask]
        t_raw = test_t[mask]  # shape (n_rows, 1)

        # window into sequences
        X_seq, Y_seq = create_sequences(
            X_raw, Y_raw,
            batch_params["gap"],
            batch_params["total_len"]
        )
        dummy = np.zeros(len(t_raw))
        T_seq, _ = create_sequences(
            t_raw.reshape(-1,1), dummy,
            batch_params["gap"],
            batch_params["total_len"]
        )

        # DataLoader for this file
        ds = TensorDataset(
            torch.tensor(X_seq, dtype=torch.float32),
            torch.tensor(Y_seq, dtype=torch.float32),
            torch.tensor(T_seq, dtype=torch.float32)
        )
        loader = DataLoader(ds,
                            batch_size=batch_params["batch_size"],
                            shuffle=False)

        # inference
        all_preds, all_trues, all_t = [], [], []
        with torch.no_grad():
            for xb, yb, tb in loader:
                preds = model(xb.to(DEVICE)).cpu().numpy()
                all_preds.append(preds)
                all_trues.append(yb.numpy())
                all_t.append(tb.numpy())

        # concat
        y_pred = np.concatenate(all_preds, axis=0)
        y_true = np.concatenate(all_trues, axis=0)
        t_seq  = np.concatenate(all_t,    axis=0)[:, 0, 0]

        # inverse‐scale
        inv_pred = scaler_y.inverse_transform(y_pred)
        inv_true = scaler_y.inverse_transform(y_true)

        # compute & print MSE
        mse = mean_squared_error(inv_true, inv_pred)
        print(f"[{tag}] {src}  MSE = {mse:.4f}")

        # build results DataFrame
        results = pd.DataFrame({
            "Time":           t_seq,
            "Actual_Mz1":     inv_true[:,0],
            "Predicted_Mz1":  inv_pred[:,0],
            "Actual_Mz2":     inv_true[:,1],
            "Predicted_Mz2":  inv_pred[:,1],
            "Actual_Mz3":     inv_true[:,2],
            "Predicted_Mz3":  inv_pred[:,2],
            "File":           src
        })

        # save with MSE in filename
        base = os.path.splitext(src)[0]
        mse_str = f"{mse:.4f}"
        out_name = f"{base}_{tag.lower()}_mse_{mse_str}_results.csv"
        out_path = os.path.join(results_dir, out_name)
        results.to_csv(out_path, index=False)
        print(f"[{tag}] Saved → {out_name}")

def main():
    # 1) load & scale *all* data once
    all_test_data, test_x, test_y, test_t, scaler_x, scaler_y = \
        load_data_new_test_data(batch_parameters, max_files=None)

    # 2) instantiate your models
    feat_dim = test_x.shape[1]
    targ_dim = test_y.shape[1]
    seq_len  = batch_parameters["total_len"] // batch_parameters["gap"]

    ffnn        = FFNNModel(       feat_dim, targ_dim, seq_len)
    transformer = TransformerModel(feat_dim, targ_dim, seq_len,
                    d_model=        hyperparameters["d_model"],
                    nhead=           hyperparameters["nhead"],
                    num_layers=      hyperparameters["num_layers"],
                    dim_feedforward= hyperparameters["dim_feedforward"],
                    dropout=          hyperparameters["dropout"],
                    layer_norm_eps=   hyperparameters["layer_norm_eps"]
    )
    one_layer   = OneLayerNN(      feat_dim, targ_dim, seq_len)

    # 3) evaluate per‐file (no plotting)
    evaluate_per_file(ffnn,        "ffnn_1.pth",        "FFNN",
                      batch_parameters,
                      scaler_x, scaler_y,
                      all_test_data, test_x, test_y, test_t)

    # evaluate_per_file(transformer, "transformer_2.pth", "Transformer",
    #                   batch_parameters,
    #                   scaler_x, scaler_y,
    #                   all_test_data, test_x, test_y, test_t)

    # evaluate_per_file(one_layer,   "one_layer_nn_2.pth","OneLayer",
    #                   batch_parameters,
    #                   scaler_x, scaler_y,
    #                   all_test_data, test_x, test_y, test_t)

if __name__ == "__main__":
    main() 