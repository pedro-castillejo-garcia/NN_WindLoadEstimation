import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from hyperparameters import batch_parameters
from features import create_sequences

# Automatically find the absolute path of NN_WindLoadEstimation
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def load_data_new_test_data(batch_parameters, max_files=None):
    """
    Loader CSV-filer, fitter MinMaxScalers og transformerer til float32.
    """
    csv_folder = os.path.join(project_root, "data/raw/thres")
    file_paths = [os.path.join(csv_folder, f)
                  for f in os.listdir(csv_folder) if f.endswith(".csv")]

    if max_files:
        file_paths = file_paths[:max_files]

    datasets = []
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        df["source_file"] = os.path.basename(file_path)
        datasets.append(df)

    features = ["Mx1", "Mx2", "Mx3", "My1", "My2", "My3",
                "Theta", "Vwx", "beta1", "beta2", "beta3", "omega_r"]
    targets = ["Mz1", "Mz2", "Mz3"]
    time_col = ["t"]

    all_test_data = pd.concat(datasets, ignore_index=True)

    # Fit scalers
    scaler_x = MinMaxScaler().fit(all_test_data[features].values)
    scaler_y = MinMaxScaler().fit(all_test_data[targets].values)

    # Transform og cast til float32
    test_x = scaler_x.transform(all_test_data[features].values).astype(np.float32)
    test_y = scaler_y.transform(all_test_data[targets].values).astype(np.float32)
    test_t = all_test_data[time_col].values.astype(np.float32)

    return all_test_data, test_x, test_y, test_t, scaler_x, scaler_y


def prepare_dataloaders_new_test_data(batch_parameters,
                                      max_files=None,
                                      return_raw=False):
    all_df, X, y, t, sc_x, sc_y = load_data_new_test_data(batch_parameters,
                                                          max_files)
    if return_raw:
        return all_df, X, y, t, sc_x, sc_y

    # ellers returnér som før
    return all_df, X, y, t, sc_x, sc_y


def evaluate_per_file(model, ckpt_name, tag,
                      batch_params,
                      scaler_x, scaler_y,
                      all_test_data, test_x, test_y, test_t):
    """
    Evaluates model per-fil:
    - Windowing
    - Inference
    - Inverse-scaling
    - Compute & save MSE
    """
    # Load weights
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(project_root, "checkpoints", ckpt_name)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE).eval()

    results_dir = os.path.join(project_root, "results_new_test_data")
    os.makedirs(results_dir, exist_ok=True)

    sources = all_test_data["source_file"].values
    unique_files = np.unique(sources)

    for src in unique_files:
        mask = (sources == src)
        X_raw = test_x[mask]
        Y_raw = test_y[mask]
        t_raw = test_t[mask]

        # Window into sequences for this file
        X_seq, Y_seq = create_sequences(
            X_raw, Y_raw,
            batch_params["gap"], batch_params["total_len"]
        )
        dummy = np.zeros(len(t_raw), dtype=np.float32)
        T_seq, _ = create_sequences(
            t_raw.reshape(-1,1), dummy,
            batch_params["gap"], batch_params["total_len"]
        )

        # Build DataLoader
        ds = TensorDataset(
            torch.tensor(X_seq, dtype=torch.float32),
            torch.tensor(Y_seq, dtype=torch.float32),
            torch.tensor(T_seq, dtype=torch.float32)
        )
        loader = DataLoader(ds,
                            batch_size=batch_params["batch_size"],
                            shuffle=False)

        # Inference
        all_preds, all_trues, all_ts = [], [], []
        with torch.no_grad():
            for xb, yb, tb in loader:
                xb = xb.to(DEVICE)
                preds = model(xb).cpu().numpy()
                all_preds.append(preds)
                all_trues.append(yb.numpy())
                all_ts.append(tb.numpy())

        y_pred = np.concatenate(all_preds, axis=0)
        y_true = np.concatenate(all_trues, axis=0)
        t_seq  = np.concatenate(all_ts, axis=0)[:, 0, 0]

        # Inverse-scale
        inv_pred = scaler_y.inverse_transform(y_pred)
        inv_true = scaler_y.inverse_transform(y_true)

        # Compute MSE
        mse = mean_squared_error(inv_true, inv_pred)
        print(f"[{tag}] {src}  MSE = {mse:.4f}")

        # Save CSV
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

        base = os.path.splitext(src)[0]
        out_name = f"{base}_{tag.lower()}_mse_{mse:.4f}_results_thresh.csv"
        save_path = os.path.join(results_dir, out_name)
        results.to_csv(save_path, index=False)
        print(f"[{tag}] Saved → {out_name}")

if __name__ == "__main__":
    test_loader, scaler_x, scaler_y, source_tensor = prepare_dataloaders_new_test_data(batch_parameters) 
    