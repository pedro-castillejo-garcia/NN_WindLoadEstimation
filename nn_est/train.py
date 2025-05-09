import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
#from features import prepare_flat_dataloaders   # <-- ny import
import tensorflow as tf
import keras as keras


from datetime import datetime
from features import load_data
from features import prepare_dataloaders
from features import prepare_flat_dataloaders
from hyperparameters import batch_parameters, hyperparameters

# Get the absolute path of the project's root directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))  # Move up one level

# Add project root to sys.path
sys.path.append(project_root)

#CNN
from models.CNN import CNNModel
#Transformer
from models.Transformer import TransformerModel
#from models.XGBoost import XGBoostModel
from models.RadialBasisFunctionModel import RBFN_model
# Define EarlyStopping class
from models.RBF_PyTorch import initialize_centroids, RBFNet


class EarlyStopping:
    """
    Stopper træningen, hvis valideringstabet ikke er forbedret
    `patience` gange i træk med mindst `delta`.
    """
    def __init__(self, patience=5, delta=5e-5):
        self.patience   = patience
        self.delta      = delta
        self.best_loss  = float("inf")
        self.counter    = 0
        self.early_stop = False

    def __call__(self, val_loss: float):
        # bedre end hidtil?
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                
def train_cnn(train_loader, val_loader, hyperparams, model_name, gap):
    """Træn CNN og gem den som <model_name> (gap bruges kun til printk)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    in_channels = train_loader.dataset.tensors[0].shape[-1]
    seq_len     = train_loader.dataset.tensors[0].shape[1]
    out_dim     = train_loader.dataset.tensors[1].shape[-1]

    model = CNNModel(in_channels=in_channels,
                     seq_length=seq_len,
                     num_outputs=out_dim).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=hyperparams["learning_rate"])
    epochs = hyperparams["epochs"]
    es = EarlyStopping(patience=5, delta=5e-5)

    train_losses, val_losses = [], []
    for epoch in range(epochs):
        # -------- train ---------------------------------------
        model.train(); epoch_train = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward(); optimizer.step()
            epoch_train += loss.item()
        train_losses.append(epoch_train / len(train_loader))

        # -------- val -----------------------------------------
        model.eval(); epoch_val = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                epoch_val += criterion(model(xb), yb).item()
        val_loss = epoch_val / len(val_loader)
        val_losses.append(val_loss)

        print(f"[CNN gap={gap}] Epoch {epoch+1}/{epochs}  "
              f"train={train_losses[-1]:.4f}  val={val_losses[-1]:.4f}")
        
        es(val_loss)
        if es.early_stop:
                print(f"[CNN] Early stopping efter {epoch+1} epochs "
                        f"(bedste val.loss = {es.best_loss:.4f})")
                break


   
    # -------- gem -------------------------------------------
    repo_root  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    checkpoints = os.path.join(repo_root, "checkpoints")
    logs_dir    = os.path.join(repo_root, "logs", "training_logs")
    os.makedirs(checkpoints, exist_ok=True)
    os.makedirs(logs_dir,    exist_ok=True)

    torch.save(model.state_dict(), os.path.join(checkpoints, model_name))
    print(f"[INFO] CNN gemt → checkpoints/{model_name}")

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    pd.DataFrame({"epoch": range(1, len(train_losses)+1),
                  "train_loss": train_losses,
                  "val_loss": val_losses}) \
      .to_csv(os.path.join(
              logs_dir,
              f"{model_name.replace('.pth','')}_logs_{ts}.csv"),
              index=False)
    return model

def train_rbfpytorch(train_loader, val_loader, xgb_data, hyperparameters,
                     model_name, gap):
    print(f"[INFO] Training RBFNet (PyTorch) | gap={gap}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 0) Tjek for NaN/inf i xgb_data før noget andet
    import numpy as _np
    assert not _np.isnan(xgb_data["X_train"]).any(), "X_train indeholder NaN!"
    assert not _np.isnan(xgb_data["y_train"]).any(), "y_train indeholder NaN!"
    assert not _np.isinf(xgb_data["X_train"]).any(), "X_train indeholder inf!"
    assert not _np.isinf(xgb_data["y_train"]).any(), "y_train indeholder inf!"

    # 1) Model + centroids
    X_train_all = torch.from_numpy(xgb_data["X_train"].astype(np.float32))
    centroids   = initialize_centroids(
        X_train_all,
        hyperparameters["num_hidden_neurons"]
    )
    model = RBFNet(
        input_dim = X_train_all.shape[1],
        num_hidden_neurons = hyperparameters["num_hidden_neurons"],
        output_dim = xgb_data["y_train"].shape[1],
        betas = hyperparameters.get("beta_init", 0.5),
        initial_centroids = centroids
    ).to(device)

    # 2) Criterion og OPT med lavere LR
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=hyperparameters.get("learning_rate", 0.1) * 0.01  # fx 100× lavere
    )

    # 3) Anomaly detection
    torch.autograd.set_detect_anomaly(True)

    es        = EarlyStopping(patience=5, delta=5e-5)
    train_losses, val_losses = [], []

    for epoch in range(1, hyperparameters["epochs"] + 1):
        # ---- train ----
        model.train()
        tot_train = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb.reshape(xb.size(0), -1))
            loss = criterion(out, yb)
            loss.backward()

            # 4) Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            tot_train += loss.item() * xb.size(0)

        train_loss = tot_train / len(train_loader.dataset)
        train_losses.append(train_loss)

        # ---- val ----
        model.eval()
        tot_val = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb.reshape(xb.size(0), -1))
                tot_val += criterion(out, yb).item() * xb.size(0)
        val_loss = tot_val / len(val_loader.dataset)
        val_losses.append(val_loss)

        print(f"Epoch {epoch:2d}  train={train_loss:.4e}  val={val_loss:.4e}")
        es(val_loss)
        if es.early_stop:
            print(f"[EarlyStopping] Stoppede efter {epoch} epoker "
                  f"(bedste val.loss = {es.best_loss:.4e})")
            break

    # ----- Gem vægte + logs som før -----
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ckpt = os.path.join(root, "checkpoints", model_name)
    torch.save(model.state_dict(), ckpt)
    print(f"[INFO] Gemte vægte → {ckpt}")

    logs_dir = os.path.join(root, "logs", "training_logs")
    os.makedirs(logs_dir, exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    pd.DataFrame({
        "epoch":      range(1, len(train_losses)+1),
        "train_loss": train_losses,
        "val_loss":   val_losses
    }).to_csv(os.path.join(logs_dir,
               f"{model_name.replace('.pth','')}_logs_{now}.csv"),
              index=False)

    return model

def train_rbfpytorch_static(train_loader, val_loader,
                            xgb_data, hyperparams,
                            model_name):
    """Statisk (non-sekventiel) træning af PyTorch-RBF-net."""
    print("[INFO] Training *static* RBFNet")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- init centroids & model ---------------------------------
    X_train_all = torch.from_numpy(xgb_data["X_train"].astype(np.float32))
    centroids   = initialize_centroids(
        X_train_all, hyperparams["num_hidden_neurons"]
    )

    model = RBFNet(
        input_dim = X_train_all.shape[1],
        num_hidden_neurons = hyperparams["num_hidden_neurons"],
        output_dim = xgb_data["y_train"].shape[1],
        betas  = hyperparams.get("beta_init", 0.5),
        initial_centroids = centroids
    ).to(dev)

    crit = nn.MSELoss()
    opt  = optim.Adam(model.parameters(),
                      lr = hyperparams.get("learning_rate", 1e-3))

    es = EarlyStopping(patience=5, delta=5e-5)
    tr_losses, va_losses = [], []

    for ep in range(1, hyperparams["epochs"] + 1):
        # ---------- train ----------
        model.train(); tot = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * xb.size(0)
        tr_loss = tot / len(train_loader.dataset)
        tr_losses.append(tr_loss)

        # ---------- val ------------
        model.eval(); tot = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(dev), yb.to(dev)
                tot += crit(model(xb), yb).item() * xb.size(0)
        va_loss = tot / len(val_loader.dataset)
        va_losses.append(va_loss)

        print(f"Epoch {ep:3d}  train={tr_loss:.4e}  val={va_loss:.4e}")
        es(va_loss)
        if es.early_stop:
            print(f"[EarlyStopping] stoppede ved epoch {ep} "
                  f"(best val={es.best_loss:.4e})")
            break

    # ---------- gem vægte + logs ----------
    ckpt_dir = os.path.join(project_root, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(model.state_dict(),
               os.path.join(ckpt_dir, model_name))
    print(f"[INFO] Gemt → checkpoints/{model_name}")

    log_dir = os.path.join(project_root, "logs", "training_logs")
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    pd.DataFrame({
        "epoch":      range(1, len(tr_losses)+1),
        "train_loss": tr_losses,
        "val_loss":   va_losses
    }).to_csv(os.path.join(
        log_dir, f"{model_name.replace('.pth','')}_logs_{ts}.csv"),
        index=False)

    return model


if __name__ == "__main__":
    SELECTED_GAP = 10
    run_params   = batch_parameters | {"gap": SELECTED_GAP}

    train_rbfn_pytorch_flag = False      
    train_rbfn_static_flag  = True      
    train_cnn_flag          = False

    # ---------------- loaders to the sequantial models -----------
    if train_rbfn_pytorch_flag or train_cnn_flag:
        train_loader, val_loader, test_loader, xgb_data, *_ = \
            prepare_dataloaders(run_params)

    if train_cnn_flag:
        model_name = f"cnn_gap{SELECTED_GAP}.pth"
        train_cnn(train_loader, val_loader,
                  hyperparameters,
                  model_name=model_name,
                  gap=SELECTED_GAP)

    if train_rbfn_pytorch_flag:
        model_name = f"rbfpytorch_gap{SELECTED_GAP}.pth"
        train_rbfpytorch(train_loader, val_loader,
                         xgb_data, hyperparameters,
                         model_name=model_name, gap=SELECTED_GAP)

    if train_rbfn_static_flag:
        tr_loader, va_loader, xgb_dict = prepare_flat_dataloaders(batch_parameters)
        train_rbfpytorch_static(tr_loader, va_loader,
                                xgb_dict, hyperparameters,
                                model_name="rbfpytorch_static.pth")
    

   