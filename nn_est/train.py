import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
#from features import prepare_flat_dataloaders   # <-- ny import
import tensorflow as tf
import keras as keras


from datetime import datetime
import time
from features import load_data
from features import prepare_dataloaders
from features import prepare_flat_dataloaders
from hyperparameters import batch_parameters, hyperparameters
from hyperparameters import batch_parameters, hyperparameters

# Get the absolute path of the project's root directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))  # Move up one level

# Add project root to sys.path
sys.path.append(project_root)

#CNN
from models.CNN import CNNModel
from models.FFNN import FFNNModel
from models.OneLayerNN import OneLayerNN
from models.TCN import TCNModel
from models.CNNLSTM import CNNLSTMModel
from models.LSTM import LSTMModel


# Save the model with current date and time
current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
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
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"[INFO] Using device: {device}")

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


#FFNN Training Function
def train_ffnn(batch_params, hyperparameters):
    print("Training FFNN")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    train_loader, val_loader, _, _, _, _ = prepare_dataloaders(batch_params)
    
    model = FFNNModel(
        input_dim=train_loader.dataset[0][0].shape[-1],
        output_dim=train_loader.dataset[0][1].shape[-1],
        seq_len=batch_params['total_len'] // batch_params['gap'],
        dropout=hyperparameters['dropout']
    )
    model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=hyperparameters['learning_rate'], weight_decay=1e-4)
    
    train_losses, val_losses = [], []
    
    for epoch in range(hyperparameters['epochs']):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        print(f"[{epoch+1}/{hyperparameters['epochs']}] Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
    
    # Create directories if they don't exist
    checkpoints_dir = os.path.join(project_root, "checkpoints")
    logs_dir = os.path.join(project_root, "logs", "training_logs")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    # Save the model with current date and time
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    model_path = os.path.join(checkpoints_dir, f"ffnn_sequenced_{current_datetime}_latest.pth") 
    torch.save(model.state_dict(), model_path)
    
    # Save training logs to CSV inside the training_logs folder
    log_path = os.path.join(logs_dir, f"ffnn_training_logs_{current_datetime}.csv")
    logs_df = pd.DataFrame({"Epoch": range(1, len(train_losses) + 1), "Train Loss": train_losses, "Val Loss": val_losses})
    logs_df.to_csv(log_path, index=False)
    print(f"Training logs saved as {log_path}")

# One-Layer NN Training Function
def train_one_layer_nn(batch_params, hyperparameters):
    print("Training One-Layer NN")
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"[INFO] Using device: {device}")
    
    train_loader, val_loader, _, _, _, _ = prepare_dataloaders(batch_params)
    
    model = OneLayerNN(
        input_dim=train_loader.dataset[0][0].shape[-1],
        output_dim=train_loader.dataset[0][1].shape[-1],
        seq_len=batch_params['total_len'] // batch_params['gap'],
    )
    model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=hyperparameters['learning_rate'], weight_decay=hyperparameters['weight_decay'])
    
    train_losses, val_losses = [], []
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Early stopping
    best_val_loss = float('inf')
    patience = 3
    counter = 0

    train_losses, val_losses = [], []

    for epoch in range(hyperparameters['epochs']):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(f"[{epoch+1}/{hyperparameters['epochs']}] Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
    
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), os.path.join(project_root, "checkpoints", f"one_layer_nn_sequenced_{current_datetime}_latest.pth"))
        else:
            counter += 1
            if counter >= patience:
                print(f"[INFO] Early stopping triggered at epoch {epoch+1}")
                break

        scheduler.step()  # update learning rate
        
    # Create directories if they don't exist
    checkpoints_dir = os.path.join(project_root, "checkpoints")
    logs_dir = os.path.join(project_root, "logs", "training_logs")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    # Save the model with current date and time
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
    # Save training logs to CSV inside the training_logs folder
    log_path = os.path.join(logs_dir, f"one_layer_nn_training_logs_{current_datetime}.csv")
    logs_df = pd.DataFrame({"Epoch": range(1, len(train_losses) + 1), "Train Loss": train_losses, "Val Loss": val_losses})
    logs_df.to_csv(log_path, index=False)
    print(f"Training logs saved as {log_path}")    



# TCN Training Function
def train_tcn(train_loader, val_loader, batch_params, hyperparameters):
    print("Training TCN")
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"[INFO] Using device: {device}")
    
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    model = TCNModel(
        input_dim=train_loader.dataset[0][0].shape[-1],
        output_dim=train_loader.dataset[0][1].shape[-1],
        seq_len=batch_params['total_len'] // batch_params['gap'],
        num_channels=hyperparameters['num_channels'],
        kernel_size=hyperparameters['kernel_size'],
        kernel_initializer = hyperparameters['kernel_initializer'],
        dropout=hyperparameters['dropout'],
        causal=hyperparameters['causal'],
        use_skip_connections=hyperparameters['use_skip_connections'],
        use_norm=hyperparameters['use_norm'],
        activation=hyperparameters['activation']
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=hyperparameters['learning_rate'], weight_decay=hyperparameters['weight_decay'])

    early_stopping = EarlyStopping(patience=5, delta=0.00005)

    train_losses = []
    val_losses = []

    # Start timer
    start_time = time.time()

    for epoch in range(hyperparameters["epochs"]):
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            X_batch = X_batch.permute(0, 2, 1)  # [batch, features, time]

            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Evaluate on validation set
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                X_batch = X_batch.permute(0, 2, 1)  # !! Important for TCN !!
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{hyperparameters['epochs']}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        end_time = time.time()
        training_duration = end_time - start_time

        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered. Stopping training.")
            break

    # Create directories if they don't exist
    checkpoints_dir = os.path.join(project_root, "checkpoints")
    logs_dir = os.path.join(project_root, "logs", "training_logs")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    model_path = os.path.join(checkpoints_dir, f"tcn_latest.pth") 
    torch.save(model.state_dict(), model_path)
    
    # Save training logs to CSV inside the training_logs folder
    log_path = os.path.join(logs_dir, f"tcn_training_logs_{current_datetime}.csv")
    logs_df = pd.DataFrame({"Epoch": range(1, len(train_losses) + 1), "Train Loss": train_losses, "Val Loss": val_losses})
    logs_df.to_csv(log_path, index=False)
    print(f"Training logs saved as {log_path}")

    # Save training time 
    train_time_df = pd.DataFrame({"Training Time (seconds)": [training_duration]})
    train_time_path = os.path.join(logs_dir, f"tcn_training_time_{current_datetime}.csv")
    train_time_df.to_csv(train_time_path, index=False)
    print(f"Training time saved at {train_time_path}")


#Training CNN-LSTM
def train_cnnlstm(train_loader, val_loader, batch_params, hyperparameters):
    print("Training CNN-LSTM")
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"[INFO] Using device: {device}")

    model = CNNLSTMModel(
        input_dim=train_loader.dataset[0][0].shape[-1],
        output_dim=train_loader.dataset[0][1].shape[-1],
        seq_len=batch_params['total_len'] // batch_params['gap'],
        cnn_filters=hyperparameters['cnn_filters'],
        lstm_hidden=hyperparameters['lstm_hidden'],
        dropout=hyperparameters['dropout'],
        dense_units=hyperparameters['dense_units']
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=hyperparameters['learning_rate'], weight_decay=hyperparameters['weight_decay'])

    early_stopping = EarlyStopping(patience=5, delta=0.00005)

    train_losses = []
    val_losses = []

    start_time = time.time()

    for epoch in range(hyperparameters["epochs"]):
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Evaluation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{hyperparameters['epochs']}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        end_time = time.time()
        training_duration = end_time - start_time

        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered. Stopping training.")
            break

    # Create directories if they don't exist
    checkpoints_dir = os.path.join(project_root, "checkpoints")
    logs_dir = os.path.join(project_root, "logs", "training_logs")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    model_path = os.path.join(checkpoints_dir, f"cnn_lstm_latest_{current_datetime}.pth") 
    torch.save(model.state_dict(), model_path)
    
    # Save training logs to CSV inside the training_logs folder
    log_path = os.path.join(logs_dir, f"CNN-LSTM_training_logs_{current_datetime}.csv")
    logs_df = pd.DataFrame({"Epoch": range(1, len(train_losses) + 1), "Train Loss": train_losses, "Val Loss": val_losses})
    logs_df.to_csv(log_path, index=False)
    print(f"Training logs saved as {log_path}")

    # Save training time 
    train_time_df = pd.DataFrame({"Training Time (seconds)": [training_duration]})
    train_time_path = os.path.join(logs_dir, f"cnnlstm_training_time_{current_datetime}.csv")
    train_time_df.to_csv(train_time_path, index=False)
    print(f"Training time saved at {train_time_path}")


#Training LSTM
def train_lstm(train_loader, val_loader, batch_params, hyperparameters):
    print("Training LSTM")
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"[INFO] Using device: {device}")

    model = LSTMModel(
        input_dim=train_loader.dataset[0][0].shape[-1],
        output_dim=train_loader.dataset[0][1].shape[-1],
        seq_len=batch_params['total_len'] // batch_params['gap'],
        lstm_hidden=hyperparameters['lstm_hidden'],
        num_layers=hyperparameters['num_layers_lstm'],
        dropout=hyperparameters['dropout'],
        dense_units=hyperparameters['dense_units']
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=hyperparameters['learning_rate'], weight_decay=hyperparameters['weight_decay'])
    
    early_stopping = EarlyStopping(patience=5, delta=0.00005)

    train_losses = []
    val_losses = []

    start_time = time.time()

    for epoch in range(hyperparameters["epochs"]):
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Evaluation 
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{hyperparameters['epochs']}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        end_time = time.time()
        training_duration = end_time - start_time

        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered. Stopping training.")
            break

    # Create directories if they don't exist
    checkpoints_dir = os.path.join(project_root, "checkpoints")
    logs_dir = os.path.join(project_root, "logs", "training_logs")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    model_path = os.path.join(checkpoints_dir, f"lstm_latest_{current_datetime}.pth") 
    torch.save(model.state_dict(), model_path)
    
    # Save training logs to CSV inside the training_logs folder
    log_path = os.path.join(logs_dir, f"LSTM_training_logs_{current_datetime}.csv")
    logs_df = pd.DataFrame({"Epoch": range(1, len(train_losses) + 1), "Train Loss": train_losses, "Val Loss": val_losses})
    logs_df.to_csv(log_path, index=False)
    print(f"Training logs saved as {log_path}")

    # Save training time 
    train_time_df = pd.DataFrame({"Training Time (seconds)": [training_duration]})
    train_time_path = os.path.join(logs_dir, f"lstm_training_time_{current_datetime}.csv")
    train_time_df.to_csv(train_time_path, index=False)
    print(f"Training time saved at {train_time_path}")

if __name__ == "__main__":              
    SELECTED_GAP = 10
    run_params   = batch_parameters | {"gap": SELECTED_GAP}
    # Load preprocessed data
    train_loader, val_loader, _, xgb_data, scaler_x, scaler_y = prepare_dataloaders(batch_parameters)
    
    # DO THIS FOR EVERY MODEL YOU WANT TO TRAIN
    
    train_transformer_flag = False  # Set to True to train Transformer
    train_xgboost_flag = False  # Set to True to train XGBoost
    train_ffnn_flag = False  # Set to True to train FFNN
    train_one_layer_nn_flag = False  # Set to True to train One-Layer NN
    train_tcn_flag = True
    train_cnnlstm_flag = True
    train_lstm_flag = True

    # Train Transformer if flag is set
    if train_transformer_flag:
        train_transformer(train_loader, val_loader, batch_parameters, hyperparameters)

    # Train XGBoost if flag is set
    if train_xgboost_flag:
        train_xgboost(xgb_data, hyperparameters)
        
    # Train FFNN if flag is set
    if train_ffnn_flag:
        train_ffnn(batch_parameters, hyperparameters)    
        
    # Train One-Layer NN if flag is set
    if train_one_layer_nn_flag:
        train_one_layer_nn(batch_parameters, hyperparameters)  

    # Train TCN if flag is set
    if train_tcn_flag:
        train_tcn(train_loader, val_loader, batch_parameters, hyperparameters)

    # Train CNN-LSTM if flag is set
    if train_cnnlstm_flag:
        train_cnnlstm(train_loader, val_loader, batch_parameters, hyperparameters)

    # Train LSTM if flag is set
    if train_lstm_flag:
        train_lstm(train_loader, val_loader, batch_parameters, hyperparameters)  
        

   

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
    

   