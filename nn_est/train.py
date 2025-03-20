import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
import keras as keras
from tensorflow.keras.callbacks import EarlyStopping




from datetime import datetime
from features import load_data
from features import prepare_dataloaders

# Get the absolute path of the project's root directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))  # Move up one level

# Add project root to sys.path
sys.path.append(project_root)

#rbf+keras
from models.RBF_Keras import initialize_centroids, build_rbf_model  # fx

from models.Transformer import TransformerModel
#from models.XGBoost import XGBoostModel
from models.RadialBasisFunctionModel import RBFN_model
# Define EarlyStopping class
"""
class EarlyStopping:
    def __init__(self, patience=10, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.best_loss = np.inf
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True"
"""

# Define training function
def train_transformer(train_loader, val_loader, batch_params, hyperparameters):
    print("Training Transformer")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = TransformerModel(
        input_dim=train_loader.dataset[0][0].shape[-1],
        output_dim=train_loader.dataset[0][1].shape[-1],
        seq_len=batch_params['total_len'] // batch_params['gap'],
        d_model=hyperparameters['d_model'],
        nhead=hyperparameters['nhead'],
        num_layers=hyperparameters['num_layers'],
        dim_feedforward=hyperparameters['dim_feedforward'],
        dropout=hyperparameters['dropout'],
        layer_norm_eps=hyperparameters['layer_norm_eps']
    )
    model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=hyperparameters['learning_rate'], weight_decay=hyperparameters['weight_decay'])
    
    early_stopping = EarlyStopping(patience=5, delta=0.00001)
    
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
        
        # Evaluate on validation set
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
        
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered. Stopping training.")
            break
    
    # Create directories if they don't exist
    checkpoints_dir = os.path.join(project_root, "checkpoints")
    logs_dir = os.path.join(project_root, "logs", "training_logs")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    # Save the model with current date and time
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    model_path = os.path.join(checkpoints_dir, f"transformer_latest.pth") 
    torch.save(model.state_dict(), model_path)
    
    # Save training logs to CSV inside the training_logs folder
    log_path = os.path.join(logs_dir, f"training_logs_{current_datetime}.csv")
    logs_df = pd.DataFrame({"Epoch": range(1, len(train_losses) + 1), "Train Loss": train_losses, "Val Loss": val_losses})
    logs_df.to_csv(log_path, index=False)
    print(f"Training logs saved as {log_path}")


# XGBoost Training Function
def train_xgboost(xgb_data, hyperparameters):
    print("Training XGBoost")

    # Initialize XGBoost model
    xgb_model = XGBoostModel(
        n_estimators=hyperparameters.get("n_estimators", 200),
        max_depth=hyperparameters.get("max_depth", 6),
        learning_rate=hyperparameters.get("learning_rate", 0.05)
    )

    # Train model
    xgb_model.train(xgb_data["X_train"], xgb_data["y_train"])

    # Validate model
    predictions = xgb_model.predict(xgb_data["X_val"])

    # Compute metrics
    rmse = np.sqrt(mean_squared_error(xgb_data["y_val"], predictions))
    mae = mean_absolute_error(xgb_data["y_val"], predictions)
    r2 = r2_score(xgb_data["y_val"], predictions)

    print(f"XGBoost Results: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

     # Save model in checkpoints directory
    checkpoints_dir = os.path.join(project_root, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    xgboost_model_path = os.path.join(checkpoints_dir, "xgboost_latest.json")
    xgb_model.model.save_model(xgboost_model_path)
    print(f"[INFO] XGBoost model saved at {xgboost_model_path}")
    
    return xgb_model

def train_rbfn(xgb_data, hyperparameters):
   
    print("[INFO] Training RBFN")

    # getting data
    X_train = xgb_data["X_train"]
    y_train = xgb_data["y_train"]
    X_val   = xgb_data["X_val"]
    y_val   = xgb_data["y_val"]

    #  setting parameters
    input_dim  = X_train.shape[1]
    output_dim = y_train.shape[1]
    num_hidden_neurons = hyperparameters.get("num_hidden_neurons", 100)
    learning_rate      = hyperparameters.get("learning_rate", 0.01)
    epochs             = hyperparameters.get("epochs", 10)
    
    #init
    model = RBFN_model(input_dim, num_hidden_neurons, output_dim, learning_rate=learning_rate)

    #train
    train_losses, val_losses = model.train(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        epochs=epochs
    )

    #val
    val_pred = model.predict(X_val)
    val_mse = mean_squared_error(y_val, val_pred)
    val_rmse = np.sqrt(val_mse)
    print(f"[INFO] RBFN Validation: MSE={val_mse:.4f}, RMSE={val_rmse:.4f}")

    # saving the modelparameters, to .npz (centroids, betas, weights)
    checkpoints_dir = os.path.join(project_root, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    model_path = os.path.join(checkpoints_dir, "rbfn_latest.npz")

    np.savez(model_path,
             centroids=model.centroids,
             betas=model.betas,
             weights=model.weights)
    print(f"[INFO] RBFN model saved at {model_path}")

    #logging epochs, train_loss, val_loss til CSV
    logs_dir = os.path.join(project_root, "logs", "training_logs")
    os.makedirs(logs_dir, exist_ok=True)
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(logs_dir, f"rbfn_training_logs_{current_datetime}.csv")
    
    logs_df = pd.DataFrame({
        "Epoch": range(1, len(train_losses)+1),
        "Train Loss": train_losses,
        "Val Loss": val_losses
    })
    logs_df.to_csv(log_path, index=False)
    print(f"[INFO] RBFN training logs saved at {log_path}")

    return model

def train_rbf_keras(xgb_data, hyperparameters):
    print("[INFO] Training RBFN (Keras version)")

    # getting data
    X_train = xgb_data["X_train"]
    y_train = xgb_data["y_train"]
    X_val   = xgb_data["X_val"]
    y_val   = xgb_data["y_val"]

    # setting parameters
    input_dim  = X_train.shape[1]
    output_dim = y_train.shape[1]
    num_hidden_neurons = hyperparameters.get("num_hidden_neurons", 100)
    learning_rate      = hyperparameters.get("learning_rate", 0.01)
    epochs             = hyperparameters.get("epochs", 10)
    batch_size         = 32  # eller læs det fra dine hyperparametre

    # KMeans initialization for centroids
    centroids_init = initialize_centroids(X_train, num_hidden_neurons)

    # Build Keras model
    model = build_rbf_model(input_dim, num_hidden_neurons, output_dim,
                            initial_centroids=centroids_init)

    # Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')

    # Early stopping
    patience = 5
    es_callback = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

    # Fit model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es_callback],
        verbose=1
    )

    # Train & validation losses
    train_losses = history.history['loss']
    val_losses   = history.history['val_loss']

    # Evaluate on val
    val_pred = model.predict(X_val)
    val_mse = mean_squared_error(y_val, val_pred)
    val_rmse = np.sqrt(val_mse)
    print(f"[INFO] Keras-RBFN Validation: MSE={val_mse:.4f}, RMSE={val_rmse:.4f}")

    

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    checkpoints_dir = os.path.join(project_root, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    model_path = os.path.join(checkpoints_dir, "rbfn_keras_latest.keras")
    
    model.save(model_path)
    print(f"[INFO] Keras-RBFN model saved at {model_path}")

    # Logging training
    logs_dir = os.path.join(project_root, "logs", "training_logs")
    os.makedirs(logs_dir, exist_ok=True)
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(logs_dir, f"rbfn_keras_training_logs_{current_datetime}.csv")

    logs_df = pd.DataFrame({
        "Epoch": range(1, len(train_losses)+1),
        "Train Loss": train_losses,
        "Val Loss": val_losses
    })
    logs_df.to_csv(log_path, index=False)
    print(f"[INFO] Keras-RBFN training logs saved at {log_path}")

    return model





if __name__ == "__main__":
    batch_params = {
        "gap": 10,
        "total_len": 100,
        "batch_size": 32,
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
        "epochs": 10,                  # CHANGE THIS TO 10 LATER
        "n_estimators": 200,
        "max_depth": 6,

        #rbf-parameters:
        "learning_rate": 0.01,
        "num_hidden_neurons": 100,
    }
    
    # Load preprocessed data

   
    #Prepare loader returns 6 values? 
    #train_loader, val_loader, xgb_data, scaler_x, scaler_y = prepare_dataloaders(batch_params)
    
    #updated version
    train_loader, val_loader, test_loader, xgb_data, scaler_x, scaler_y = prepare_dataloaders(batch_params)

    
    # DO THIS FOR EVERY MODEL YOU WANT TO TRAIN
    
    train_transformer_flag = False  # Set to True to train Transformer
    train_xgboost_flag = False  # Set to True to train XGBoost
    train_rbfn_flag = True  # vil nu betyde: "kør Keras RBF"


    # Train Transformer if flag is set
    if train_transformer_flag:
        train_transformer(train_loader, val_loader, batch_params, hyperparameters)

    # Train XGBoost if flag is set
    if train_xgboost_flag:
        train_xgboost(xgb_data, hyperparameters)

    if train_rbfn_flag:
        #train_rbfn(xgb_data, hyperparameters)
        train_rbf_keras(xgb_data, hyperparameters)
   