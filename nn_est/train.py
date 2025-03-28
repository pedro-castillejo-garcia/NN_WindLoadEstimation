import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from datetime import datetime
from features import load_data
from features import prepare_dataloaders
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

# Define EarlyStopping class
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
                self.early_stop = True

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
    log_path = os.path.join(logs_dir, f"transformer_training_logs_{current_datetime}.csv")
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

#FFNN Training Function
def train_ffnn(batch_params, hyperparameters):
    print("Training FFNN")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    
    model_path = os.path.join(checkpoints_dir, f"ffnn_latest.pth") 
    torch.save(model.state_dict(), model_path)
    
    # Save training logs to CSV inside the training_logs folder
    log_path = os.path.join(logs_dir, f"ffnn_training_logs_{current_datetime}.csv")
    logs_df = pd.DataFrame({"Epoch": range(1, len(train_losses) + 1), "Train Loss": train_losses, "Val Loss": val_losses})
    logs_df.to_csv(log_path, index=False)
    print(f"Training logs saved as {log_path}")



# One-Layer NN Training Function
def train_one_layer_nn(batch_params, hyperparameters):
    print("Training One-Layer NN")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, _, _, _, _ = prepare_dataloaders(batch_params)
    
    model = OneLayerNN(
        input_dim=train_loader.dataset[0][0].shape[-1],
        output_dim=train_loader.dataset[0][1].shape[-1],
        seq_len=batch_params['total_len'] // batch_params['gap'],
        dropout=hyperparameters['dropout']
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
            torch.save(model.state_dict(), os.path.join(project_root, "checkpoints", "one_layer_nn_latest.pth"))
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = TCNModel(
        input_dim=train_loader.dataset[0][0].shape[-1],
        output_dim=train_loader.dataset[0][1].shape[-1],
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

    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0

    train_losses = []
    val_losses = []

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

        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                X_batch = X_batch.permute(0, 2, 1)
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                val_loss += loss.item()

                all_preds.append(predictions.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{hyperparameters['epochs']}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            patience_counter = 0
            checkpoint_dir = os.path.join(project_root, "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "tcn_latest.pth"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # Log metrics after final epoch
    y_true = np.vstack(all_targets)
    y_pred = np.vstack(all_preds)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"TCN Validation Metrics: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

    # # Save logs
    # logs_dir = os.path.join(project_root, "logs", "training_logs")
    # os.makedirs(logs_dir, exist_ok=True)
    # current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # log_path = os.path.join(logs_dir, f"tcn_training_logs_{current_datetime}.csv")
    # df_logs = pd.DataFrame({"Epoch": range(1, len(train_losses)+1), "Train Loss": train_losses, "Val Loss": val_losses})
    # df_logs.to_csv(log_path, index=False)
    # print(f"Training logs saved at: {log_path}") 
    
    # Create directories if they don't exist
    checkpoints_dir = os.path.join(project_root, "checkpoints")
    logs_dir = os.path.join(project_root, "logs", "training_logs")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    # Save the model with current date and time
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    model_path = os.path.join(checkpoints_dir, f"TCN_latest.pth") 
    torch.save(model.state_dict(), model_path)
    
    # Save training logs to CSV inside the training_logs folder
    log_path = os.path.join(logs_dir, f"TCN_training_logs_{current_datetime}.csv")
    logs_df = pd.DataFrame({"Epoch": range(1, len(train_losses) + 1), "Train Loss": train_losses, "Val Loss": val_losses})
    logs_df.to_csv(log_path, index=False)
    print(f"Training logs saved as {log_path}")

    
    

#Training CNN-LSTM
def train_cnnlstm(train_loader, val_loader, batch_params, hyperparameters):
    print("Training CNN-LSTM")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    # optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters.get("learning_rate", 1e-4), weight_decay=hyperparameters.get("weight_decay",1e-4))
    optimizer = optim.AdamW(model.parameters(), lr=hyperparameters['learning_rate'], weight_decay=hyperparameters['weight_decay'])

    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0
    train_losses = []
    val_losses = []

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

        model.eval()
        val_loss = 0.0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                val_loss += loss.item()
                all_preds.append(predictions.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{hyperparameters['epochs']}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            patience_counter = 0
            checkpoint_dir = os.path.join(project_root, "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "cnn_lstm_latest.pth"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # Evaluation metrics
    y_true = np.vstack(all_targets)
    y_pred = np.vstack(all_preds)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"CNN-LSTM Validation Metrics: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

    # # Save logs
    # logs_dir = os.path.join(project_root, "logs", "training_logs")
    # os.makedirs(logs_dir, exist_ok=True)
    # current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # log_path = os.path.join(logs_dir, f"cnn_lstm_training_logs_{current_datetime}.csv")
    # df_logs = pd.DataFrame({"Epoch": range(1, len(train_losses)+1), "Train Loss": train_losses, "Val Loss": val_losses})
    # df_logs.to_csv(log_path, index=False)
    # print(f"Training logs saved at: {log_path}")


    # Create directories if they don't exist
    checkpoints_dir = os.path.join(project_root, "checkpoints")
    logs_dir = os.path.join(project_root, "logs", "training_logs")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    # Save the model with current date and time
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    model_path = os.path.join(checkpoints_dir, f"CNN_LSTM_latest.pth") 
    torch.save(model.state_dict(), model_path)
    
    # Save training logs to CSV inside the training_logs folder
    log_path = os.path.join(logs_dir, f"CNN_LSTM_training_logs_{current_datetime}.csv")
    logs_df = pd.DataFrame({"Epoch": range(1, len(train_losses) + 1), "Train Loss": train_losses, "Val Loss": val_losses})
    logs_df.to_csv(log_path, index=False)
    print(f"Training logs saved as {log_path}")


#Training LSTM
def train_lstm(train_loader, val_loader, batch_params, hyperparameters):
    print("Training LSTM")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LSTMModel(
        input_dim=train_loader.dataset[0][0].shape[-1],
        output_dim=train_loader.dataset[0][1].shape[-1],
        lstm_hidden=hyperparameters['lstm_hidden'],
        num_layers=hyperparameters['num_layers_lstm'],
        dropout=hyperparameters['dropout'],
        dense_units=hyperparameters['dense_units']
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=hyperparameters['learning_rate'], weight_decay=hyperparameters['weight_decay'])
    
    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0
    train_losses = []
    val_losses = []

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

        model.eval()
        val_loss = 0.0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                val_loss += loss.item()
                all_preds.append(predictions.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{hyperparameters['epochs']}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            patience_counter = 0
            checkpoint_dir = os.path.join(project_root, "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "lstm_latest.pth"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # Metrics
    y_true = np.vstack(all_targets)
    y_pred = np.vstack(all_preds)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"LSTM Validation Metrics: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

    # # Save logs
    # logs_dir = os.path.join(project_root, "logs", "training_logs")
    # os.makedirs(logs_dir, exist_ok=True)
    # current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # log_path = os.path.join(logs_dir, f"lstm_training_logs_{current_datetime}.csv")
    # df_logs = pd.DataFrame({"Epoch": range(1, len(train_losses)+1), "Train Loss": train_losses, "Val Loss": val_losses})
    # df_logs.to_csv(log_path, index=False)
    # print(f"Training logs saved at: {log_path}")

    # Create directories if they don't exist
    checkpoints_dir = os.path.join(project_root, "checkpoints")
    logs_dir = os.path.join(project_root, "logs", "training_logs")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    # Save the model with current date and time
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    model_path = os.path.join(checkpoints_dir, f"LSTM_latest.pth") 
    torch.save(model.state_dict(), model_path)
    
    # Save training logs to CSV inside the training_logs folder
    log_path = os.path.join(logs_dir, f"LSTM_training_logs_{current_datetime}.csv")
    logs_df = pd.DataFrame({"Epoch": range(1, len(train_losses) + 1), "Train Loss": train_losses, "Val Loss": val_losses})
    logs_df.to_csv(log_path, index=False)
    print(f"Training logs saved as {log_path}")




if __name__ == "__main__":              
    
    # Load preprocessed data
    train_loader, val_loader, _, xgb_data, scaler_x, scaler_y = prepare_dataloaders(batch_parameters)
    
    # DO THIS FOR EVERY MODEL YOU WANT TO TRAIN
    
    train_transformer_flag = False  # Set to True to train Transformer
    train_xgboost_flag = False  # Set to True to train XGBoost
    train_ffnn_flag = True  # Set to True to train FFNN
    train_one_layer_nn_flag = False  # Set to True to train One-Layer NN
    train_tcn_flag = False
    train_cnnlstm_flag = False
    train_lstm_flag = False

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
        