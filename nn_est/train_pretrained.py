"""
train_pretrained.py

This script illustrates how to adapt and fine-tune a pretrained Hugging Face MobileBERT
model for a time-series forecasting problem using your existing project structure.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from features import prepare_dataloaders  # Your data loading utilities
# Make sure to install transformers if you haven't:
# pip install transformers
from transformers import MobileBertModel, MobileBertConfig

# ------------------------------------------------------------------------------------
# Adjust these paths according to your project structure
# ------------------------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.append(project_root)

# ------------------------------------------------------------------------------------
# EarlyStopping class (as you used in train.py)
# ------------------------------------------------------------------------------------
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

# ------------------------------------------------------------------------------------
# MobileBertTimeSeries model
# ------------------------------------------------------------------------------------
from transformers import MobileBertConfig, MobileBertModel

class MobileBertTimeSeries(nn.Module):
    def __init__(self, pretrained_name: str, input_dim: int, output_dim: int):
        super().__init__()
        
        # 1) Create config and override embedding_size = hidden_size
        self.config = MobileBertConfig.from_pretrained(pretrained_name)
        self.config.embedding_size = self.config.hidden_size  # both = 512

        # 2) Load model with updated config
        self.mobilebert = MobileBertModel.from_pretrained(pretrained_name, config=self.config, ignore_mismatched_sizes=True)

        # 3) Freeze MobileBERT
        for param in self.mobilebert.parameters():
            param.requires_grad = False

        # 4) Now MobileBERT’s embeddings expect shape [batch_size, seq_len, 512]
        self.input_proj = nn.Linear(input_dim, self.config.hidden_size)  # 512
        self.regressor = nn.Linear(self.config.hidden_size, output_dim)

    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        input_embeds = self.input_proj(x)  # -> [batch_size, seq_len, 512]
        attention_mask = torch.ones(input_embeds.shape[:2], dtype=torch.long, device=x.device)
        outputs = self.mobilebert(inputs_embeds=input_embeds, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # shape [batch_size, 512]
        return self.regressor(pooled_output)

# ------------------------------------------------------------------------------------
# Training Function
# ------------------------------------------------------------------------------------
def train_pretrained(train_loader, val_loader, batch_params, hyperparams):
    """
    Train MobileBertTimeSeries model for time-series forecasting, freezing the
    pretrained MobileBERT weights and only training the new input projection
    and regression head.
    """
    print("Training MobileBERT (Pretrained) for Time-Series Forecasting")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Example: input_dim and output_dim can be derived from train_loader
    #   - We assume each batch is: (X_batch, y_batch)
    #   - X_batch shape: [batch_size, seq_len, input_dim]
    #   - y_batch shape: [batch_size, output_dim]
    example_X, example_y = next(iter(train_loader))
    input_dim = example_X.shape[-1]
    output_dim = example_y.shape[-1]

    # Initialize the model
    model = MobileBertTimeSeries(
        pretrained_name="google/mobilebert-uncased",  # or any MobileBERT variant
        input_dim=input_dim,
        output_dim=output_dim
    ).to(device)

    # We only train the projection + regressor layers
    # (mobilebert.* were already frozen)
    params_to_train = [p for p in model.parameters() if p.requires_grad]

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        params_to_train,
        lr=hyperparams['learning_rate'],
        weight_decay=hyperparams['weight_decay']
    )

    # Optional early stopping
    early_stopping = EarlyStopping(patience=5, delta=0.00001)

    train_losses = []
    val_losses = []

    for epoch in range(hyperparams['epochs']):
        model.train()
        running_train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()

        train_loss = running_train_loss / len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                running_val_loss += loss.item()

        val_loss = running_val_loss / len(val_loader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{hyperparams['epochs']}, "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered. Stopping training.")
            break

    # Save checkpoint and training logs
    checkpoints_dir = os.path.join(project_root, "checkpoints")
    logs_dir = os.path.join(project_root, "logs", "training_logs")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    model_path = os.path.join(checkpoints_dir, "mobilebert_latest.pth")
    torch.save(model.state_dict(), model_path)
    print(f"[INFO] Model checkpoint saved at {model_path}")

    log_path = os.path.join(logs_dir, f"mobilebert_training_logs_{current_datetime}.csv")
    logs_df = pd.DataFrame({
        "Epoch": range(1, len(train_losses) + 1),
        "Train Loss": train_losses,
        "Val Loss": val_losses
    })
    logs_df.to_csv(log_path, index=False)
    print(f"[INFO] Training logs saved at {log_path}")

    return model

# ------------------------------------------------------------------------------------
# Main script entry point
# ------------------------------------------------------------------------------------
if __name__ == "__main__":

    # Example batch parameters (adapt as needed)
    batch_params = {
        "gap": 10,
        "total_len": 100,
        "batch_size": 32,
    }

    # Example hyperparameters
    hyperparams = {
        "epochs": 100,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
    }

    # Prepare Data
    train_loader, val_loader, test_loader, _, scaler_x, scaler_y = prepare_dataloaders(batch_params)

    # Train the MobileBERT-based model
    model = train_pretrained(train_loader, val_loader, batch_params, hyperparams)

    # # -------------------------------------------------------------------------
    # # (Optional) Evaluate on the test set
    # # -------------------------------------------------------------------------
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.eval()
    # model.to(device)

    # criterion = nn.MSELoss()
    # test_loss = 0.0
    # all_preds = []
    # all_targets = []

    # with torch.no_grad():
    #     for X_batch, y_batch in test_loader:
    #         X_batch, y_batch = X_batch.to(device), y_batch.to(device)
    #         preds = model(X_batch)
    #         loss = criterion(preds, y_batch)
    #         test_loss += loss.item()

    #         # Store for evaluation metrics
    #         all_preds.append(preds.cpu().numpy())
    #         all_targets.append(y_batch.cpu().numpy())

    # test_loss /= len(test_loader)
    # all_preds = np.vstack(all_preds)
    # all_targets = np.vstack(all_targets)

    # rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    # mae = mean_absolute_error(all_targets, all_preds)
    # r2 = r2_score(all_targets, all_preds)

    # print(f"\n[TEST RESULTS] Loss: {test_loss:.4f} | RMSE: {rmse:.4f}, "
    #       f"MAE: {mae:.4f}, R²: {r2:.4f}")
