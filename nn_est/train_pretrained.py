import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Import the data loading function
from features import load_data

# NEW: Import from Hugging Face
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerModel


# Define EarlyStopping class (unchanged)
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

                
# A small regression head to map from d_model -> your 3 targets
class RegressionHead(nn.Module):
    def __init__(self, d_model, output_dim):
        super().__init__()
        self.linear = nn.Linear(d_model, output_dim)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model) or (batch_size, d_model)
        return self.linear(x)


def train_model(train_loader, val_loader, hyperparameters):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ##################################################
    # 1) Create a Hugging Face TimeSeriesTransformer
    ##################################################
    # Because you have 10 time steps total (in X_batch),
    # you can treat them all as 'context_length'=10
    # and 'prediction_length'=1 (just forecasting the final step).
    # This is single-step forecasting for 3 target variables.
    #
    # input_size=12 means each time-step is 12-dimensional.
    #
    # We'll also set num_encoder_layers=num_decoder_layers=2
    # to mimic your old "num_layers=2" in the custom model.
    ##################################################

    config = TimeSeriesTransformerConfig(
        prediction_length=1,
        context_length=10,
        input_size=12,            # 12 input features
        num_encoder_layers=hyperparameters['num_layers'],
        num_decoder_layers=hyperparameters['num_layers'],
        d_model=hyperparameters['d_model'],
        n_head=hyperparameters['nhead'],
        dim_feedforward=hyperparameters['dim_feedforward'],
        dropout=hyperparameters['dropout'],
        # layer_norm_eps can be set if you like:
        # layer_norm_eps=hyperparameters['layer_norm_eps'],
    )

    # Build the TimeSeriesTransformerModel
    model = TimeSeriesTransformerModel(config)

    # Build a small linear head for 3 targets
    regression_head = RegressionHead(d_model=hyperparameters['d_model'],
                                     output_dim=train_loader.dataset[0][1].shape[-1])

    model.to(device)
    regression_head.to(device)

    # Define MSE loss and optimizer (AdamW)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        list(model.parameters()) + list(regression_head.parameters()),
        lr=hyperparameters['learning_rate'],
        weight_decay=hyperparameters['weight_decay']
    )
    
    early_stopping = EarlyStopping(patience=5, delta=1e-5)

    train_losses, val_losses = [], []

    for epoch in range(hyperparameters['epochs']):
        ##################################################
        # Training Loop
        ##################################################
        model.train()
        regression_head.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            # X_batch: (batch_size, 10, 12)
            # y_batch: (batch_size, 3)
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            # Forward pass
            # We'll feed all 10 steps as "past_values" (context),
            # and *no* future_values since it's single-step prediction.
            outputs = model(past_values=X_batch)
            # outputs.last_hidden_state: (batch_size, 10, d_model)

            # We only want the final time-step's hidden state:
            hidden_final_step = outputs.last_hidden_state[:, -1, :]  # (batch_size, d_model)

            # Map hidden state -> 3 target values
            predictions = regression_head(hidden_final_step)  # (batch_size, 3)

            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        ##################################################
        # Validation Loop
        ##################################################
        model.eval()
        regression_head.eval()
        val_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                outputs = model(past_values=X_batch)
                hidden_final_step = outputs.last_hidden_state[:, -1, :]
                predictions = regression_head(hidden_final_step)

                loss = criterion(predictions, y_batch)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{hyperparameters['epochs']}, "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered. Stopping training.")
            break

    ##################################################
    # Plot training vs. validation loss
    ##################################################
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid()
    plt.show()

    ##################################################
    # Save the model + head
    ##################################################
    torch.save({
        "transformer_config": config.to_dict(),
        "transformer_state_dict": model.state_dict(),
        "regression_head_state_dict": regression_head.state_dict()
    }, "hf_transformer_model.pth")

    print("Model saved as hf_transformer_model.pth")


if __name__ == "__main__":
    # Same batch/sequence parameters as before
    batch_params = {
        "gap": 10,
        "total_len": 100,
        "batch_size": 16,
    }

    # Hyperparameters for your HF transformer
    hyperparameters = {
        "dropout": 0.5,
        "d_model": 64,
        "nhead": 4,
        "num_layers": 2,
        "dim_feedforward": 256,
        "layer_norm_eps": 1e-5,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "epochs": 10,
    }

    train_loader, val_loader, _, _ = load_data(batch_params)
    train_model(train_loader, val_loader, hyperparameters)
