import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

# Automatically find the absolute path of NN_WindLoadEstimation
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def create_sequences(data, targets, gap, total_len):
        X_seq, y_seq = [], []
        for i in range(len(data) - total_len + 1):
            X_seq.append(data[i + gap - 1 : i + total_len: gap])
            y_seq.append(targets[i + total_len - 1])
        return np.array(X_seq), np.array(y_seq)

def load_data(batch_params):
    # Define file paths using absolute paths
    file_paths = [
        os.path.join(project_root, "data/raw/wind_speed_11_n.csv"),
        os.path.join(project_root, "data/raw/wind_speed_13_n.csv"),
        os.path.join(project_root, "data/raw/wind_speed_15_n.csv"),
        os.path.join(project_root, "data/raw/wind_speed_17_n.csv"),
        os.path.join(project_root, "data/raw/wind_speed_19_n.csv")
    ]
    
    # Load datasets
    datasets = [pd.read_csv(file) for file in file_paths]
    
    # Define features and targets
    features = ["Mx1", "Mx2", "Mx3", "My1", "My2", "My3", "Theta", "Vwx", "beta1", "beta2", "beta3", "omega_r"]
    targets = ["Mz1", "Mz2", "Mz3"]
    
    train_data = []
    val_data = []
    test_data = []
    
    i = 0
    
    
    # Split datasets
    for dataset in datasets:
        n = len(dataset)
        test_start_idx = int((i % 5) * 0.2 * n)
        test_end_idx = test_start_idx + int(0.2 * n)
        
        # Extract test data first
        test_data.append(dataset.iloc[test_start_idx:test_end_idx])
        remaining_data = dataset.drop(dataset.index[test_start_idx:test_end_idx])
        
        # Split remaining data into training and validation
        train_end_idx = int(0.8 * len(remaining_data))
        train_data.append(remaining_data.iloc[:train_end_idx])
        val_data.append(remaining_data.iloc[train_end_idx:])
        
        i += 1
    
    # Concatenate datasets
    train_data = pd.concat(train_data, ignore_index=True)
    val_data = pd.concat(val_data, ignore_index=True)
    test_data = pd.concat(test_data, ignore_index=True)
    
    # Initialize scalers
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    # Scale the features and targets
    train_x = scaler_x.fit_transform(train_data[features].values)
    train_y = scaler_y.fit_transform(train_data[targets].values)
    
    val_x = scaler_x.transform(val_data[features].values)
    val_y = scaler_y.transform(val_data[targets].values)
    
    test_x = scaler_x.transform(test_data[features].values)
    test_y = scaler_y.transform(test_data[targets].values)
    
    return train_x, train_y, val_x, val_y, test_x, test_y, scaler_x, scaler_y
 
def prepare_dataloaders(batch_params):
    
    train_x, train_y, val_x, val_y, test_x, test_y, scaler_x, scaler_y = load_data(batch_params)    
    
    # Create sequences
    train_seq_x, train_seq_y = create_sequences(train_x, train_y, batch_params['gap'], batch_params['total_len'])
    val_seq_x, val_seq_y = create_sequences(val_x, val_y, batch_params['gap'], batch_params['total_len'])
    test_seq_x, test_seq_y = create_sequences(test_x, test_y, batch_params['gap'], batch_params['total_len'])
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(train_seq_x, dtype=torch.float32)
    y_train_tensor = torch.tensor(train_seq_y, dtype=torch.float32)
    X_val_tensor = torch.tensor(val_seq_x, dtype=torch.float32)
    y_val_tensor = torch.tensor(val_seq_y, dtype=torch.float32)
    X_test_tensor = torch.tensor(test_seq_x, dtype=torch.float32)
    y_test_tensor = torch.tensor(test_seq_y, dtype=torch.float32)
    
    # Data for Transformer (sequence data)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_params['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_params['batch_size'], shuffle=False)
    
    # Data for XGBoost (flattened data)
    X_train_flat = train_seq_x.reshape(train_seq_x.shape[0], -1)
    y_train_flat = train_seq_y
    X_val_flat = val_seq_x.reshape(val_seq_x.shape[0], -1)
    y_val_flat = val_seq_y
    X_test_flat = test_seq_x.reshape(test_seq_x.shape[0], -1)
    y_test_flat = test_seq_y

    xgb_data = {
        'X_train': X_train_flat,
        'y_train': y_train_flat,
        'X_val': X_val_flat,
        'y_val': y_val_flat,
        'X_test': X_test_flat,
        'y_test': y_test_flat
    }

    return train_loader, val_loader, test_loader, xgb_data, scaler_x, scaler_y

if __name__ == "__main__":
    batch_params = {
        "gap": 10,
        "total_len": 100,
        "batch_size": 16,
    }
    
    train_loader, val_loader, test_loader, xgb_data, scaler_x, scaler_y = prepare_dataloaders(batch_params)
