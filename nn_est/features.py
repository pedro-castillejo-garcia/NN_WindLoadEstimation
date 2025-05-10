import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

from hyperparameters import batch_parameters
from hyperparameters import batch_parameters

# Automatically find the absolute path of NN_WindLoadEstimation
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def create_sequences(data, targets, gap, total_len):
        X_seq, y_seq = [], []
        for i in range(len(data) - total_len + 1):
            X_seq.append(data[i + gap - 1 : i + total_len: gap])
            y_seq.append(targets[i + total_len - 1])
        return np.array(X_seq, dtype=np.float32), np.array(y_seq,dtype=np.float32)

def load_data(batch_parameters):
def load_data(batch_parameters):
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
    
    total_rows = sum(len(df) for df in datasets)

    
    total_rows = sum(len(df) for df in datasets)

    # Define features and targets
    features = ["Mx1", "Mx2", "Mx3", "My1", "My2", "My3", "Theta", "Vwx", "beta1", "beta2", "beta3", "omega_r"]
    targets = ["Mz1", "Mz2", "Mz3"]


    train_data = []
    val_data = []
    test_data = []

    for i, dataset in enumerate(datasets):

    for i, dataset in enumerate(datasets):
        n = len(dataset)
        
        # Calculate test set indices
        
        # Calculate test set indices
        test_start_idx = int((i % 5) * 0.2 * n)
        test_end_idx = test_start_idx + int(0.2 * n)

        # Extract test set
        test_split = dataset.iloc[test_start_idx:test_end_idx]
        test_data.append(test_split)

        # Extract test set
        test_split = dataset.iloc[test_start_idx:test_end_idx]
        test_data.append(test_split)
        
        # Remaining after removing test
        # Remaining after removing test
        remaining_data = dataset.drop(dataset.index[test_start_idx:test_end_idx])
    
        # Now split the remaining into train (75%) and val (25%)
        train_end_idx = int(0.75 * len(remaining_data))
        train_split = remaining_data.iloc[:train_end_idx]
        val_split = remaining_data.iloc[train_end_idx:]

        train_data.append(train_split)
        val_data.append(val_split)

    # Combine splits
    
        # Now split the remaining into train (75%) and val (25%)
        train_end_idx = int(0.75 * len(remaining_data))
        train_split = remaining_data.iloc[:train_end_idx]
        val_split = remaining_data.iloc[train_end_idx:]

        train_data.append(train_split)
        val_data.append(val_split)

    # Combine splits
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

def prepare_dataloaders(batch_parameters):

def prepare_dataloaders(batch_parameters):
    
    train_x, train_y, val_x, val_y, test_x, test_y, scaler_x, scaler_y = load_data(batch_parameters)    
    train_x, train_y, val_x, val_y, test_x, test_y, scaler_x, scaler_y = load_data(batch_parameters)    
    
    # Create sequences
    train_seq_x, train_seq_y = create_sequences(train_x, train_y, batch_parameters['gap'], batch_parameters['total_len'])
    val_seq_x, val_seq_y = create_sequences(val_x, val_y, batch_parameters['gap'], batch_parameters['total_len'])
    test_seq_x, test_seq_y = create_sequences(test_x, test_y, batch_parameters['gap'], batch_parameters['total_len'])
    train_seq_x, train_seq_y = create_sequences(train_x, train_y, batch_parameters['gap'], batch_parameters['total_len'])
    val_seq_x, val_seq_y = create_sequences(val_x, val_y, batch_parameters['gap'], batch_parameters['total_len'])
    test_seq_x, test_seq_y = create_sequences(test_x, test_y, batch_parameters['gap'], batch_parameters['total_len'])
    
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
    
    train_loader = DataLoader(train_dataset, batch_size=batch_parameters['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_parameters['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_parameters['batch_size'], shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_parameters['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_parameters['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_parameters['batch_size'], shuffle=False)
    
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
#For static rbfmodel
def prepare_flat_dataloaders(batch_parameters):
    """
    Loader til statiske/non-sekventielle modeller.
    Returnerer train_loader, val_loader, xgb_data inklusive scaler_y
    """
    (train_x, train_y,
     val_x,   val_y,
     _t_x, _t_y,        # test-split bruges ikke her
     _scaler_x, scaler_y) = load_data(batch_parameters)   # <-- tag scaler_y med

    # ---------- PyTorch ----------
    X_tr = torch.tensor(train_x, dtype=torch.float32)
    y_tr = torch.tensor(train_y, dtype=torch.float32)
    X_va = torch.tensor(val_x,   dtype=torch.float32)
    y_va = torch.tensor(val_y,   dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_tr, y_tr),
        batch_size=batch_parameters["batch_size"],
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_va, y_va),
        batch_size=batch_parameters["batch_size"],
        shuffle=False
    )

    # ---------- dict til RBF / XGBoost m.m. ----------
    xgb_data = {
        "X_train":  train_x,
        "y_train":  train_y,
        "X_val":    val_x,
        "y_val":    val_y,
        "scaler_y": scaler_y        # nu findes variablen!
    }
    return train_loader, val_loader, xgb_data


if __name__ == "__main__":
        
    train_loader, val_loader, test_loader, xgb_data, scaler_x, scaler_y = prepare_dataloaders(batch_parameters)

        
    train_loader, val_loader, test_loader, xgb_data, scaler_x, scaler_y = prepare_dataloaders(batch_parameters)