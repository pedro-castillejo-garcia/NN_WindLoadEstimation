import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

from hyperparameters import batch_parameters
from features import create_sequences

# Automatically find the absolute path of NN_WindLoadEstimation
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def load_data_new_test_data(batch_parameters):
    
    csv_folder = os.path.join(project_root, "data/raw/Systol Files/Fc")
    file_paths = [os.path.join(csv_folder, f) for f in os.listdir(csv_folder) if f.endswith(".csv")]

    datasets = [pd.read_csv(f) for f in file_paths]
    
    features = ["Mx1", "Mx2", "Mx3", "My1", "My2", "My3", "Theta", "Vwx", "beta1", "beta2", "beta3", "omega_r"]
    targets = ["Mz1", "Mz2", "Mz3"]

    all_test_data = pd.concat(datasets, ignore_index=True)

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    test_x = scaler_x.fit_transform(all_test_data[features].values)
    test_y = scaler_y.fit_transform(all_test_data[targets].values)

    return file_paths, test_x, test_y, scaler_x, scaler_y

def prepare_dataloaders_new_test_data(batch_parameters):
    
    file_paths, test_x, test_y, scaler_x, scaler_y = load_data_new_test_data(batch_parameters)

    # Create sequences
    test_seq_x, test_seq_y = create_sequences(test_x, test_y, batch_parameters['gap'], batch_parameters['total_len'])

    # Convert to PyTorch tensors
    X_test_tensor = torch.tensor(test_seq_x, dtype=torch.float32)
    y_test_tensor = torch.tensor(test_seq_y, dtype=torch.float32)

    # Dataset and DataLoader
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_parameters['batch_size'], shuffle=False)

    return test_loader, scaler_x, scaler_y

if __name__ == "__main__":
    test_loader, scaler_x, scaler_y = prepare_dataloaders_new_test_data(batch_parameters) 
    