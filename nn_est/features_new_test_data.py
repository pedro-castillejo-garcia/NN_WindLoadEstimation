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

def load_data_new_test_data(batch_parameters, max_files=None):
    csv_folder = os.path.join(project_root, "data/raw/Systol Files/Fc")
    file_paths = [os.path.join(csv_folder, f) for f in os.listdir(csv_folder) if f.endswith(".csv")]

    if max_files:
        file_paths = file_paths[:max_files]
        
    datasets = []
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        df["source_file"] = os.path.basename(file_path)
        datasets.append(df)
    
    features = ["Mx1", "Mx2", "Mx3", "My1", "My2", "My3", "Theta", "Vwx", "beta1", "beta2", "beta3", "omega_r"]
    targets = ["Mz1", "Mz2", "Mz3"]
    time = ["t"]

    all_test_data = pd.concat(datasets, ignore_index=True)

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    test_x = scaler_x.fit_transform(all_test_data[features].values)
    test_y = scaler_y.fit_transform(all_test_data[targets].values)
    test_t = all_test_data[time].values

    return all_test_data, test_x, test_y, test_t, scaler_x, scaler_y

def prepare_dataloaders_new_test_data(batch_parameters, max_files=None):
    
    all_test_data, test_x, test_y, test_t, scaler_x, scaler_y = load_data_new_test_data(batch_parameters, max_files)

    # Create sequences
    test_seq_x, test_seq_y = create_sequences(test_x, test_y, batch_parameters['gap'], batch_parameters['total_len'])
    dummy_targets = np.zeros(len(test_t))  # Create a dummy array of zeros with the same length as `test_t` to later create the sequences with the time
    test_seq_t, _ = create_sequences(test_t, dummy_targets, batch_parameters['gap'], batch_parameters['total_len'])

    # Verify that the length of source_files_seq matches the number of sequences
    print(f"Size of test_seq_x: {len(test_seq_x)}")
    
    # Convert to PyTorch tensors
    X_test_tensor = torch.tensor(test_seq_x, dtype=torch.float32)
    y_test_tensor = torch.tensor(test_seq_y, dtype=torch.float32)
    t_test_tensor = torch.tensor(test_seq_t, dtype=torch.float32)

    source_files = all_test_data["source_file"].values
    source_seq, _ = create_sequences(source_files.reshape(-1, 1), dummy_targets, batch_parameters['gap'], batch_parameters['total_len'])
    source_tensor = np.array([s[0][0] for s in source_seq])  # one source filename per sequence

    # Convert to PyTorch tensors
    source_tensor = np.array([s for s in source_tensor])  # list of filenames (strings)

    # Dataset and DataLoader
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor, t_test_tensor, torch.tensor(range(len(source_tensor))))
    test_loader = DataLoader(test_dataset, batch_size=batch_parameters['batch_size'], shuffle=False)
    
    return test_loader, scaler_x, scaler_y, source_tensor


if __name__ == "__main__":
    test_loader, scaler_x, scaler_y, source_tensor = prepare_dataloaders_new_test_data(batch_parameters) 
    