import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import math


file_paths = [
    "./data/raw/wind_speed_11_n.csv",
    "./data/raw/wind_speed_13_n.csv",
    "./data/raw/wind_speed_15_n.csv",
    "./data/raw/wind_speed_17_n.csv",
    "./data/raw/wind_speed_19_n.csv"
]

# Load datasets
datasets = [pd.read_csv(file) for file in file_paths]

# Define features and targets
features = ["Mx1", "Mx2", "Mx3", "My1", "My2", "My3", "Theta", "Vwx", 
            "beta1", "beta2", "beta3", "omega_r"]
targets = ["Mz1", "Mz2", "Mz3"]

train_data_x = []
train_data_y = []
val_data_x = []
val_data_y = []
test_data = {}

Hyperparameters['seq_len'] = Hyperparameters['total_len']//Hyperparameters['gap']

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

def create_sequences(data, targets, gap, total_len):
    X_seq, y_seq = [], []
    for i in range(len(data) - total_len + 1):
        X_seq.append(data[i + gap - 1 : i + total_len: gap])
        y_seq.append(targets[i + total_len - 1])
    return np.array(X_seq), np.array(y_seq)

# Split datasets
i = 0
for dataset in datasets:

    n = len(dataset)
    train_end_idx = int(0.6 * n)
    val_end_idx = int(0.8 * n)
    
    # Sequential splits
    train_segment = dataset.iloc[:train_end_idx]
    val_segment = dataset.iloc[train_end_idx:val_end_idx]
    test_segment = dataset.iloc[val_end_idx:]

    if i==0: #fit the scaler only on the first training set
      train_segment_x = scaler_x.fit_transform(train_segment[features].values)
      train_segment_y = scaler_y.fit_transform(train_segment[targets].values)

    else:
      train_segment_x = scaler_x.transform(train_segment[features].values)
      train_segment_y = scaler_y.transform(train_segment[targets].values)

    val_segment_x = scaler_x.transform(val_segment[features].values)
    val_segment_y = scaler_y.transform(val_segment[targets].values)

    train_seq_x,train_seq_y = create_sequences(train_segment_x, train_segment_y, Hyperparameters['gap'], Hyperparameters['total_len'])
    val_seq_x,val_seq_y = create_sequences(val_segment_x, val_segment_y, Hyperparameters['gap'], Hyperparameters['total_len'])

    # Append to lists
    train_data_x.append(train_seq_x)
    train_data_y.append(train_seq_y)
    val_data_x.append(val_seq_x)
    val_data_y.append(val_seq_y)
    test_data[i] = test_segment
    i += 1

train_data_x = np.concatenate(train_data_x, axis=0)
train_data_y = np.concatenate(train_data_y, axis=0)
val_data_x = np.concatenate(val_data_x, axis=0)
val_data_y = np.concatenate(val_data_y, axis=0)