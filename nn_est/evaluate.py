import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_squared_error
from features import load_data
from models.Transformer import TransformerModel

# Get the absolute path of the project's root directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))

# Add project root to sys.path
import sys
sys.path.append(project_root)

# Load test data
batch_params = {
    "gap": 10,
    "total_len": 100,
    "batch_size": 16,
}

hyperparameters = {
    "dropout": 0.5,
    "d_model": 64,
    "nhead": 4,
    "num_layers": 2,
    "dim_feedforward": 256,
    "layer_norm_eps": 1e-5,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
}

_, _, test_data_x, test_data_y = load_data(batch_params)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the latest trained model
checkpoints_dir = os.path.join(project_root, "checkpoints")
latest_model = sorted(os.listdir(checkpoints_dir))[-1]
model_path = os.path.join(checkpoints_dir, latest_model)

model = TransformerModel(
    input_dim=test_data_x.shape[-1],
    output_dim=test_data_y.shape[-1],
    seq_len=batch_params['total_len'] // batch_params['gap'],
    d_model=hyperparameters['d_model'],
    nhead=hyperparameters['nhead'],
    num_layers=hyperparameters['num_layers'],
    dim_feedforward=hyperparameters['dim_feedforward'],
    dropout=hyperparameters['dropout'],
    layer_norm_eps=hyperparameters['layer_norm_eps']
)

model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)