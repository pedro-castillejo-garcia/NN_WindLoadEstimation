import torch
from torch import nn
import math
from mamba_ssm import Mamba

class MambaModel(nn.Module):
    def __init__(self, input_dim, output_dim, seq_len, d_model=512, num_layers=6, dropout=0.1):
        super(MambaModel, self).__init__()
        
        self.input_embedding = nn.Linear(input_dim, d_model)

        # Stack multiple Mamba layers
        self.mamba_layers = nn.Sequential(
            *[Mamba(d_model=d_model) for _ in range(num_layers)]
        )

        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, output_dim)

    def forward(self, x):
        x = self.input_embedding(x)  # Encode input to d_model dimensions
        #x = self.positional_encoding(x)
        x = self.mamba_layers(x)  # Apply Mamba layers
        x = x[:, -1, :]  # Take the output of the last time step
        x = self.dropout(x)
        x = self.linear1(x)  # Decode to target dimensions
        return x