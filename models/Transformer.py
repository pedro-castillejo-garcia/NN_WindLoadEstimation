import torch
from torch import nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        positions = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        self.encoding[:, 0::2] = torch.sin(positions * div_term)
        self.encoding[:, 1::2] = torch.cos(positions * div_term)
        self.encoding = self.encoding.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', self.encoding)

    def forward(self, x):
      # Add positional encoding to input
      x = x + self.pe[:, :x.size(1), :]
      return x

# Define Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, seq_len, d_model=512, nhead=8, num_layers=6, dim_feedforward = 2048, dropout=0.1, layer_norm_eps = 1e-5):
        super(TransformerModel, self).__init__()
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=seq_len)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead, num_encoder_layers= num_layers, num_decoder_layers = num_layers, dim_feedforward = dim_feedforward, dropout = dropout, layer_norm_eps = layer_norm_eps, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, output_dim)

    def forward(self, x):
        x = self.input_embedding(x)  # Encode input to d_model dimensions
        x = self.positional_encoding(x)
        x = self.transformer(x, x)  # Apply transformer
        x = x[:, -1, :]  # Take the output of the last time step
        x = self.dropout(x)
        x = self.linear1(x)  # Decode to target dimensions
        return x