import torch
from torch import nn

class LSTMModel(nn.Module):
    def __init__(self, input_dim, output_dim, lstm_hidden=64, num_layers=2, dropout=0.3, dense_units=256):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=lstm_hidden, num_layers=num_layers,
                            batch_first=True, dropout=dropout)
        self.dense1 = nn.Linear(lstm_hidden, dense_units)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.output = nn.Linear(dense_units, output_dim)

    def forward(self, x):
        # Input shape: [batch, seq_len, input_dim]
        x, _ = self.lstm(x)  # x: [batch, seq_len, hidden]
        x = x[:, -1, :]  # Take the output from the last time step
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.relu(x)  # Final activation for consistency
        return x