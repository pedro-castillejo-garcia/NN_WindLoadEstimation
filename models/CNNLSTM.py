import torch
from torch import nn

class CNNLSTMModel(nn.Module):
    def __init__(self, input_dim, output_dim, seq_len, cnn_filters=32, lstm_hidden=32, dropout=0.1, dense_units=256):
        super(CNNLSTMModel, self).__init__()
        self.cnn = nn.Conv1d(in_channels=input_dim, out_channels=cnn_filters, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=1)  # No real effect but included for consistency
        self.lstm = nn.LSTM(input_size=cnn_filters, hidden_size=lstm_hidden, num_layers=2, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.dense1 = nn.Linear(lstm_hidden, dense_units)
        self.output = nn.Linear(dense_units, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Input: [batch_size, seq_len, input_dim] → need to permute for CNN
        x = x.permute(0, 2, 1)  # → [batch, input_dim, seq_len]
        x = self.cnn(x)         # → [batch, cnn_filters, seq_len]
        x = self.pool(x)        # → [batch, cnn_filters, seq_len]
        x = x.permute(0, 2, 1)  # → [batch, seq_len, cnn_filters] for LSTM

        x, _ = self.lstm(x)     # → [batch, seq_len, lstm_hidden]
        x = x[:, -1, :]         # Take the output from the last time step
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.relu(x)        # Apply ReLU to final output for consistency
        return x