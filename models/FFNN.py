import torch
from torch import nn

class FFNNModel(nn.Module):
    def __init__(self, input_dim, output_dim, seq_len=20, dropout=0.3):
        super(FFNNModel, self).__init__()
        self.fc1 = nn.Linear(input_dim * seq_len, 512)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x