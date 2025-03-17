import torch
from torch import nn


class FFNNModel(nn.Module):
    def __init__(self, input_dim, output_dim, seq_len=20, dropout=0.3):
        super(FFNNModel, self).__init__()
        self.fc1 = nn.Linear(input_dim * seq_len, 1024)
        self.bn1 = nn.BatchNorm1d(1024)  # Batch Normalization
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)

        self.fc4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(dropout)

        self.fc5 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input
        x = self.relu1(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = self.relu2(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.relu3(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        x = self.relu4(self.bn4(self.fc4(x)))
        x = self.dropout4(x)
        x = self.fc5(x)
        return x