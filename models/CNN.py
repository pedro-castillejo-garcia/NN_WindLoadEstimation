import torch
import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self, in_channels, seq_length, num_outputs):
        super(CNNModel, self).__init__()
        # 1. konvolutionslag: bruger kernel_size=3 med padding=1 for at bevare længden
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        # Max pooling for at reducere længden med faktor 2
        self.pool = nn.MaxPool1d(kernel_size=2)
        # 2. konvolutionslag:
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        # Efter to pooling-lag bliver sekvenslængden reduceret med en faktor på 4.
        self.fc = nn.Linear(64 * (seq_length // 4), num_outputs)
        
    def forward(self, x):
        # x har form (batch, seq_length, features)
        # Vi permuterer for at få (batch, channels, seq_length)
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc(x)
        return x

