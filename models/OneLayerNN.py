import torch
from torch import nn

# One Layer NN without dropout

class OneLayerNN(nn.Module):
    def __init__(self, input_dim, output_dim, seq_len=100):
        super(OneLayerNN, self).__init__()
        self.fc1 = nn.Linear(input_dim * seq_len, output_dim)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input
        x = self.fc1(x)
        x = self.activation(x)
        return x