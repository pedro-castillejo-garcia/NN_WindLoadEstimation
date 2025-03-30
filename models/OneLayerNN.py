import torch
from torch import nn

# Without activation function mse: 324,98 
# class OneLayerNN(nn.Module):
#     def __init__(self, input_dim, output_dim, seq_len=100, dropout=0.3):
#         super(OneLayerNN, self).__init__()
#         self.fc1 = nn.Linear(input_dim * seq_len, output_dim)
#         self.dropout = nn.Dropout(dropout)
    
#     def forward(self, x):
#         x = x.view(x.size(0), -1)  # Flatten input
#         x = self.fc1(x)
#         x = self.dropout(x)
#         return x

# With ELU activation function mse: 219,05 and Tanh activation function I got mse: , 
# with LeakyReLU I got mse 228.76, with ReLU I got mse: 207,95, 189,95 with lenght 500; mse: 182,98 with length 1000

#Train OneLayerNN without droupout
class OneLayerNN(nn.Module):
    def __init__(self, input_dim, output_dim, seq_len=100):
        super(OneLayerNN, self).__init__()
        self.fc1 = nn.Linear(input_dim * seq_len, output_dim)
        self.activation = nn.ReLU()
        #self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input
        x = self.fc1(x)
        x = self.activation(x)
        #x = self.dropout(x)
        return x