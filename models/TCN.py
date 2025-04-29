import torch
from torch import nn
from pytorch_tcn import TCN

class TCNModel(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        num_channels=[32, 64, 64],
        kernel_size=6,
        kernel_initializer = 'kaiming_uniform',
        dropout=0.2,
        causal=True,
        use_skip_connections=False,
        use_norm='weight_norm',
        activation='relu'
    ):
        super(TCNModel, self).__init__()

        self.tcn = TCN(
            num_inputs=input_dim,
            num_channels=num_channels,
            kernel_size=kernel_size,
            kernel_initializer = kernel_initializer,
            dropout=dropout,
            causal=causal,
            use_skip_connections=use_skip_connections,
            use_norm=use_norm,
            activation=activation,
            input_shape='NCL'
        )

        self.linear = nn.Linear(num_channels[-1], output_dim)
        self.output_activation = nn.ReLU()

    def forward(self, x):
        # x shape: [batch_size, input_channels, sequence_length]
        x = self.tcn(x)
        x = self.linear(x[:, :, -1])  # Take output at last time step
        x = self.output_activation(x)
        return x