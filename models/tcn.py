import torch.nn as nn
from .base_model import BaseModel

class Chomp1d(nn.Module):
    """Chomps off extra padding at the end after convolution."""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]

"""Residual block for TCN."""
class TCNResidualBlock(nn.Module):    
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TCNResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, 
                              stride=stride, padding=padding, dilation=dilation)
        
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(
            self.conv1, 
            self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

"""Temporal Convolutional Network"""
class TCN(BaseModel):
    def __init__(self, input_shape, num_channels=[128, 256, 128], kernel_size=2, dilation_layer=3, dropout=0.2):
        """
        Args:
            num_channels: List of channel sizes for each TCN Residual Block layer.
            dilation_layer: Number of dilation layers in each TCN Residual Block.
        """
        super(TCN, self).__init__()
        feature_dim = input_shape[2]

        # Build mutilple TCNResidualBlock as TCN layer.
        layers = []
        for i in range(len(num_channels)):
            
            in_channels = feature_dim if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            for j in range(dilation_layer):
                dilation_size = 2 ** j
                layers.append(TCNResidualBlock(
                    in_channels, out_channels, kernel_size,
                    stride=1, dilation=dilation_size,
                    padding=(kernel_size-1) * dilation_size,
                    dropout=dropout
                ))

                in_channels = out_channels
                
        self.network = nn.Sequential(*layers)

        self.fc = nn.Linear(num_channels[-1], 1)
        
    def forward(self, x):
        x = x.transpose(1, 2)  # (batch_size, window_length, feature_dim) to (batch_size, feature_dim, window_length)
        out = self.network(x)
        out = out[:, :, -1] # Get the output for the last time step.
        out = self.fc(out)
        return out 