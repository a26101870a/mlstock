import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel

class CNNLSTM(BaseModel):
    # input_shape is (batch_size, window_length, feature_dim)
    def __init__(self, input_shape, lstm_hidden_dim=64, lstm_layers=2):
        super(CNNLSTM, self).__init__()
        self.feature_dim = input_shape[2]
        
        # CNN layers
        self.conv1 = nn.Conv1d(
            in_channels=self.feature_dim,
            out_channels=self.feature_dim,
            kernel_size=5
        )
        
        self.conv2 = nn.Conv1d(
            in_channels=self.feature_dim,
            out_channels=self.feature_dim * 2,
            kernel_size=5
        )
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.feature_dim * 2,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_dim, lstm_hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(lstm_hidden_dim * 2, 1)
        )
        
    def forward(self, x):
        # Transpose for CNN: (batch_size, feature_dim, window_length)
        x = x.transpose(1, 2)
        
        # CNN layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # Transpose back for LSTM: (batch_size, window_length, feature_dim)
        x = x.transpose(1, 2)
        
        # Initialize LSTM hidden states
        h_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h_0, c_0))
        out = out[:, -1, :] # Take the last output of LSTM
        out = self.fc(out)
        
        return out