import torch.nn as nn
from .base_model import BaseModel

class FullyConnected(BaseModel):
    # input_shape is (batch_size, sequence_length, n_features)
    def __init__(self, input_shape):
        super(FullyConnected, self).__init__()

        seq_len = input_shape[1]
        n_features = input_shape[2]

        self.fc = nn.Sequential(
            nn.Linear(seq_len*n_features, 1)
        )
    
    def forward(self, x):
        # Flatten the input
        x = x.view(x.shape[0], -1)  
        x = self.fc(x)
        return x