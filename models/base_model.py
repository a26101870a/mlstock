import torch
import torch.nn as nn

class BaseModel(nn.Module):

    def __init__(self):
        super(BaseModel, self).__init__()
        
    def save(self, path):
        torch.save(self.state_dict(), path)
        
    def load(self, path):
        self.load_state_dict(torch.load(path, weights_only=True))

    def get_model_name(self):
        return self.__class__.__name__