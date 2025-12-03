import torch.nn as nn
from abc import abstractmethod

class BaseAE(nn.Module):
    def __init__(self, latent_dim: int, input_dim: int=784):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim

    @abstractmethod
    def encode(self):
        pass

    @abstractmethod
    def decode(self):
        pass

    @abstractmethod
    def forward(self):
        pass