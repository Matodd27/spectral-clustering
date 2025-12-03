from .base import BaseAE
import torch.nn as nn

class FFAutoencoder(BaseAE):
    def __init__(self, latent_dim, input_dim):
        super().__init__(latent_dim=latent_dim, input_dim=input_dim)
        self.flatten = nn.Flatten()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(),
            nn.Linear(512, 128), nn.ReLU(),
            nn.LayerNorm(128)
        )

        self.to_latent = nn.Linear(128, latent_dim)
        self.latent_bn = nn.BatchNorm1d(latent_dim, affine=False)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 512), nn.ReLU(),
            nn.Linear(512, input_dim), nn.Sigmoid()
        )

    def encode(self, x):
        x = self.flatten(x)
        h = self.encoder(x)
        z = self.to_latent(h)
        z = self.latent_bn(z)
        return z
    
    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        xhat = self.decode(z)
        return xhat