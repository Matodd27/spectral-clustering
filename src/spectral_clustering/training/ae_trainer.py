from typing import Callable, Optional, Any

import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn


class AETrainer:
    def __init__(
        self,
        loss_fn: Optional[Callable[[torch.Tensor, Any], torch.Tensor]] = None,
        lr: float = 1e-3,
        device: Optional[str] = None,
    ):
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.loss_fn = loss_fn or (lambda x, out: nn.functional.mse_loss(out, x))
        self.lr = lr

    def fit(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        num_epochs: int = 20,
        optimiser: Optional[torch.optim.Optimizer] = None,
    ):
        model.to(self.device)
        model.train()

        if optimiser is None:
            optimiser = torch.optim.Adam(model.parameters(), lr=self.lr)

        for epoch in range(num_epochs):
            total_loss = 0.0
            total_samples = 0

            for batch_idx, (x, *_) in enumerate(train_loader):
                x = x.to(self.device)
                optimiser.zero_grad()
                outputs = model(x)                 # AE: x_hat; VAE: (x_hat, mu, logvar), etc.
                loss = self.loss_fn(x, outputs)    # loss_fn decides how to unpack outputs

                loss.backward()
                optimiser.step()

                batch_size = x.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

            avg_loss = total_loss / total_samples
            print(f"Epoch {epoch + 1}/{num_epochs} - avg loss: {avg_loss:.4f}")

        return model

    @torch.no_grad()
    def encode_dataset(
        self,
        model: nn.Module,
        data,
        batch_size: int = 256,
    ):
        model.to(self.device)
        model.eval()

        if isinstance(data, DataLoader):
            loader = data
        else:
            loader = DataLoader(data, batch_size=batch_size, shuffle=False)
        
        latents = []

        for x, *rest in loader:
            x = x.to(self.device)
            z = model.encode(x)[0]
            if isinstance(z, tuple):
                z = z[0]
            else:
                z = z[0]
            latents.append(z.cpu())

        return torch.cat(latents, dim=0)

    @torch.no_grad()
    def decode_latent(
        self,
        model: nn.Module,
        data,
        batch_size: int=256
    ):
        model.to(self.device)
        model.eval()

        if isinstance(data, DataLoader):
            loader = data
        else:
            loader = DataLoader(data, batch_size=batch_size, shuffle=False)

        decoded = []

        for batch in loader:
            if isinstance(batch, (tuple, list)):
                z = batch[0]
            else:
                z = batch

            z = z.to(self.device)
            x = model.decode(z)
            decoded.append(x.cpu())

        return torch.cat(decoded, dim=0)