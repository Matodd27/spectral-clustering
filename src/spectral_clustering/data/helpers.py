import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Tuple


def dataloader_to_numpy(
    loader: DataLoader,
    flatten: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []

    for x, y in loader:
        x = x.cpu()
        y = y.cpu()

        if flatten:
            x = x.view(x.size(0), -1)

        xs.append(x)
        ys.append(y)

    X = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0)

    return X.numpy(), y.numpy()
