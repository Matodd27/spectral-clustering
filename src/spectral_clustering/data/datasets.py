from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_mnist_dataloaders(batch_size=64, root="data/raw", train_shuffle=True):
    transform = transforms.ToTensor()
    train = datasets.MNIST(root, train=True, download=True, transform=transform)
    test = datasets.MNIST(root, train=False, download=True, transform=transform)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=train_shuffle),
        DataLoader(test, batch_size=batch_size, shuffle=False),
    )
    
def get_fashionmnist_dataloaders(batch_size=64, root="data/raw", train_shuffle=True):
    transform = transforms.ToTensor()
    train = datasets.FashionMNIST(root, train=True, download=True, transform=transform)
    test = datasets.FashionMNIST(root, train=False, download=True, transform=transform)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=train_shuffle),
        DataLoader(test, batch_size=batch_size, shuffle=False),
    )