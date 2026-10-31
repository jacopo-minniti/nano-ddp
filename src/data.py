"""Dataset selection for the classification trainer."""

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


def mnist(data_dir, download=False):
    return MNIST(data_dir, train=True, transform=ToTensor(), download=download)


# A dataset must return (input tensor, class label) for each index.
DATASETS = {"mnist": mnist}


def load_dataset(name, data_dir, download=False):
    try:
        factory = DATASETS[name]
    except KeyError as exc:
        raise ValueError(f"unknown dataset {name!r}; available: {', '.join(DATASETS)}") from exc
    return factory(data_dir, download=download)
