import numpy as np

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms


def get_dataloader(cifar10_dir, batch_size, num_workers):
    # creating dataset and dataloader
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2470, 0.2435, 0.2616])
    normalize = transforms.Normalize(mean, std)

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        # random crop
        # which is very essential
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        normalize
    ])

    train_dataset = CIFAR10(
        cifar10_dir,
        train=True,
        transform=train_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    test_dataset = CIFAR10(
        cifar10_dir,
        train=False,
        transform=test_transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False
    )

    return train_loader, test_loader
