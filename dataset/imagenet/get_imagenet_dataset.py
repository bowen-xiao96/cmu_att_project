import os, sys
import numpy as np

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

from PIL import Image


def get_dataloader(root_dir, batch_size, num_workers):
    # get unpreprocessed imagenet dataset

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    normalize = transforms.Normalize(mean, std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, interpolation=Image.LANCZOS),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    train_dataset = ImageFolder(
        os.path.join(root_dir, 'train'),
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
        transforms.Resize(256, interpolation=Image.LANCZOS),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    test_dataset = ImageFolder(
        os.path.join(root_dir, 'val'),
        transform=test_transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False
    )

    return train_loader, test_loader
