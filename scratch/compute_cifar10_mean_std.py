import os, sys
import numpy as np

import torch

from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms

cifar10_dir = r'cifar'
dataset = CIFAR10(
    cifar10_dir,
    train=True,
    transform=transforms.ToTensor()
)

dataloader = DataLoader(
    dataset,
    batch_size=256,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
    drop_last=False
)

# calculate mean and std of cifar10 training set
data = list()
for x, _ in dataloader:
    # x: torch tensor
    # n x c x h x w
    data.append(x)

data = torch.cat(data, dim=0)
data = data.numpy()
print(data.shape)

data = np.transpose(data, (1, 0, 2, 3))
data = np.reshape(data, (data.shape[0], -1))
print(data.shape)
mean = np.mean(data, axis=1)
std = np.std(data, axis=1)

print(mean, std)
