import os, sys
import torch

from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import DataLoader
from torchvision import transforms

if __name__ == '__main__':
    assert len(sys.argv) > 1
    flag = sys.argv[1]

    if flag == 'cifar10':
        cifar10_dir = r'.'
        dataset = CIFAR10(
            cifar10_dir,
            train=True,
            transform=transforms.ToTensor()
        )

    elif flag == 'cifar100':
        cifar100_dir = '.'
        dataset = CIFAR100(
            cifar100_dir,
            train=True,
            transform=transforms.ToTensor()
        )

    else:
        raise NotImplementedError

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
    print(data.size())

    data = torch.transpose(data, 0, 1).contiguous()
    data = data.view(data.size(0), -1)

    # convert to double to get better precision
    data = data.double()

    mean = torch.mean(data, dim=1)
    std = torch.std(data, dim=1)
    print(mean, std)
