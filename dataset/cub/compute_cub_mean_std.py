import os, sys

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from PIL import Image

if __name__ == '__main__':
    IMAGE_DIR = r'/data2/bowenx/attention/pay_attention/dataset/cub/cropped'

    dataset = ImageFolder(
        IMAGE_DIR,
        transform=transforms.Compose([
            transforms.Resize((224, 224), interpolation=Image.LANCZOS),
            transforms.ToTensor()
        ]))

    dataloader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False
    )

    data = list()
    for img, _ in dataloader:
        data.append(img)

    data = torch.cat(data, dim=0)
    print(data.size())

    data = torch.transpose(data, 0, 1).contiguous()
    data = data.view(data.size(0), -1)

    # convert to double to get better precision
    data = data.double()

    mean = torch.mean(data, dim=1)
    std = torch.std(data, dim=1)
    print(mean, std)
