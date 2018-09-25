import os, sys
import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms


def get_dataloader(root_dir, mode, batch_size, num_workers, sigma=0.0):
    def add_noise_and_to_tensor(img):
        # assume `img` is a PIL image
        # `img` ranges from 0 to 255 (np.uint8)
        img = np.array(img).astype(np.float32)

        if sigma > 0.0:
            # add Gaussian noise
            n = np.random.normal(0.0, sigma, img.shape)
            img += n
            img = np.clip(img, 0.0, 255.0)

        if mode == 'caffe':
            # need to swap from RGB to BGR
            img = img[..., ::-1]
        else:
            # need to scale between 0 and 1
            img /= 255.0

        # convert from HWC to CHW
        img = np.transpose(img, (2, 0, 1)).copy()
        return torch.from_numpy(img)

    if mode == 'caffe':
        mean = np.array([103.939, 116.779, 123.68])
        std = np.array([1.0, 1.0, 1.0])
    else:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

    normalize = transforms.Normalize(mean, std)
    transform = transforms.Compose([
        # standard ImageNet testing
        transforms.Resize(256, interpolation=Image.LANCZOS),
        transforms.CenterCrop(224),
        transforms.Lambda(add_noise_and_to_tensor),
        normalize
    ])

    # create dataset and dataloader
    dataset = ImageFolder(os.path.join(root_dir, 'val'), transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    return dataloader
