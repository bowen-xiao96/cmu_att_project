import os, sys
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as A

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

from torchvision.models import vgg16
from model.multiple_recurrent_l import *
from util.metric import accuracy


def get_dataloader(root_dir, mode, noise, batch_size, num_workers, sigma=None):
    def add_noise_and_to_tensor(img):
        # assume img is a PIL image
        img = np.array(img).astype(np.float32)
        if noise:
            # img ranges from 0 to 255 (original np.uint8)
            n = np.random.normal(0.0, sigma, img.shape)
            img += n
            img = np.clip(img, 0.0, 255.0)

        if mode == 1:
            # need to swap from RGB to BGR
            img = img[..., ::-1]
        else:
            # need to scale between 0 and 1
            img /= 255.0

        # convert from HWC to CHW
        img = np.transpose(img, (2, 0, 1)).copy()
        return torch.from_numpy(img)

    if mode == 1:
        mean = np.array([103.939, 116.779, 123.68])
        std = np.array([1.0, 1.0, 1.0])
    else:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

    normalize = transforms.Normalize(mean, std)
    transform = transforms.Compose([
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


if __name__ == '__main__':
    # fix the seed of the random generator
    np.random.seed(0)

    assert len(sys.argv) > 3
    # mode=0: original vgg pytorch model, mode=1: original vgg caffe model
    # mode=2: recurrent gating model
    # occlusion=0: no noise, occlusion=1: add noise
    # if occlusion=1, we need an additional sigma parameter
    model_file = sys.argv[1]
    mode = int(sys.argv[2])
    noise = int(sys.argv[3])
    if noise == 1:
        sigma = float(sys.argv[4])
    else:
        sigma = None

    # load model
    if mode == 2:
        # recurrent gating model
        # such hyperparams are fixed
        connections = (
            (13, 8, 256, 128, 2),
            (20, 15, 512, 256, 2)
        )
        model = MultipleRecurrentModel(network_cfg, connections, 5, 1000)
        model = nn.DataParallel(model)
        _, _, state_dict = torch.load(model_file)

    else:
        # ordinary vgg model
        model = vgg16(num_classes=1000, init_weights=False)
        state_dict = torch.load(model_file)
        if mode == 1:
            # the caffe model
            m = {'classifier.1.weight': 'classifier.0.weight', 'classifier.1.bias': 'classifier.0.bias',
                 'classifier.4.weight': 'classifier.3.weight', 'classifier.4.bias': 'classifier.3.bias'}
            state_dict = OrderedDict([(m[k] if k in m else k, v) for k, v in state_dict.items()])

    # print(state_dict.keys())
    model.load_state_dict(state_dict)
    model.eval()
    model.cuda()
    del state_dict

    # load dataset
    imagenet_dir = '/data2/simingy/data/Imagenet'
    test_loader = get_dataloader(imagenet_dir, mode, noise, 256, 8, sigma)

    pred = list()
    gt = list()

    for x, y in test_loader:
        x = A.Variable(x.cuda(), volatile=True)
        y = A.Variable(y.cuda(), volatile=True)

        output = model(x)
        pred.append(output.data.cpu())
        gt.append(y.data.cpu())

    pred = torch.cat(pred, dim=0)
    gt = torch.cat(gt, dim=0)

    prec1, prec5 = accuracy(pred, gt, topk=(1, 5))
    print(prec1[0], prec5[0])
