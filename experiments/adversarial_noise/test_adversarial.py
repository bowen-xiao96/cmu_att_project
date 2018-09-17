import os, sys
import numpy as np
from collections import OrderedDict
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as A

from torchvision.models import vgg16

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

assert len(sys.argv) > 5
GPU_ID = int(sys.argv[1])
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)
mode = int(sys.argv[2])
weight_file = sys.argv[3]
num_iter = int(sys.argv[4])
eps = float(sys.argv[5])

from attack import *
sys.path.insert(0, '/data2/bowenx/attention/pay_attention')
from dataset.imagenet.get_imagenet_dataset import get_dataloader
from model.multiple_recurrent_l import *
from util.metric import accuracy

# the average drop of the two models with the adversarial noise
if mode == 0:
    model = vgg16(num_classes=1000, init_weights=False)
    model.load_state_dict(torch.load(weight_file))

    batch_size = 128

else:
    connections = (
        (13, 8, 256, 128, 2),
        (20, 15, 512, 256, 2)
    )
    model = MultipleRecurrentModel(network_cfg, connections, 5, 1000)
    _, _, state_dict = torch.load(weight_file)
    state_dict = OrderedDict([(k.replace('module.', ''), v) for k, v in state_dict.items()])
    model.load_state_dict(state_dict)
    del state_dict

    batch_size = 16

model = model.cuda()
model.eval()
for param in model.parameters():
    param.requires_grad = False

# set up dataset
_, test_loader = get_dataloader('/data2/simingy/data/Imagenet', batch_size, 8)


def convert_to_pil(array):
    array = array * std + mean
    array = (array * 255.0).astype(np.uint8)
    array = np.transpose(np.squeeze(array), (1, 2, 0))

    return Image.fromarray(array, mode='RGB')


original_score = list()
noise_score = list()
labels = list()
visualize = False

for i, (x, y) in enumerate(test_loader):
    original_img, noise, img_with_noise, score = FGSM_attack(model, x, y, num_iter, eps)

    original_score.append(score[0])
    noise_score.append(score[-1])
    labels.append(y.numpy())

    if visualize and i == 0:
        # draw visualization
        num_to_draw = min(10, batch_size)

        for j in range(num_to_draw):
            plt.clf()

            for k in range(num_iter):
                if k == 0:
                    old = original_img[j]
                else:
                    old = img_with_noise[k - 1][j]

                plt.subplot(num_iter, 3, k * 3 + 1)
                plt.axis('off')
                plt.imshow(convert_to_pil(old))

                n = noise[k][j]
                plt.subplot(num_iter, 3, k * 3 + 2)
                plt.axis('off')
                plt.imshow(convert_to_pil(n))

                new = img_with_noise[k][j]
                plt.subplot(num_iter, 3, k * 3 + 3)
                plt.axis('off')
                plt.imshow(convert_to_pil(new))

            plt.savefig('adversarial_%d' % j, dpi=1000)


# calculate accuracy
labels = torch.from_numpy(np.concatenate(labels, axis=0))
original_score = torch.from_numpy(np.concatenate(original_score, axis=0))
original_top_1, original_top_5 = accuracy(original_score, labels, topk=(1, 5))

noise_score = torch.from_numpy(np.concatenate(noise_score, axis=0))
noise_top_1, noise_top_5 = accuracy(noise_score, labels, topk=(1, 5))
print(original_top_1[0], original_top_5[0], noise_top_1[0], noise_top_5[0])
