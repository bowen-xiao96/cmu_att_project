# train the vgg attention model from scratch on ImageNet
# utilize multiple GPUS

import os, sys
import math
import numpy as np
import pickle

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as A
import torch.optim as optim

import Trainer
from imagenet_dataset import *

assert len(sys.argv) > 1
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
TAG = sys.argv[1]

# define model
cfg = [64, 64, 128, 128, 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M', 512, 'M', 512, 'M', 512]
attention_layers = [13, 20, 27]


# only add an average pooling on the whole network
class ImageNetAttentionModel(nn.Module):
    def __init__(self, cfg, attention_layers, num_class, dropout=0.5):
        super(ImageNetAttentionModel, self).__init__()

        # cfg: network structure (see above)
        # attention_layers: index of layers to be used to calculate attention
        # num_class: number of classification categories
        self.attention_layers = attention_layers

        # set up backbone network
        self.backbone = nn.ModuleList()
        input_dim = 3
        for v in cfg:
            if v == 'M':
                self.backbone.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                self.backbone.append(nn.Conv2d(input_dim, v, kernel_size=3, padding=1))
                self.backbone.append(nn.ReLU(inplace=True))
                input_dim = v

        # the final global average pooling layer
        self.backbone.append(nn.AvgPool2d(kernel_size=7, stride=7))

        # set up attention layers
        self.attention = nn.ModuleList()
        concat_dim = list()
        for v in attention_layers:
            m = nn.ModuleList()
            feature_dim = self.backbone[v - 1].out_channels
            if feature_dim != 512:
                # we need an additional fc layer to project global feature to the lower-dimensional space
                # if their dimensions do not match
                m.append(nn.Linear(512, feature_dim))

            # attention scoring layer, implement as a 1x1 convolution
            m.append(nn.Conv2d(feature_dim, 1, kernel_size=1))

            self.attention.append(m)
            concat_dim.append(feature_dim)

        self.fc1 = nn.Linear(512, 512)
        self.classifier = nn.Sequential(
            nn.Linear(sum(concat_dim), 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(1024, num_class)
        )

    def forward(self, x):
        feature_maps = list()
        for i, layer in enumerate(self.backbone):
            x = layer(x)

            # after relu layer
            if i in self.attention_layers:
                feature_maps.append(x)

        # global feature
        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        features = list()
        for i, feature_map in enumerate(feature_maps):
            if len(self.attention[i]) == 2:
                # project the global feature
                new_x = self.attention[i][0](x)
            else:
                new_x = x

            # attention score map (do 1x1 convolution on the addition of feature map and the global feature)
            score = self.attention[i][-1](feature_map + new_x.view(new_x.size(0), -1, 1, 1))
            old_shape = score.size()
            score = F.softmax(
                score.view(old_shape[0], -1), dim=1
            ).view(old_shape)

            # weighted sum the feature map
            weighted_sum = torch.sum(torch.sum(score * feature_map, dim=3), dim=2)
            features.append(weighted_sum)

        return self.classifier(torch.cat(features, dim=1))


def initialize_vgg(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            nn.init.normal(m.weight.data, mean=0.0, std=math.sqrt(2.0 / n))
            nn.init.constant(m.bias.data, 0.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant(m.weight.data, 1.0)
            nn.init.constant(m.bias.data, 0.0)

        # leave fc layers default initialized


model = ImageNetAttentionModel(cfg, attention_layers, 1000, dropout=0.5)
initialize_vgg(model)
model = nn.DataParallel(model, device_ids=(0, 1, 2, 3)).cuda()

train_loader, test_loader = get_imagenet_dataset(
    imagenet_path,
    64,
    8
)

criterion = nn.CrossEntropyLoss()
criterion.cuda()
init_lr = 0.01

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=init_lr,
    momentum=0.9,
    weight_decay=1e-4
)


def lr_sched(optimizer, epoch):
    lr = init_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


Trainer.start(
    model=model,
    optimizer=optimizer,
    train_dataloader=train_loader,
    test_dataloader=test_loader,
    criterion=criterion,
    max_epoch=61,
    lr_sched=lr_sched,
    display_freq=50,
    output_dir=TAG,
    save_every=1,
    max_keep=50
)
