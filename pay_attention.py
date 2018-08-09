import math
import numpy as np

import pickle
import math

import torch
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import torch.autograd as A


import torchvision
import torchvision.transforms as transforms

from utils import *

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms

cfg = [64, 64, 128, 128, 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M', 512, 'M', 512, 'M']

attention_layers = [13, 20, 27]



class VGG16Modified(nn.Module):
    def __init__(self, cfg, num_class):
        super(VGG16Modified, self).__init__()

        backbone = list()
        input_dim = 3
        for v in cfg:
            if v == 'M':
                backbone.append(nn.MaxPool2d(kernel_size=2, stride=2))
            elif v == 'D':
                backbone.append(nn.Dropout2d(p=0.4))
            else:
                backbone.append(nn.Conv2d(input_dim, v, kernel_size=3, padding=1))
                backbone.append(nn.ReLU(inplace=True))
                input_dim = v

        self.backbone = nn.Sequential(*backbone)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.6),
            nn.Linear(512, num_class)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class AttentionNetwork(nn.Module):

    def __init__(self, cfg, args, num_class=10):
        

    def __init__(self, cfg, attention_layers, num_class, dropout=0.5):

        super(AttentionNetwork, self).__init__()

        # cfg: network structure (see above)
        # attention_layers: index of layers to be used to calculate attention
        # num_class: number of classification categories
        self.attention_layers = getAttentionLayer(cfg)

        # set up backbone network
        self.backbone = nn.ModuleList()
        self.fclayers = nn.ModuleList()
        input_dim = 3
        layer_depth = getDepth(cfg)
        for j in range(1, layer_depth + 1):
            i = str(j)
            if getMaxPooling(i, cfg):
                self.backbone.append(nn.MaxPool2d(kernel_size=2, stride=2))
            if getConv(i, cfg):
                output_dim, kernel_size, padding = getConvSetting(i, cfg)
                self.backbone.append(nn.Conv2d(input_dim, output_dim, 
                                                kernel_size=kernel_size, 
                                                padding=padding))
                input_dim = output_dim

                if getBN(i, cfg):
                    self.backbone.append(nn.BatchNorm2d(output_dim))

                self.backbone.append(nn.ReLU(inplace=True))

            
            if getFC(i, cfg):
                output_dim, dropout_rate, output = getFCSetting(i, cfg)
                self.fclayers.append(nn.Linear(input_dim, output_dim))
                if output != 1:
                    self.fclayers.append(nn.ReLU(inplace=True))
                if dropout_rate != 0:
                    self.fclayers.append(nn.Dropout(dropout_rate))

                input_dim = output_dim      

        # Set up Attention Layers
        if self.attention_layers != []:

            attention_layers = self.attention_layers

            self.attention = nn.ModuleList()
            concat_dim = list()
            for v in attention_layers:
                m = nn.ModuleList()
                print(self.backbone[v-1])
                feature_dim = self.backbone[v - 1].out_channels
                if feature_dim != 512:
                    # we need an additional fc layer to project global feature to the lower-dimensional space
                    # if their dimensions do not match
                    m.append(nn.Linear(512, feature_dim))

                # attention scoring layer, implement as a 1x1 convolution
                m.append(nn.Conv2d(feature_dim, 1, kernel_size=1))
                self.attention.append(m)
                concat_dim.append(feature_dim)

            
            self.att_fc = nn.Linear(sum(concat_dim), num_class)
        
        # Initialize weights

        if args.init_weight == 'vgg':

            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                    m.bias.data.zero_()

        elif args.init_weight == 'xavier':

            for m in self.modules():
                xavier_init(m)


                input_dim = v

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
            nn.Dropout(p=dropout),
            nn.Linear(sum(concat_dim), num_class)
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

        for i, layer in enumerate(self.fclayers):
            x = layer(x)
        
        if len(self.attention_layers) != 0:
            features = list()
            for i, feature_map in enumerate(feature_maps):
                if len(self.attention[i]) == 2:
                    # project the global feature
                    new_x = self.attention[i][0](x)
                else:
                    new_x = x

                # attention score map (do 1x1 convolution on the addition of feature map and the global feature)
                score = self.attention[i][-1](feature_map + new_x.view(new_x.size(0), -1, 1, 1))
                score = F.softmax(score, dim=1)

                # weighted sum the feature map
                weighted_sum = torch.sum(torch.sum(score * feature_map, dim=3), dim=2)
                features.append(weighted_sum)

            x = self.att_fc(torch.cat(features, dim=1))

        return x

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
            score = F.softmax(score, dim=1)

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
        elif isinstance(m, nn.Linear):
            nn.init.normal(m.weight.data, mean=0.0, std=0.01)
            nn.init.constant(m.bias.data, 0.0)


def get_dataloader(cifar10_dir, batch_size, num_workers):
    # creating dataset and dataloader
    mean = np.array([0.49139968, 0.48215827, 0.44653124])
    std = np.array([0.24703233, 0.24348505, 0.26158768])
    normalize = transforms.Normalize(mean, std)

    # during training, only random horizontal flip is used for augmentation
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
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

