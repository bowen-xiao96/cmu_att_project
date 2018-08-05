import os, sys
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as A

from torch.utils.data import dataloader, dataset
from torchvision.transforms import transforms

cfg = [64, 64, 128, 128, 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M', 512, 'M', 512, 'M']
attention_layers = [13, 20, 27]


class AttentionNetwork(nn.Module):
    def __init__(self, cfg, attention_layers, num_class):
        super(AttentionNetwork, self).__init__()

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
        self.fc2 = nn.Linear(sum(concat_dim), num_class)

    def forward(self, x):
        feature_maps = list()
        for i, layer in enumerate(self.backbone):
            x = layer(x)
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
            score = F.softmax(score, dim=1)

            # weighted sum the feature map
            weighted_sum = torch.sum(torch.sum(score * feature_map, dim=3), dim=2)
            features.append(weighted_sum)

        return self.fc2(torch.cat(features, dim=1))
