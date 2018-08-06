import os, sys
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


class AttentionNetwork(nn.Module):
    def __init__(self, cfg, num_class=10):
        
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

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        feature_maps = list()
        for i, layer in enumerate(self.backbone):
            x = layer(x)
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
                score = F.softmax(score)

                # weighted sum the feature map
                weighted_sum = torch.sum(torch.sum(score * feature_map, dim=3), dim=2)
                features.append(weighted_sum)

            x = self.att_fc(torch.cat(features, dim=1))

        return x
