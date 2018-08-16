import math
import numpy as np

import pickle
import math
#import matplotlib.pyplot as plt
#from matplotlib import colors
#from matplotlib import cm
from PIL import Image

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


class AttentionNetwork(nn.Module):
    def __init__(self, cfg, args, num_class=10):

        super(AttentionNetwork, self).__init__()

        # cfg: network structure (see above)
        # attention_layers: index of layers to be used to calculate attention
        # num_class: number of classification categories
        self.attention_layers = getAttentionLayer(cfg)
        
        self.save_att_map = args.save_att_map
        self.print_fe = args.print_fe
        save_data_path = os.path.join('/data2/simingy/model_data/', args.expId)
        os.system('mkdir -p %s' % save_data_path)

        self.save_data = save_data_path
        
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
                input_, output_dim, dropout_rate, output = getFCSetting(i, cfg)
                if input_ != 0:
                    input_dim = input_
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

    def forward(self, x, print_fe=0):
        self.print_fe = print_fe
        feature_maps = list()

        # Save origin images
        if self.save_att_map == 1:
            mean = np.array([0.49139968, 0.48215827, 0.44653124])
            std = np.array([0.24703233, 0.24348505, 0.26158768])

            input_img = x[0].data.cpu().numpy()
            input_img = np.transpose(input_img, (1, 2, 0))
            input_img = ((input_img * std) + mean) * 255.0
            input_img = (input_img).astype(np.uint8)
            print("input_img:", input_img.shape)
            save_img = Image.fromarray(input_img)
            save_img.save(os.path.join(self.save_data, 'input-0.png'))
        
        
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            # after relu layer
            if i in self.attention_layers:
                feature_maps.append(x)

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
                old_shape = score.size()
  

                score = F.softmax(score.view(old_shape[0], -1), dim=1).view(old_shape)
                #print(score)
                if self.save_att_map == 1:
                    cm = get_jet()
                    print("score map:", score.shape)
                    att_map = nn.UpsamplingBilinear2d(size=(32, 32))(score)
                    att_map = att_map.data.cpu().numpy()
                    att_map = np.reshape(np.transpose(att_map[0], (1, 2, 0)), (32, 32))
                    att_map = (att_map * 255.0).astype(np.int8)
                    new_map = gray2color(att_map, cm)
                    new_map.save(os.path.join(self.save_data, 'att-'+str(i)+'.png'))

                if self.print_fe == 1:
                    print("************index***********", i)
                    print("score map max number:", torch.max(score))
                    #print("score shape:", score.shape)
                    print("fire neurons:", torch.sum(score > 1e-3))
                    #print("score map sum:", torch.sum(score))  

                # weighted sum the feature map
                weighted_sum = torch.sum(torch.sum(score * feature_map, dim=3), dim=2)
                features.append(weighted_sum)

            x = self.att_fc(torch.cat(features, dim=1))

        return x

