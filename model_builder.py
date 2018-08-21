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

        # attention_layers: index of layers to be used to calculate attention
        # num_class: number of classification categories
        self.attention_layers = getAttentionLayer(cfg)


        # Attention Recurrent model
        self.att_r_layers = getAtt_Recurrent(cfg)
        self.att_channel = args.att_channel
        #self.att_r_type = args.att_r_type

        # Attention Recurrent V2 model
        self.att_r_v2_layers = getAtt_Recurrent_v2(cfg)
        self.start_layers = []

        # the path to save feature map, attention map, origin image
        self.save_att_map = args.save_att_map
        save_data_path = os.path.join('/data2/simingy/model_data/', args.expId)
        os.system('mkdir -p %s' % save_data_path)
        self.save_data = save_data_path
        
        # set up backbone network
        self.backbone = nn.ModuleList()

        # set up fc layers
        self.fclayers = nn.ModuleList()
 
        # Start to build the model!
        input_dim = 3
        layer_depth = getDepth(cfg)

        for j in range(1, layer_depth + 1):
            i = str(j)
            # whether max pooling
            if getMaxPooling(i, cfg):
                self.backbone.append(nn.MaxPool2d(kernel_size=2, stride=2))

            # whether do convolution
            if getConv(i, cfg):
                output_dim, kernel_size, padding = getConvSetting(i, cfg)
                self.backbone.append(nn.Conv2d(input_dim, output_dim, 
                                                kernel_size=kernel_size, 
                                                padding=padding))
                input_dim = output_dim

                if getBN(i, cfg):
                    self.backbone.append(nn.BatchNorm2d(output_dim))

                self.backbone.append(nn.ReLU(inplace=True))

            # whether FC layer
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


        # Set up Attention Recurrent Layers
        if self.att_r_layers != []:
            self.att_recurrent_b = nn.ModuleList()
            self.att_recurrent_f = nn.ModuleList()

            for v in self.att_r_layers:
                m = nn.ModuleList()
                self.att_r_unroll_count = getAtt_RecurrentSetting(v, cfg)
                feature_dim = self.backbone[v - 1].out_channels
                if feature_dim != 512:
                    m.append(nn.Linear(512, feature_dim))
                m.append(nn.Conv2d(feature_dim, 1, kernel_size=1))
                self.att_recurrent_b.append(m)
                
                match_dim = nn.Sequential(
                        nn.Conv2d(feature_dim + self.att_channel, feature_dim, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True)
                        )
                self.att_recurrent_f.append(match_dim)

        # Set up Attention Recurrent V2 Layers
        if self.att_r_v2_layers != []:
            self.att_recurrent_b = nn.ModuleList()
            self.att_recurrent_f = nn.ModuleList()

            for v in self.att_r_v2_layers:
                m = nn.ModuleList()
                self.att_r_unroll_account, start_layer = getAtt_Recurrent_v2_Setting(v, cfg)
                self.start_layers.append(start_layer)

                feature_dim = self.backbone[v-1].out_channels
                m.append(nn.ConvTranspose2d(feature_dim, feature_dim / 4, kernel_size=3, stride=2, padding=1))
                m.append(nn.ConvTranspose2d(feature_dim / 4, feature_dim / 8, kernel_size=3, stride=2, padding=1))
                m.append(nn.ConvTranspose2d(feature_dim / 8, feature_dim / 16, kernel_size=2, stride=2, padding=1))
                m.append(nn.Conv2d(feature_dim / 16, 1, kernel_size=1))
                self.att_recurrent_b.append(m)
                
                start_feature_dim = self.backbone[start_layer-1].out_channels
                match_dim = nn.Sequential(
                        nn.Conv2d(start_feature_dim + 1, start_feature_dim, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True)
                        )

                self.att_recurrent_f.append(match_dim)
               



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
        
        # whether print familiarity effect numbers
        self.print_fe = print_fe 
        feature_maps = list()
        end_feature_maps = list()

        # Save origin images
        if self.save_att_map == 1:
            mean = np.array([0.49139968, 0.48215827, 0.44653124])
            std = np.array([0.24703233, 0.24348505, 0.26158768])

            input_img = x[0].data.cpu().numpy()
            input_img = np.transpose(input_img, (1, 2, 0))
            input_img = ((input_img * std) + mean) * 255.0
            input_img = (input_img).astype(np.uint8)
            #print("input_img:", input_img.shape)
            save_img = Image.fromarray(input_img)
            save_img.save(os.path.join(self.save_data, 'input-0.png'))
        
        break_point = []

        # Backbone part 
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            # after relu layer
            if i in self.attention_layers:
                feature_maps.append(x)
            if i in self.att_r_layers:
                feature_maps.append(x)
            if i in self.start_layers:
                feature_maps.append(x)
            if i in self.att_r_v2_layers:
                break_point.append(i + 1)
                break

        # Attention Recurrent V2
        if self.att_r_v2_layers != []:
            for i, feature_map in enumerate(feature_maps):
                recurrent_buf = list()
                recurrent_buf.append(feature_map)
                for j in range(self.att_r_unroll_account):
                    prev = recurrent_buf[-1]

                    for k, layer in enumerate(self.att_recurrent_b[i]):
                         x = layer(x)
                    
                    old_shape = x.size()

                    score = F.softmax(x.view(old_shape[0], -1), dim=1).view(old_shape)

                    x = self.att_recurrent_f[i](torch.cat([score, prev], dim=1))

                    recurrent_buf.append(x)

                    for k in range(self.start_layers[i] + 1, break_point[i]):
                        x = self.backbone[k](x)

                for j in range(break_point[i], len(self.backbone)):
                    x = self.backbone[j](x)


        x = x.view(x.size(0), -1)

        for i, layer in enumerate(self.fclayers):
            # Whether add attention recurrent
            if self.att_r_layers != []:
                if i == len(self.fclayers) - 1:
                    break
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
                
                '''Print out some information'''
                if self.save_att_map == 1:
                    cm = get_jet()
                    att_map = nn.UpsamplingBilinear2d(size=(32, 32))(score)
                    att_map = att_map.data.cpu().numpy()
                    att_map = np.reshape(np.transpose(att_map[0], (1, 2, 0)), (32, 32))
                    att_map = (att_map * 255.0).astype(np.int8)
                    new_map = gray2color(att_map, cm)
                    new_map.save(os.path.join(self.save_data, 'att-'+str(i)+'.png'))

                if self.print_fe == 1:
                    print("index:", i)
                    print("score map max number:", torch.max(score))
                    print("fire neurons:", torch.sum(score > 1e-3))
                   

                # weighted sum the feature map
                weighted_sum = torch.sum(torch.sum(score * feature_map, dim=3), dim=2)
                features.append(weighted_sum)

            x = self.att_fc(torch.cat(features, dim=1))

        if self.att_r_layers != []:
            
            for i, feature_map in enumerate(feature_maps):
                recurrent_buf = list()
                recurrent_buf.append(feature_map)
                for j in range(self.att_r_unroll_count):
                    
                    prev = recurrent_buf[-1]
                    if len(self.att_recurrent_b[i]) == 2:
                        x = self.att_recurrent_b[i][0](x)

                    score = self.att_recurrent_b[i][-1](prev + x.view(x.size(0), -1, 1, 1))
                    old_shape = score.size()
                
                    score = F.softmax(score.view(old_shape[0], -1), dim=1).view(old_shape)

                    #score = tile(score, dim=1, n_tile=self.att_channel) 
                    score = score.expand(old_shape[0], self.att_channel, old_shape[2], old_shape[3])

                    x = self.att_recurrent_f[i](torch.cat([score, prev], dim=1))

                    recurrent_buf.append(x)
                    
                    for k in range(self.att_r_layers[i] + 1, len(self.backbone)):
                        x = self.backbone[k](x)
                    x = x.view(x.size(0), -1)
                    for k in range(len(self.fclayers) - 1):
                        x = self.fclayers[k](x)
            
            x = self.fclayers[-1](x)

        return x

