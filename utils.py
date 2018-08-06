import os, sys
import numpy as np

import cPickle

import json
import copy
import argparse
import torch
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as A

import torchvision
import torchvision.transforms as transforms


def xavier_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform(m.weight)
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

def postprocess_config(cfg):
    cfg = copy.deepcopy(cfg)
    for k in cfg.get("network_list", ["decode", "encode"]):
        if k in cfg:
            ks = cfg[k].keys()
            for _k in ks:
                if _k.isdigit():
                    cfg[k][int(_k)] = cfg[k].pop(_k)
    return cfg

def adjust_learning_rate(optimizer, epoch, init_lr=0.01, lr_decay=0.5, lr_freq=30):
    """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
    lr = init_lr * (lr_decay ** (epoch // lr_freq))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def getConv(i, cfg, layer_name='layers'):
    ret_val = False
    if 'conv' in cfg[layer_name][i]:
        ret_val = True

    return ret_val

def getFC(i, cfg, layer_name='layers'):
    ret_val = False
    if 'fc' in cfg[layer_name][i]:
        ret_val = True

    return ret_val

def getFCSetting(i, cfg, layer_name='layers'):
    output_dim = cfg[layer_name][i]['fc']['num_features']
    dropout = 0
    whether_output = 0
    if 'dropout' in cfg[layer_name][i]['fc']:
        dropout = cfg[layer_name][i]['fc']['dropout']
    if 'output' in cfg[layer_name][i]['fc']:
        whether_output = cfg[layer_name][i]['fc']['output']

    return output_dim, dropout, whether_output

def getBN(i, cfg, layer_name='layers'):
    ret_val = False
    if 'bn' in cfg[layer_name][i]:
        if cfg[layer_name][i]['bn'] == 1:
            ret_val = True

    return ret_val

def getMaxPooling(i, cfg, layer_name='layers'):
    ret_val = False
    if 'pool' in cfg[layer_name][i]:
        if 'max' in cfg[layer_name][i]['pool']['type']:
            ret_val = True

    return ret_val

def getConvSetting(i, cfg, layer_name='layers'):
    output_dim = cfg[layer_name][i]['conv']['num_filters']
    kernel_size = cfg[layer_name][i]['conv']['filter_size']
    padding = 1
    return output_dim, kernel_size, padding

def getAttentionLayer(cfg):
    ret_val = []
    if 'attention_layers' in cfg:
        print("attention layers:", cfg['attention_layers'])
        for i in range(len(cfg['attention_layers'])):
                ret_val.append(cfg['attention_layers'][i])

    return ret_val

def getDepth(cfg):
    return cfg['depth']


