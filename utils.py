import os, sys
import numpy as np
import math
import cPickle
from PIL import Image
from matplotlib import cm

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
from torchvision.datasets import CIFAR10
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def get_dataloader(cifar10_dir, batch_size, num_workers):
    # creating dataset and dataloader
    mean = np.array([0.49139968, 0.48215827, 0.44653124])
    std = np.array([0.24703233, 0.24348505, 0.26158768])
    normalize = transforms.Normalize(mean, std)

    # during training, only random horizontal flip is used for augmentation
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        normalize
    ])

    train_dataset = CIFAR10(
        cifar10_dir,
        train=True,
        transform=train_transform,
        download=True,
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
        transform=test_transform,
        download=True,
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

def get_Imagenetloader(imn_dir, batch_size, num_workers):
    # creating dataset and dataloader
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    normalize = transforms.Normalize(mean, std)

    # during training, only random horizontal flip is used for augmentation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    train_dataset = ImageFolder(
        os.path.join(imn_dir, 'train'),
        transform=train_transform
    )
    

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        normalize
    ])

    val_dataset = ImageFolder(
        os.path.join(imn_dir, 'val'),
        transform=val_transform
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    print("Load Data Done!")
    return train_loader, val_loader

def xavier_init(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform(m.weight)
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

def vgg_init(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            nn.init.normal(m.weight.data, mean=0.0, std=math.sqrt(2.0 / n))
            nn.init.constant(m.bias.data, 0.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant(m.weight.data, 1.0)
            nn.init.constant(m.bias.data, 0.0)

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
    input_ = 0
    if 'dropout' in cfg[layer_name][i]['fc']:
        dropout = cfg[layer_name][i]['fc']['dropout']
    if 'output' in cfg[layer_name][i]['fc']:
        whether_output = cfg[layer_name][i]['fc']['output']
    if 'input' in cfg[layer_name][i]['fc']:
        input_ = cfg[layer_name][i]['fc']['input']

    return input_, output_dim, dropout, whether_output

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
        for i in range(len(cfg['attention_layers'])):
                ret_val.append(cfg['attention_layers'][i])

    return ret_val

def getDepth(cfg):
    return cfg['depth']

def getAtt_Recurrent(cfg):
    att_recurrent_layer = []
    
    if 'att_r' in cfg:
        for i, number in enumerate(cfg['att_r']):
            att_recurrent_layer.append(int(number))

    return att_recurrent_layer

def getAtt_RecurrentSetting(i, cfg):
    v = str(i)
    unroll_count = cfg['att_r'][v]['unroll_count']

    return unroll_count

def get_jet():
    colormap_int = np.zeros((256, 3), np.uint8)
    colormap_float = np.zeros((256, 3), np.float)
                
    for i in range(0, 256, 1):
        colormap_int[i, 0] = np.uint8(np.round(cm.jet(i)[0] * 255.0))
        colormap_int[i, 1] = np.uint8(np.round(cm.jet(i)[1] * 255.0))
        colormap_int[i, 2] = np.uint8(np.round(cm.jet(i)[2] * 255.0))
                                                    
    return colormap_int

def gray2color(gray_array, color_map):
    rows, cols = gray_array.shape
    color_array = np.zeros((rows, cols, 3), dtype=np.uint8)
                
    for i in range(0, rows):
        for j in range(0, cols):
            color_array[i, j] = color_map[gray_array[i, j]]
                                                    
    color_image = Image.fromarray(color_array)
    return color_image

def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index)


def load_parallel(pre_dict):
    for k, v in pre_dict.items():
        if k[:7] == 'module.':
            return 1
        else:
            return 0
        
