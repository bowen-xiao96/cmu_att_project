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

model_urls = {
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
}

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
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
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
        num_workers=num_workers,
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

def getReLU(i, cfg, layer_name='layers'):
    ret_val = 0
    if 'relu' in cfg[layer_name][i]:
        ret_val = cfg[layer_name][i]['relu']

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
    if 'padding' in cfg[layer_name][i]['conv']:
        padding = cfg[layer_name][i]['conv']['padding']

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

def getAtt_Recurrent_v2(cfg):
    att_recurrent_layer = []
    if 'att_r_v2' in cfg:
        for i, number in enumerate(cfg['att_r_v2']):
            att_recurrent_layer.append(int(number))

    return att_recurrent_layer

def getAtt_Recurrent_v2_Setting(i, cfg):
    v = str(i)
    unroll_count = cfg['att_r_v2'][v]['unroll_count']
    start_layer = cfg['att_r_v2'][v]['back']
    return unroll_count, start_layer

def getGate_Recurrent(cfg):
    gate_recurrent_layer = []
    if 'gate_r' in cfg:
        for i, number in enumerate(cfg['gate_r']):
            gate_recurrent_layer.append(int(number))

    return gate_recurrent_layer

def getGate_Recurrent_v2(cfg):
    gate_recurrent_layer = []
    if 'gate_r_v2' in cfg:
        for i, number in enumerate(cfg['gate_r_v2']):
            gate_recurrent_layer.append(int(number))

    return gate_recurrent_layer

def getGate_Recurrent_Setting(i, cfg):
    v = str(i)
    unroll_count = cfg['gate_r'][v]['unroll_count']
    start_layer = cfg['gate_r'][v]['back']
    spatial_reduce = (cfg['gate_r'][v]['spatial_reduce'] == 1)
    gate_filter_size = cfg['gate_r'][v]['gate_filter_size']
    return unroll_count, start_layer, spatial_reduce, gate_filter_size

def getGate_Recurrent_v2_Setting(i, cfg):
    v = str(i)
    unroll_count = cfg['gate_r_v2'][v]['unroll_count']
    start_layer = cfg['gate_r_v2'][v]['back']
    gate_filter_size = cfg['gate_r_v2'][v]['gate_filter_size']
    gate_dropout = 0
    if 'dropout' in cfg['gate_r_v2'][v]:
        gate_dropout = cfg['gate_r_v2'][v]['dropout']

    return unroll_count, start_layer, gate_filter_size, gate_dropout

def get_Intermediate_loss(cfg):
    extra_loss = 0
    if 'loss_params' in cfg:
        extra_loss = 1
    return extra_loss

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
    ret_val = 0
    for k, v in pre_dict.items():
        if 'module' in k:
            ret_val = 1
    
    return ret_val


class PredictionModule(nn.Module):
    def __init__(self, input_dim, attention_map_dim, spatial_reduce, dropout=0.5):
        super(PredictionModule, self).__init__()
        self.spatial_reduce = spatial_reduce

        self.stream_a = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Conv2d(input_dim, attention_map_dim, kernel_size=1),
            nn.Dropout(p=dropout),
            nn.Softplus(beta=1.0)
        )

        self.stream_d = nn.Conv2d(input_dim, attention_map_dim, kernel_size=1)

    def forward(self, x):
        stream_a = self.stream_a(x) + 0.1

        # do spatial normalization on all channels
        spatial_sum = torch.sum(torch.sum(stream_a, dim=3, keepdim=True), dim=2, keepdim=True)
        stream_a = stream_a / spatial_sum

        stream_d = self.stream_d(x)

        # mix together and classify
        output = stream_a * stream_d

        # if spatial_reduce is on, we output the mean value of the spatial map
        if self.spatial_reduce:
            output = torch.mean(torch.mean(output, dim=3), dim=2)
        
        return output

weights = list()
gamma = 0
alpha = 0

def get_loss_params(cfg):
    global weights, alpha, gamma
    
    if 'gamma' in cfg['loss_params']:
        gamma = cfg['loss_params']['gamma']
    if 'alpha' in cfg['loss_params']:
        alpha = cfg['loss_params']['alpha']
    if 'unroll_count' in cfg['loss_params']:
        unroll_count = cfg['loss_params']['unroll_count']

    for i in range(unroll_count):
        if i == 0:
            weights.append(gamma)
        else:
            weights.append(weights[-1] * gamma)

    # like (0.9**5, 0.9**4, 0.9**3, 0.9**2, 0.9)
    weights = torch.FloatTensor(list(reversed(weights)))
    # normalize to sum=1.0
    weights /= torch.sum(weights)
    weights = A.Variable(weights.cuda())

def gate_criterion(pred, y):

    intermediate_pred, final_pred = pred

    # batch_size * unroll_count * num_class
    old_size = intermediate_pred.size()
    batch_size, time_step = old_size[:2]
    # flatten into (-1, num_class)
    new_size = torch.Size([batch_size * time_step] + list(old_size[2:]))
    intermediate_pred = intermediate_pred.view(new_size)

    loss_1 = F.cross_entropy(
            intermediate_pred,
            y.repeat(time_step, 1).transpose(0, 1).contiguous().view(-1), reduce=False
            )
    # average over batches
    loss_1 = torch.sum(loss_1 * weights.repeat(batch_size)) / batch_size

    loss_2 = F.cross_entropy(final_pred, y)

    return alpha * loss_1 + loss_2

def gate_v2_criterion(pred, y):

    unroll_number = len(pred)
    batch_size = pred[0][0]
    loss_list = []

    final_loss = 0.0
    for i in range(unroll_number):
        loss_temp = F.cross_entropy(pred[i], y)
        final_loss += loss_temp * weights[i]
    
    return final_loss
    
def add_gaussian_noise(image, mean=0.0, stddev=0.5):
 
    img_data = image.data.cpu().numpy()
    noise = np.random.normal(mean, stddev, size=(img_data.shape))
    mask = np.random.choice([0, 1], size=(img_data.shape[2], img_data.shape[3]), p=[5./6, 1./6])
    mask = np.expand_dims(mask, axis=0)
    mask = np.expand_dims(mask, axis=0)
    mask = np.tile(mask, [noise.shape[0], noise.shape[1], 1, 1])

    noise = noise * mask
    #image = image * Variable(torch.from_numpy((1 - mask)).float().cuda())
    noise = Variable(torch.from_numpy(noise).float().cuda())

    return image + noise


