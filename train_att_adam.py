import os, sys
import math
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as A
import torch.optim as optim

import torch.backends.cudnn as cudnn

cudnn.benchmark = True

import Trainer
from pay_attention import cfg, AttentionNetwork, attention_layers, initialize_vgg, get_dataloader

assert len(sys.argv) > 2
GPU_ID = int(sys.argv[1])
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)
TAG = sys.argv[2]

model = AttentionNetwork(cfg, attention_layers, 10)
initialize_vgg(model)

train_loader, test_loader = get_dataloader(
    '/data2/bowenx/attention/cifar',
    256,
    4
)

criterion = nn.CrossEntropyLoss()
init_lr = 0.001

optimizer = optim.Adam(
    model.parameters(),
    lr=init_lr,
    weight_decay=5e-4
)


def lr_sched(optimizer, epoch):
    lr = init_lr * (0.5 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


Trainer.start(
    model=model,
    optimizer=optimizer,
    train_dataloader=train_loader,
    test_dataloader=test_loader,
    criterion=criterion,
    max_epoch=120,
    lr_sched=None,
    display_freq=50,
    output_dir=TAG,
    save_every=5,
    max_keep=20
)
