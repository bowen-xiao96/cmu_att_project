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
from pay_attention import cfg, AttentionNetwork, initialize_vgg, get_dataloader

assert len(sys.argv) > 1
GPU_ID = int(sys.argv[1])
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)

model = AttentionNetwork(cfg, (20,), 10)
initialize_vgg(model)
model.load_state_dict(torch.load(r''))  # TODO

train_loader, test_loader = get_dataloader(
    '/data2/bowenx/attention/cifar',
    256,
    4
)

criterion = nn.CrossEntropyLoss()
init_lr = 0.01

optimizer = optim.SGD(
    model.parameters(),
    lr=init_lr,
    momentum=0.9,
    weight_decay=5e-4
)


def lr_sched(optimizer, epoch):
    lr = init_lr * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


Trainer.start(
    model=model,
    optimizer=optimizer,
    train_dataloader=train_loader,
    test_dataloader=test_loader,
    criterion=criterion,
    max_epoch=50,
    lr_sched=lr_sched,
    display_freq=50,
    output_dir='modified_vgg',
    save_every=5,
    max_keep=20
)