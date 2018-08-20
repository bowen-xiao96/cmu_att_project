import os, sys

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, '/data2/bowenx/attention/pay_attention')

from util import Trainer
from util.model_tools import initialize_vgg
from model.pay_attention import *
from dataset.imagenet.get_imagenet_dataset import get_dataloader

assert len(sys.argv) > 1
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
TAG = sys.argv[2]

model = AttentionNetwork(cfg, attention_layers, 1000)
initialize_vgg(model)
model.cuda()

train_loader, test_loader = get_dataloader(
    '/data2/simingy/data/Imagenet',
    64,
    8
)

criterion = nn.CrossEntropyLoss()
criterion.cuda()
init_lr = 0.001

optimizer = optim.Adam(
    model.parameters(),
    lr=init_lr,
    weight_decay=1e-4
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
    max_epoch=180,
    lr_sched=lr_sched,
    display_freq=50,
    output_dir=TAG,
    save_every=20,
    max_keep=20
)
