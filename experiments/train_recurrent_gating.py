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
from model.recurrent_gating import *
from dataset.cifar.get_cifar10_dataset import get_dataloader
from util.Padam import Padam

assert len(sys.argv) > 3
GPU_ID = int(sys.argv[1])
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)
unroll_count = int(sys.argv[2])
TAG = sys.argv[3]

print('Unrolling time step: %d' % unroll_count)

model = RecurrentGatingModel(network_cfg, unroll_count, 10)
initialize_vgg(model)
model.cuda()

train_loader, test_loader = get_dataloader(
    '/data2/bowenx/attention/pay_attention/dataset/cifar',
    256,
    1
)

criterion = nn.CrossEntropyLoss()
criterion.cuda()
init_lr = 0.0001

optimizer = Padam(
    model.parameters(),
    lr=init_lr,
    weight_decay=1e-3
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
    save_every=5,
    max_keep=50
)
