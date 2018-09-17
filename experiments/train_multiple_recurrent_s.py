import os, sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, '/data2/bowenx/attention/pay_attention')

from util import Trainer
from util.model_tools import initialize_vgg
from model.multiple_recurrent_s import *
from dataset.cifar.get_cifar10_dataset import get_dataloader

assert len(sys.argv) > 3
GPU_ID = int(sys.argv[1])
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)

unroll_count = int(sys.argv[2])
print('Unrolling time step: %d' % unroll_count)
TAG = sys.argv[3]

connections = (
    (6, 3, 128, 64, 2),
)
model = MultipleRecurrentModel(network_cfg, connections, unroll_count, 10)
initialize_vgg(model)
model.cuda()

train_loader, test_loader = get_dataloader(
    '/data2/bowenx/attention/pay_attention/dataset/cifar',
    128,
    1
)

gamma = 0.5
weights = list()
for i in range(unroll_count):
    if i == 0:
        weights.append(gamma)
    else:
        weights.append(weights[-1] * gamma)

weights = torch.FloatTensor(list(reversed(weights)))
weights /= torch.sum(weights)
print('Loss weights for different time steps:')
print(weights)
weights = A.Variable(weights.cuda(), requires_grad=False)


def criterion(pred, y):
    # batch_size * unroll_count * num_class
    old_size = pred.size()
    batch_size, time_step = old_size[:2]
    # flatten into (-1, num_class)
    new_size = torch.Size([batch_size * time_step] + list(old_size[2:]))
    intermediate_pred = pred.view(new_size)

    loss_1 = F.cross_entropy(
        intermediate_pred,
        y.repeat(time_step, 1).transpose(0, 1).contiguous().view(-1),
        reduce=False
    )
    # average over batches
    loss = torch.sum(loss_1 * weights.repeat(batch_size)) / batch_size
    return loss


init_lr = 0.0001


def lr_sched(optimizer, epoch):
    lr = init_lr * (0.1 ** (epoch // 100))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


optimizer = optim.Adam(
    model.parameters(),
    lr=init_lr,
    weight_decay=1e-4
)

Trainer.start(
    model=model,
    optimizer=optimizer,
    train_dataloader=train_loader,
    test_dataloader=test_loader,
    criterion=criterion,
    max_epoch=120,
    lr_sched=lr_sched,
    call_back_=None,
    display_freq=50,
    output_dir=TAG,
    save_every=1,
    max_keep=120
)
