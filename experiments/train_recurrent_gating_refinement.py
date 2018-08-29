import os, sys
import numpy as np

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, '/data2/bowenx/attention/pay_attention')

from util import Trainer
from util.model_tools import initialize_vgg
from model.recurrent_gating_refinement import *
from dataset.cifar.get_cifar10_dataset import get_dataloader

assert len(sys.argv) > 3
GPU_ID = int(sys.argv[1])
if GPU_ID != -1:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

unroll_count = int(sys.argv[2])
TAG = sys.argv[3]

print('Unrolling time step: %d' % unroll_count)

model = RecurrentGatingRefinementModel(network_cfg, unroll_count, True, 10)
initialize_vgg(model)
model.cuda()

train_loader, test_loader = get_dataloader(
    '/data2/bowenx/attention/pay_attention/dataset/cifar',
    256,
    1
)

alpha = 0.5  # control the weight between losses at final classifier and intermediate prediction
gamma = 0.5  # control the weight of losses at each time step
weights = list()
for i in range(unroll_count):
    if i == 0:
        weights.append(gamma)
    else:
        weights.append(weights[-1] * gamma)

# like (gamma**5, gamma**4, gamma**3, gamma**2, gamma)
weights = torch.FloatTensor(list(reversed(weights)))
# normalize to sum=1.0
weights /= torch.sum(weights)
print('Loss weights for different time steps:')
print(weights)
weights = A.Variable(weights.cuda(), requires_grad=False)


def criterion(pred, y):
    intermediate_pred, final_pred = pred

    # batch_size * unroll_count * num_class
    old_size = intermediate_pred.size()
    batch_size, time_step = old_size[:2]
    # flatten into (-1, num_class)
    new_size = torch.Size([batch_size * time_step] + list(old_size[2:]))
    intermediate_pred = intermediate_pred.view(new_size)

    loss_1 = F.cross_entropy(
        intermediate_pred,
        y.repeat(time_step, 1).transpose(0, 1).contiguous().view(-1),
        reduce=False
    )
    # average over batches
    loss_1 = torch.sum(loss_1 * weights.repeat(batch_size)) / batch_size

    loss_2 = F.cross_entropy(final_pred, y)

    return alpha * loss_1 + loss_2


init_lr = 0.0001

optimizer = optim.Adam(
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
