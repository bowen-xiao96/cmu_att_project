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
from model.multiple_recurrent_l import *
from dataset.imagenet.get_imagenet_dataset import get_dataloader

#
# parse params
#
assert len(sys.argv) > 3
# gpu_id, unroll_count, tag, weight_file
GPU_ID = int(sys.argv[1])
if GPU_ID != -1:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

unroll_count = int(sys.argv[2])
TAG = sys.argv[3]

if len(sys.argv) > 4:
    weight_file = sys.argv[4]
else:
    weight_file = None

print('Unrolling time step: %d' % unroll_count)

#
# create model
#
connections = (
    (13, 8, 256, 128, 2),
    (20, 15, 512, 256, 2),
    # (27, 22, 512, 512, 2)
)
model = MultipleRecurrentModel(network_cfg, connections, unroll_count, 1000)
initialize_vgg(model)

if GPU_ID == -1:
    model = nn.DataParallel(model)

if weight_file:
    print('Loading weight file: ' + weight_file)
    loaded = torch.load(weight_file)
    if isinstance(loaded, tuple):
        loaded = loaded[-1]
    state_dict = {k.replace('features', 'backbone'): v for k, v in loaded.items()}

    print(state_dict.keys())
    print(model.state_dict().keys())

    model.load_state_dict(state_dict, strict=False)
    del loaded, state_dict

model.cuda()

#
# load dataset
#
train_loader, test_loader = get_dataloader(
    '/data2/simingy/data/Imagenet',
    48,
    8
)

#
# prepare loss function
#
gamma = 0.5
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


#
# prepare optimizer and lr scheduler
#
init_lr = 0.00001

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
    save_every=1,
    max_keep=50
)
