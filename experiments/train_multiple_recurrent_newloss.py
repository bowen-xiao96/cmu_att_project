import os, sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, '/data2/bowenx/attention/pay_attention')

from util import Trainer_newloss as Trainer
from util.model_tools import initialize_vgg
from model.multiple_recurrent_newloss import *
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
model = MultipleRecurrentModel(network_cfg, connections, unroll_count, 1000, ('final', ))
initialize_vgg(model)

if weight_file:
    print('Loading weight file: ' + weight_file)
    state_dict = torch.load(weight_file)
    if isinstance(state_dict, tuple):
        state_dict = state_dict[-1]
    state_dict = {k.replace('features', 'backbone'): v for k, v in state_dict.items()}

    print(state_dict.keys())
    print(model.state_dict().keys())

    model.load_state_dict(state_dict, strict=False)
    del state_dict

if GPU_ID == -1:
    model = nn.DataParallel(model)
model.cuda()

#
# load dataset
#
train_loader, test_loader = get_dataloader(
    '/data2/simingy/data/Imagenet',
    52,
    8
)
max_step = len(train_loader)

#
# prepare optimizer and lr scheduler
#

vgg_params = list(model.module.backbone.parameters()) + list(model.module.classifier.parameters())
gating_params = list(model.module.gating.parameters())

optimizer = optim.Adam([
    {'params': vgg_params, 'lr': 1e-6, 'weight_decay': 1e-4},
    {'params': gating_params, 'lr': 1e-5, 'weight_decay': 1e-4},
])


def call_back(epoch, step, locals_dict, globals_dict):
    optimizer = globals_dict['optimizer_']

    if epoch == 0:
        if step <= max_step // 3:
            return
        elif step <= 2 * max_step // 3:
            lr = 3e-6
        else:
            lr = 1e-6

    elif epoch == 1 and step <= max_step // 2:
        lr = 3e-7
    else:
        lr = 1e-7

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


Trainer.start(
    model=model,
    optimizer=optimizer,
    train_dataloader=train_loader,
    test_dataloader=test_loader,
    max_epoch=3,
    lr_sched=None,
    call_back=call_back,
    display_freq=50,
    output_dir=TAG,
    save_every=1,
    max_keep=50
)
