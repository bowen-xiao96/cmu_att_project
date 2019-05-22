import os, sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, '/home/simingy/cmu_att_project/')

from utils import Trainer
from utils.model_tools import load_parallel
from utils.model_tools import initialize_vgg
from model.vgg16 import *
from model.gate import *
#from dataset.imagenet.get_imagenet_dataset import get_dataloader
sys.path.insert(0, '/home/simingy/cmu_att_project/experiments/dprime')
from get_data import get_dataloader

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

test_model = 1



#
# create model
#

model = VGG16(network_cfg, 1000)
initialize_vgg(model)
print('# parameters num:', sum(param.numel() for param in model.parameters()))
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
#train_loader, test_loader = get_dataloader(
#    '/data2/simingy/data/Imagenet',
#    32,
#    8
#)
imagenet_dir = '/data2/simingy/data/Imagenet/val'
mode = 'pytorch'
test_loader = get_dataloader(imagenet_dir, mode, 50, 8, 50)
train_loader = test_loader
max_step = len(train_loader)

criterion = nn.CrossEntropyLoss()
criterion.cuda()
init_lr = 0.1

optimizer = optim.SGD(
    model.parameters(),
    lr=init_lr,
    momentum=0.9,
    weight_decay=1e-5
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
    max_epoch=3,
    lr_sched=lr_sched,
    display_freq=50,
    output_dir=TAG,
    save_every=1,
    max_keep=50,
    test_model=1,
)
