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
from model.multiple_recurrent_nosplit import *
from model.gate import *
from dataset.imagenet.get_imagenet_dataset import get_dataloader
#sys.path.insert(0, '/home/simingy/cmu_att_project/experiments/noise')
#from get_noise import get_dataloader
#sys.path.insert(0, '/home/simingy/cmu_att_project/experiments/dprime')
#from get_data import get_dataloader
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

print('Unrolling time step: %d' % unroll_count)

#
# create model
#
connections = (
    #(27, 20, 512, 512, 2),
    #(20, 13, 512, 256, 2),
    (13, 8, 256, 128, 2),
)
model = MultipleRecurrentModel(network_cfg, connections, unroll_count, 1000, gating_module=GatingModule8)
initialize_vgg(model)
print('# parameters num:', sum(param.numel() for param in model.parameters()))
print(model)

'''
if weight_file:
    print('Loading weight file: ' + weight_file)
    state_dict = torch.load(weight_file)
    if isinstance(state_dict, tuple):
        state_dict = state_dict[-1]
    state_dict = {k.replace('features', 'backbone'): v for k, v in state_dict.items()}

    print(state_dict.keys())
    print(model.state_dict().keys())
    net_parallel_flag = load_parallel(model.state_dict())
        
    load_parallel_flag = load_parallel(state_dict)
    if net_parallel_flag != load_parallel_flag:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
    else:
        new_state_dict = state_dict
    
    model.load_state_dict(new_state_dict)

    del new_state_dict
'''

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
    80,
    8
)
#imagenet_dir = '/data2/simingy/data/Imagenet/val/'
#mode = 'pytorch'
#test_loader = get_dataloader(imagenet_dir, mode, 50, 8, 50)
#train_loader = test_loader
max_step = len(train_loader)

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

#vgg_params = list(model.module.backbone.parameters()) + list(model.module.classifier.parameters())
#gating_params = list(model.module.gating.parameters())
vgg_params = list(model.backbone.parameters()) + list(model.classifier.parameters())
gating_params = list(model.gating.parameters())

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
    lr = 1e-7
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


Trainer.start(
    model=model,
    optimizer=optimizer,
    train_dataloader=train_loader,
    test_dataloader=test_loader,
    criterion=criterion,
    max_epoch=3,
    lr_sched=None,
    call_back=call_back,
    display_freq=50,
    output_dir=TAG,
    save_every=1,
    max_keep=50,
    #test_model=test_model
)