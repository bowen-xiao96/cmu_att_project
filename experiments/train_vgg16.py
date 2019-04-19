import os, sys

import torch.backends.cudnn as cudnn

cudnn.benchmark = True

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, '/home/simingy/cmu_att_project/')

from utils import Trainer
from utils.model_tools import load_parallel
from utils.model_tools import initialize_vgg
from torchvision.models import vgg16
#from dataset.cifar.get_cifar10_dataset import get_dataloader
sys.path.insert(0, '/home/simingy/cmu_att_project/experiments/noise')
from get_noise import get_dataloader


assert len(sys.argv) > 2
GPU_ID = int(sys.argv[1])
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
TAG = sys.argv[2]

model = vgg16()
print("*********************")
print(model)
'''
model.classifier = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(512, 512),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5),
    nn.Linear(512, 10)
)
initialize_vgg(model)'''

state_dict = torch.load('/home/simingy/vgg16-397923af.pth')

print(state_dict.keys())
print(model.state_dict().keys())
model.load_state_dict(state_dict)


model = nn.DataParallel(model)
model.cuda()

imagenet_dir = '/data2/simingy/data/Imagenet/'
test_loader = get_dataloader(imagenet_dir, 'pytorch', 64, 8, 0)
train_loader = test_loader

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
    max_epoch=180,
    lr_sched=lr_sched,
    display_freq=50,
    output_dir=TAG,
    save_every=5,
    max_keep=50,
    test_model=1
)
