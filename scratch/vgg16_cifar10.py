import os, sys
import time
import math
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as A
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms

import torch.backends.cudnn

torch.backends.cudnn.benchmark = True

assert len(sys.argv) > 2
GPU_ID = int(sys.argv[1])
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)

MODEL_TAG = sys.argv[2]

# ========== config begins ==========
# optimizer
init_lr = 0.1
momentum = 0.9
weight_decay = 1e-5

# training
batch_size = 256
lr_decay_step = 25
lr_decay_gamma = 0.1
max_epoch = 50

# model
model_name = 'vgg16'
with_bn = False
num_class = 10

# misc
cifar10_dir = '/data2/bowenx/cifar'
output_dir = os.path.join('vgg16_cifar10', MODEL_TAG)
num_workers = 5
print_freq = 50
# ========== config ends ==========

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# utility functions
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = init_lr * (lr_decay_gamma ** (epoch // lr_decay_step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# define the model
vgg_cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, model_name, with_bn, num_class):
        super(VGG, self).__init__()
        self.features = self._make_layers(vgg_cfg[model_name], with_bn)

        # input is 32 x 32
        # downsampled to 1 x 1 (stride = 32)
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, num_class),
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    @staticmethod
    def _make_layers(cfg, with_bn):
        layers = list()
        in_channels = 3

        for v in cfg:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(in_channels, v, kernel_size=3, padding=1))

                if with_bn:
                    layers.append(nn.BatchNorm2d(v))

                layers.append(nn.ReLU(inplace=True))
                in_channels = v

        return nn.Sequential(*layers)


# creating dataset and dataloader
mean = np.array([0.49139968, 0.48215827, 0.44653124])
std = np.array([0.24703233, 0.24348505, 0.26158768])
normalize = transforms.Normalize(mean, std)

# during training, only random horizontal flip is used for augmentation
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

train_dataset = CIFAR10(
    cifar10_dir,
    train=True,
    transform=train_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    drop_last=True
)

test_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize
])

test_dataset = CIFAR10(
    cifar10_dir,
    train=False,
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
    drop_last=False
)

# creating model
print(MODEL_TAG)
model = VGG(model_name, with_bn, num_class)
model.cuda()
print(model)


# initialize weights
# tricky, very important
def init_params(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        nn.init.normal(m.weight.data, mean=0.0, std=math.sqrt(2.0 / n))
        nn.init.constant(m.bias.data, 0.0)

    # leave all other layers default initialized


model.apply(init_params)

criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(model.parameters(), lr=init_lr, momentum=momentum, weight_decay=weight_decay)


def train(epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()

    end = time.time()
    for i, (x, y) in enumerate(train_loader):
        data_time.update(time.time() - end)

        x = A.Variable(x.cuda())
        y = A.Variable(y.cuda())

        output = model(x)
        loss = criterion(output, y)

        # measure accuracy
        prec1, prec5 = accuracy(output.data, y.data, topk=(1, 5))
        losses.update(loss.data[0], x.size(0))
        top1.update(prec1[0], x.size(0))
        top5.update(prec5[0], x.size(0))

        # backwards
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))

    print(' * Epoch: [{0}] TRAIN * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.3f}'
          .format(epoch, top1=top1, top5=top5, loss=losses))


def test(epoch):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()

    for x, y in test_loader:
        x = A.Variable(x.cuda())
        y = A.Variable(y.cuda())

        output = model(x)
        loss = criterion(output, y)

        prec1, prec5 = accuracy(output.data, y.data, topk=(1, 5))
        losses.update(loss.data[0], x.size(0))
        top1.update(prec1[0], x.size(0))
        top5.update(prec5[0], x.size(0))

    print(' * Epoch: [{0}] TEST  * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.3f}'
          .format(epoch, top1=top1, top5=top5, loss=losses))

    return top1.avg


def main():
    for i in range(max_epoch):
        adjust_learning_rate(optimizer, i)
        train(i)
        test_acc = test(i)

        torch.save(
            (i, test_acc, model.state_dict()),  # saved data
            os.path.join(output_dir, str(i) + '.pkl'),
            pickle_protocol=pickle.HIGHEST_PROTOCOL
        )


if __name__ == '__main__':
    main()
