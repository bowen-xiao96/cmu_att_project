import os, sys
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as A

import Trainer
from pay_attention import initialize_vgg, get_dataloader

network_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']


class RecurrentGatingModel(nn.Module):
    def __init__(self, network_cfg, unroll_count, num_class, dropout=0.5):
        super(RecurrentGatingModel, self).__init__()
        self.unroll_count = unroll_count

        # backbone network
        self.backbone = nn.ModuleList()
        input_dim = 3
        for v in network_cfg:
            if v == 'M':
                self.backbone.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                self.backbone.append(nn.Conv2d(input_dim, v, kernel_size=3, padding=1))
                self.backbone.append(nn.ReLU(inplace=True))
                input_dim = v

        # classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(512, num_class)
        )

        # recurrent connection
        # start at conv3_1, end at conv4_3
        self.start_idx = 10
        self.end_idx = 22

        self.gating = nn.Sequential(
            nn.Conv2d(512 + 128, 128, kernel_size=1),
            nn.Sigmoid()
        )

        self.projection = nn.Conv2d(512, 128, kernel_size=1)

    def forward(self, x):
        recurrent_buf = list()

        # layers before recurrent
        for i, layer in enumerate(self.backbone):
            x = layer(x)

            if i == self.start_idx - 1:
                recurrent_buf.append(x)
            elif i == self.end_idx:
                break

        # do recurrent
        for i in range(self.unroll_count):
            prev = recurrent_buf[-1]
            x = F.upsample(x, scale_factor=2, mode='bilinear')

            # gated mixing
            gate = self.gating(torch.cat((x, prev), dim=1))
            x = gate * self.projection(x)  + (1.0 - gate) * prev

            # push result into the buffer
            recurrent_buf.append(x)

            # remaining layers
            for i in range(self.start_idx, self.end_idx + 1):
                x = self.backbone[i](x)

        for i in range(self.end_idx + 1, len(self.backbone)):
            x = self.backbone[i](x)

        x = x.view(x.size(0), -1)
        return self.classifier(x)


if __name__ == '__main__':
    assert len(sys.argv) > 1
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    TAG = sys.argv[1]

    model = RecurrentGatingModel(network_cfg, 6, 10)
    initialize_vgg(model)
    model = nn.DataParallel(model).cuda()

    train_loader, test_loader = get_dataloader(
        '/data2/bowenx/attention/pay_attention/cifar',
        256,
        4
    )

    criterion = nn.CrossEntropyLoss().cuda()
    init_lr = 0.001

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=init_lr,
        weight_decay=5e-4
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
