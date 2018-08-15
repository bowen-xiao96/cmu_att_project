import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as A

network_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']


class RecurrentGatingModel(nn.Module):
    def __init__(self, network_cfg, unroll_count, num_class, input_size=224, dropout=0.5):
        super(RecurrentGatingModel, self).__init__()
        self.unroll_count = unroll_count

        # backbone network
        self.backbone = nn.ModuleList()
        input_dim = 3
        stride = 1
        for v in network_cfg:
            if v == 'M':
                self.backbone.append(nn.MaxPool2d(kernel_size=2, stride=2))
                stride *= 2
            else:
                self.backbone.append(nn.Conv2d(input_dim, v, kernel_size=3, padding=1))
                self.backbone.append(nn.ReLU(inplace=True))
                input_dim = v

        # classifier
        feature_map_size = int(math.floor(float(input_size) / stride))
        self.classifier = nn.Sequential(
            nn.Linear(feature_map_size * feature_map_size * 512, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(1024, num_class)
        )

        # recurrent connection
        # start at conv3_1, end at conv4_3
        self.start_idx = 10
        self.end_idx = 22
        self.recurrent = nn.Sequential(
            nn.Conv2d(512 + 128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

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
            gate = self.recurrent(torch.cat((prev, x), dim=1))
            x = prev * gate
            recurrent_buf.append(x)

            # remaining layers
            for i in range(self.start_idx, self.end_idx + 1):
                x = self.backbone[i](x)

        for i in range(self.end_idx + 1, len(self.backbone)):
            x = self.backbone[i](x)

        x = x.view(x.size(0), -1)
        print(x.shape)
        return self.classifier(x)


if __name__ == '__main__':
    model = RecurrentGatingModel(network_cfg, 5, 10)
    input_sample = A.Variable(torch.zeros(1, 3, 224, 224))
    print(model(input_sample).data)
