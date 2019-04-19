import torch
import os, sys
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as A
import numpy as np
sys.path.insert(0, '/home/simingy/cmu_att_project/utils/')
from PCA import my_PCA


network_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']


class VGG16(nn.Module):
    def __init__(self, network_cfg, num_class, last_dim=7, dropout=0.5):
        super(VGG16, self).__init__()

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

        # we use fully connected layer as classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * last_dim * last_dim, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_class)
        )

    def forward(self, x):
        for i in range(len(self.backbone)):
            x = self.backbone[i](x)
            if(i == 14):
               a = x.view(x.size(0), -1)
               a = my_PCA(a)
               #plt.figure()
               #plt.scatter(a[:,0], a[:,1])
               #plt.savefig('/home/simingy/test.png')
               np.save('/data2/simingy/pca/noise_50.npy', a.data.cpu().numpy())
       
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
